# app/vector_search.py
"""
VectorSearch module: FAISS-only persistent index with singleton pattern,
embedding dimension normalization, and optional text chunking.
"""
from __future__ import annotations
import os
import sys
import json
import uuid
import threading
from typing import List, Dict, Any, Callable, Optional, Union

import numpy as np
import faiss
from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("vector_search")

_vector_search_singleton: Optional["VectorSearch"] = None
_singleton_lock = threading.Lock()


def get_vector_search(
    index_path: Optional[str] = None,
    embedding_fn: Optional[Callable] = None,
    dim: int = 128
):
    """Singleton accessor for VectorSearch.

    If an existing singleton uses a different index_path than requested, re-initialize it.
    This is useful for tests that want to use separate indexes.
    """
    global _vector_search_singleton
    with _singleton_lock:
        requested_path = index_path or getattr(config, "VECTOR_INDEX_PATH", "./faiss/index.faiss")
        if _vector_search_singleton is None:
            _vector_search_singleton = VectorSearch(index_path=requested_path, embedding_fn=embedding_fn, dim=dim)
            return _vector_search_singleton

        # If the caller explicitly requests a different index_path, reinitialize singleton.
        if getattr(_vector_search_singleton, "index_path", None) != requested_path:
            logger.info("get_vector_search: reinitializing singleton for new index_path %s", requested_path)
            _vector_search_singleton.clear()
            _vector_search_singleton = VectorSearch(index_path=requested_path, embedding_fn=embedding_fn, dim=dim)
        return _vector_search_singleton


class VectorSearch:
    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_fn: Optional[Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]] = None,
        dim: int = 128,
    ):
        try:
            # Allow config override
            self.index_path = index_path or getattr(config, "VECTOR_INDEX_PATH", "./faiss/index.faiss")
            # Ensure directory exists (handle case index_path is in current dir)
            index_dir = os.path.dirname(self.index_path) or "."
            os.makedirs(index_dir, exist_ok=True)
            self.meta_path = f"{self.index_path}.meta.json"

            self._io_lock = threading.RLock()
            self._embedding_fn_user = embedding_fn
            self.dim = int(dim)
            self._init_embedding_fn(self.dim)

            self._faiss_index: Optional[faiss.IndexFlatIP] = None
            self._metadata: Dict[str, Dict[str, Any]] = {}
            self._ids_list: List[str] = []

            # Load persisted state if present
            self._load_meta()
            self._load_index()

            logger.info(
                "VectorSearch initialized at %s (FAISS dim=%d, entries=%d)",
                self.index_path, self.dim, len(self._ids_list)
            )
        except Exception as e:
            logger.exception("Failed to initialize VectorSearch")
            raise CustomException(e, sys)

    # ---------------------------
    # Embeddings
    # ---------------------------
    def _init_embedding_fn(self, preferred_dim: int) -> None:
        if self._embedding_fn_user:
            # If user supplied embedding function, wrap it to ensure shape & normalization
            self.embedding_fn = self._wrap_embedding_fn(self._embedding_fn_user)
            self.dim = preferred_dim
        else:
            logger.warning("Using deterministic fallback embedding (dim=%d)", preferred_dim)
            self.embedding_fn = self._wrap_embedding_fn(self._default_embedding)
            self.dim = preferred_dim

    def _wrap_embedding_fn(self, fn: Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]):
        """
        Wrap user-provided embedding fn to accept either string or list and
        ensure each returned vector is numeric length self.dim and normalized.
        """
        def wrapper(x: Union[str, List[str]]):
            # If list input: produce list-of-lists
            if isinstance(x, (list, tuple)):
                out = []
                # allow fn to accept list directly or require single strings
                try:
                    res = fn(x)
                    # If function returned a flat list for a single string, handle below
                    if isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], (list, tuple, np.ndarray)):
                        # res is already list of vectors
                        for v in res:
                            arr = np.asarray(v, dtype=np.float32)
                            out.append(self._adjust_and_normalize(arr).tolist())
                        return out
                    # Otherwise, fall back to per-item call
                except Exception:
                    res = None

                # Per-item call fallback
                for item in x:
                    r = fn(item)
                    arr = np.asarray(r if r is not None else [], dtype=np.float32)
                    out.append(self._adjust_and_normalize(arr).tolist())
                return out
            else:
                r = fn(x)
                arr = np.asarray(r if r is not None else [], dtype=np.float32)
                return self._adjust_and_normalize(arr).tolist()
        return wrapper

    def _default_embedding(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for i, c in enumerate(str(text or "")):
            vec[i % self.dim] += (ord(c) % 97) * 0.001
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _adjust_and_normalize(self, vec: np.ndarray) -> np.ndarray:
        """Pad or trim to self.dim and return normalized float32 vector."""
        if vec is None:
            vec = np.zeros(self.dim, dtype=np.float32)
        if vec.size < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[:vec.size] = vec
            vec = padded
        elif vec.size > self.dim:
            vec = vec[: self.dim]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    # ---------------------------
    # Persistence
    # ---------------------------
    def _load_meta(self):
        with self._io_lock:
            if os.path.exists(self.meta_path):
                try:
                    with open(self.meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._ids_list = data.get("ids", []) or []
                    self._metadata = data.get("metadata", {}) or {}
                    # allow meta to override dim (useful if index was created with different dim)
                    meta_dim = int(data.get("dim", self.dim))
                    if meta_dim != self.dim:
                        logger.warning("Meta dim (%d) differs from requested dim (%d); using meta dim", meta_dim, self.dim)
                    self.dim = meta_dim
                except Exception:
                    logger.exception("Failed to load meta JSON; starting fresh meta")
                    self._ids_list = []
                    self._metadata = {}

    def _save_meta(self):
        with self._io_lock:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump({"ids": self._ids_list, "metadata": self._metadata, "dim": self.dim}, f, indent=2)

    def _load_index(self):
        with self._io_lock:
            if os.path.exists(self.index_path):
                try:
                    idx = faiss.read_index(self.index_path)
                    if idx.d != self.dim:
                        logger.warning("FAISS index dim mismatch (index=%d vs requested=%d); reinitializing empty index", idx.d, self.dim)
                        # reinit empty index with our dim (we keep metadata but index is empty)
                        self._faiss_index = faiss.IndexFlatIP(self.dim)
                    else:
                        self._faiss_index = idx
                except Exception:
                    logger.exception("Failed to read FAISS index; creating a new one")
                    self._faiss_index = faiss.IndexFlatIP(self.dim)
            else:
                # Fresh index
                self._faiss_index = faiss.IndexFlatIP(self.dim)

    def _save_index(self):
        with self._io_lock:
            if self._faiss_index is not None:
                faiss.write_index(self._faiss_index, self.index_path)

    # ---------------------------
    # Upsert & Search
    # ---------------------------
    def upsert_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Documents: list of {"id": <opt>, "text": <str>, "meta": <dict>}
        Returns list of inserted ids (in same order).
        """
        with self._io_lock:
            ids_added = []
            texts = []
            metas = []

            for doc in documents:
                doc_id = doc.get("id") or uuid.uuid4().hex
                text = doc.get("text") or ""
                meta = doc.get("meta", {}) or {}
                ids_added.append(doc_id)
                texts.append(text)
                metas.append((doc_id, meta, text))

            if not texts:
                return []

            # Compute embeddings (list of lists)
            emb = self._embed_texts(texts)  # numpy array shape (n, dim)
            if emb.ndim == 1:
                emb = np.expand_dims(emb, 0)
            emb = emb.astype(np.float32)

            # Add to FAISS index (index keeps insertion order, indices map to _ids_list)
            self._faiss_index.add(emb)

            # Append metadata and ids in insertion order
            for doc_id, meta, text in metas:
                self._metadata[doc_id] = {"text": text, **meta}
                self._ids_list.append(doc_id)

            # Persist
            try:
                self._save_index()
                self._save_meta()
            except Exception:
                logger.exception("Failed persisting index/meta after upsert (continuing)")

            return ids_added

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Ensure returned shape is (n, dim) numpy.ndarray float32.
        The wrapped embedding_fn supports list input where possible.
        """
        res = self.embedding_fn(texts)
        arr = np.asarray(res, dtype=np.float32)
        # If embedding function returned single vector for the whole list, reshape appropriately
        if arr.ndim == 1 and len(texts) > 1:
            # fallback: call per item
            per = [np.asarray(self.embedding_fn(t), dtype=np.float32) for t in texts]
            arr = np.vstack([self._adjust_and_normalize(a) for a in per])
        # ensure proper shape (n, dim)
        if arr.ndim == 1:
            arr = np.expand_dims(self._adjust_and_normalize(arr), 0)
        else:
            arr = np.vstack([self._adjust_and_normalize(a) for a in arr])
        return arr

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Return top_k results as list of {"id","score","text","meta"} ordered by decreasing score.
        """
        q_vec = np.asarray(self.embedding_fn(query), dtype=np.float32)
        q_vec = self._adjust_and_normalize(q_vec)

        results: List[Dict[str, Any]] = []
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []

        # Perform search
        D, I = self._faiss_index.search(np.expand_dims(q_vec, 0), top_k)
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self._ids_list):
                continue
            doc_id = self._ids_list[idx]
            md = self._metadata.get(doc_id, {})
            results.append({"id": doc_id, "score": float(score), "text": md.get("text"), "meta": md})
        return results

    # ---------------------------
    # Utilities
    # ---------------------------
    def clear(self):
        with self._io_lock:
            self._metadata = {}
            self._ids_list = []
            self._faiss_index = faiss.IndexFlatIP(self.dim)
            try:
                if os.path.exists(self.index_path):
                    os.remove(self.index_path)
                if os.path.exists(self.meta_path):
                    os.remove(self.meta_path)
            except Exception:
                logger.exception("Failed to remove index/meta files during clear (ignored)")

    def info(self) -> Dict[str, Any]:
        with self._io_lock:
            return {
                "index_path": self.index_path,
                "meta_path": self.meta_path,
                "dim": self.dim,
                "entries": len(self._ids_list),
            }

    def list_ids(self) -> List[str]:
        return self._ids_list.copy()
