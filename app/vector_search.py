"""
app/vector_search.py

FAISS-backed VectorSearch with:
 - singleton accessor (get_vector_search)
 - persistent FAISS index + JSON metadata (.meta.json)
 - deterministic fallback embedding when none provided
 - improved logging, public helpers and clearer exceptions

Notes:
 - Removing single vectors from an IndexFlatIP is not cheap; remove_vector raises
   NotImplementedError and suggests a rebuild approach if you actually need deletion.
 - The internal metadata stores "text" plus user-provided meta merged in.
"""
from __future__ import annotations
import os
import sys
import json
import uuid
import threading
from typing import List, Dict, Any, Callable, Optional, Union

import numpy as np

try:
    import faiss
except Exception as e:
    # allow module import even if faiss import fails at runtime; initialization will raise later
    faiss = None  # type: ignore

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("vector_search")

_vector_search_singleton: Optional["VectorSearch"] = None
_singleton_lock = threading.Lock()


def get_vector_search(
    index_path: Optional[str] = None,
    embedding_fn: Optional[Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]] = None,
    dim: int = 128,
) -> "VectorSearch":
    """
    Return a singleton VectorSearch instance. If index_path differs from the
    existing singleton's path, the singleton is re-initialized for the new path.
    """
    global _vector_search_singleton
    with _singleton_lock:
        requested_path = index_path or getattr(config, "VECTOR_INDEX_PATH", "./faiss/index.faiss")
        if _vector_search_singleton is None:
            _vector_search_singleton = VectorSearch(index_path=requested_path, embedding_fn=embedding_fn, dim=dim)
            return _vector_search_singleton

        # Reinitialize if different index path requested
        if getattr(_vector_search_singleton, "index_path", None) != requested_path:
            logger.info("get_vector_search: reinitializing singleton for new index_path=%s", requested_path)
            try:
                _vector_search_singleton.clear()
            except Exception:
                logger.exception("get_vector_search: failed to clear previous singleton (continuing)")
            _vector_search_singleton = VectorSearch(index_path=requested_path, embedding_fn=embedding_fn, dim=dim)
        else:
            # If embedding_fn provided, replace wrapper so embedding behavior follows caller expectation
            if embedding_fn is not None:
                _vector_search_singleton._embedding_fn_user = embedding_fn
                _vector_search_singleton._init_embedding_fn(dim)
        return _vector_search_singleton


class VectorSearch:
    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_fn: Optional[Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]] = None,
        dim: int = 128,
    ):
        """
        Create or load a FAISS index stored at index_path and associated metadata JSON.

        index_path: path to faiss file (disk)
        embedding_fn: callable(text) -> vector or callable([texts]) -> [vectors]
        dim: embedding dimensionality (integer)
        """
        try:
            self.index_path = index_path or getattr(config, "VECTOR_INDEX_PATH", "./faiss/index.faiss")
            index_dir = os.path.dirname(self.index_path) or "."
            os.makedirs(index_dir, exist_ok=True)
            self.meta_path = f"{self.index_path}.meta.json"

            self._io_lock = threading.RLock()
            self._embedding_fn_user = embedding_fn
            self.dim = int(dim)

            # initialize embedding wrapper (may fallback to deterministic)
            self._init_embedding_fn(self.dim)

            # internal state
            self._faiss_index: Optional["faiss.Index"] = None
            self._metadata: Dict[str, Dict[str, Any]] = {}
            self._ids_list: List[str] = []

            # load persisted state
            self._load_meta()
            self._load_index()

            logger.info("VectorSearch initialized: index=%s dim=%d entries=%d", self.index_path, self.dim, len(self._ids_list))
        except Exception as exc:
            logger.exception("VectorSearch.__init__ failed")
            raise CustomException(exc, sys)

    # ---------------------------
    # Embedding helpers
    # ---------------------------
    def _init_embedding_fn(self, preferred_dim: int) -> None:
        """
        Wrap the user's embedding function (or fallback) so that returned vectors
        are ensured to be length self.dim (pad/trim) and normalized.
        """
        if self._embedding_fn_user:
            self.embedding_fn = self._wrap_embedding_fn(self._embedding_fn_user)
            self.dim = preferred_dim
            logger.debug("Using user-supplied embedding function (dim=%d)", self.dim)
        else:
            logger.warning("Using deterministic fallback embedding (dim=%d)", preferred_dim)
            self.embedding_fn = self._wrap_embedding_fn(self._default_embedding)
            self.dim = preferred_dim

    def _wrap_embedding_fn(self, fn: Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]):
        """
        Returns a wrapper that accepts a single string or list of strings and
        guarantees float32 normalized vectors of length self.dim.
        """
        def wrapper(x: Union[str, List[str]]):
            if isinstance(x, (list, tuple)):
                # try bulk call first
                try:
                    res = fn(x)
                    # If res is list-of-vectors, normalize each
                    if isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], (list, tuple, np.ndarray)):
                        out = []
                        for v in res:
                            arr = np.asarray(v, dtype=np.float32)
                            out.append(self._adjust_and_normalize(arr).tolist())
                        return out
                except Exception:
                    logger.debug("Bulk embedding call failed (falling back to per-item)", exc_info=True)
                # per-item fallback
                out = []
                for item in x:
                    try:
                        r = fn(item)
                    except Exception:
                        logger.exception("Embedding function failed for item; using zero vector", exc_info=True)
                        r = None
                    arr = np.asarray(r if r is not None else [], dtype=np.float32)
                    out.append(self._adjust_and_normalize(arr).tolist())
                return out
            else:
                try:
                    r = fn(x)
                except Exception:
                    logger.exception("Embedding function failed for single input; using zero vector", exc_info=True)
                    r = None
                arr = np.asarray(r if r is not None else [], dtype=np.float32)
                return self._adjust_and_normalize(arr).tolist()
        return wrapper

    def _default_embedding(self, text: str) -> List[float]:
        """Deterministic fallback embedding (cheap and stable)."""
        vec = np.zeros(self.dim, dtype=np.float32)
        for i, c in enumerate(str(text or "")):
            vec[i % self.dim] += (ord(c) % 97) * 0.001
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _adjust_and_normalize(self, vec: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Ensure a vector has length self.dim: pad with zeros if shorter,
        trim if longer, and normalize to unit length (float32).
        """
        if not isinstance(vec, np.ndarray):
            vec = np.asarray(vec if vec is not None else [], dtype=np.float32)
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
    # Persistence helpers
    # ---------------------------
    def _load_meta(self) -> None:
        with self._io_lock:
            if os.path.exists(self.meta_path):
                try:
                    with open(self.meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._ids_list = data.get("ids", []) or []
                    self._metadata = data.get("metadata", {}) or {}
                    meta_dim = int(data.get("dim", self.dim))
                    if meta_dim != self.dim:
                        logger.warning("Meta dim %d differs from requested dim %d; using meta dim", meta_dim, self.dim)
                    self.dim = meta_dim
                    logger.debug("Loaded metadata: ids=%d", len(self._ids_list))
                except Exception:
                    logger.exception("Failed to load meta json; starting fresh metadata")
                    self._ids_list = []
                    self._metadata = {}

    def _save_meta(self) -> None:
        with self._io_lock:
            try:
                with open(self.meta_path, "w", encoding="utf-8") as f:
                    json.dump({"ids": self._ids_list, "metadata": self._metadata, "dim": self.dim}, f, indent=2)
                logger.debug("Saved meta to %s (ids=%d)", self.meta_path, len(self._ids_list))
            except Exception:
                logger.exception("Failed to save meta JSON")

    def _load_index(self) -> None:
        with self._io_lock:
            # ensure FAISS available
            if faiss is None:
                raise CustomException(RuntimeError("faiss module not available; install faiss to use VectorSearch"), sys)
            if os.path.exists(self.index_path):
                try:
                    idx = faiss.read_index(self.index_path)
                    if idx.d != self.dim:
                        logger.warning("FAISS index dim mismatch (index=%d vs requested=%d). Reinitializing empty index.", idx.d, self.dim)
                        self._faiss_index = faiss.IndexFlatIP(self.dim)
                    else:
                        self._faiss_index = idx
                    logger.debug("Loaded FAISS index from %s (ntotal=%d)", self.index_path, getattr(self._faiss_index, "ntotal", 0))
                except Exception:
                    logger.exception("Failed to read FAISS index; creating a new empty index")
                    self._faiss_index = faiss.IndexFlatIP(self.dim)
            else:
                # fresh index
                self._faiss_index = faiss.IndexFlatIP(self.dim)
                logger.debug("Created new FAISS IndexFlatIP (dim=%d)", self.dim)

    def _save_index(self) -> None:
        with self._io_lock:
            try:
                if self._faiss_index is not None:
                    faiss.write_index(self._faiss_index, self.index_path)
                    logger.debug("Wrote FAISS index to %s (ntotal=%d)", self.index_path, getattr(self._faiss_index, "ntotal", 0))
            except Exception:
                logger.exception("Failed to write FAISS index to disk")

    # ---------------------------
    # Public helpers
    # ---------------------------
    def save_index(self) -> None:
        """Public wrapper to persist index + meta to disk (safe to call)."""
        with self._io_lock:
            self._save_index()
            self._save_meta()
            logger.info("Index and metadata persisted to disk (index=%s)", self.index_path)

    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return stored metadata for a document id, or None if not found."""
        with self._io_lock:
            return self._metadata.get(doc_id)

    def info(self) -> Dict[str, Any]:
        """Return summary info about index and metadata."""
        with self._io_lock:
            return {"index_path": self.index_path, "meta_path": self.meta_path, "dim": self.dim, "entries": len(self._ids_list)}

    def get_info(self) -> Dict[str, Any]:
        """Alias for info()."""
        return self.info()

    # ---------------------------
    # Insertion & search
    # ---------------------------
    def add_vector(self, doc_id: str, vector: Union[List[float], np.ndarray], meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a single vector with id and metadata.
        Vector is padded/trimmed and normalized to self.dim.
        """
        with self._io_lock:
            if self._faiss_index is None:
                raise CustomException(RuntimeError("FAISS index not initialized"), sys)
            vec = np.asarray(vector if vector is not None else [], dtype=np.float32)
            vec_adj = self._adjust_and_normalize(vec)
            try:
                self._faiss_index.add(np.expand_dims(vec_adj, 0))
                # append metadata + id
                self._ids_list.append(doc_id)
                merged_meta = {"text": (meta or {}).get("text") if isinstance(meta, dict) else None}
                if isinstance(meta, dict):
                    merged_meta.update(meta)
                # ensure textual copy exists
                if merged_meta.get("text") is None:
                    merged_meta["text"] = ""
                self._metadata[doc_id] = merged_meta
                # persist
                try:
                    self._save_index()
                    self._save_meta()
                except Exception:
                    logger.exception("add_vector: failed to persist index/meta (continuing)")
                logger.debug("Added vector id=%s (index_entries=%d)", doc_id, len(self._ids_list))
            except Exception:
                logger.exception("add_vector: failed to add vector")
                raise

    def upsert_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Insert a list of documents. Each document should be {"id": <opt>, "text": <str>, "meta": <dict>}.
        Returns list of IDs inserted (in the same order).
        """
        with self._io_lock:
            ids_added: List[str] = []
            texts: List[str] = []
            metas: List[Dict[str, Any]] = []
            for doc in documents:
                doc_id = doc.get("id") or uuid.uuid4().hex
                text = doc.get("text") or ""
                meta = doc.get("meta", {}) or {}
                ids_added.append(doc_id)
                texts.append(text)
                metas.append({"id": doc_id, "meta": meta, "text": text})

            if not texts:
                logger.debug("upsert_documents called with empty documents list")
                return []

            emb = self._embed_texts(texts)  # numpy array (n, dim)
            if emb.ndim == 1:
                emb = np.expand_dims(emb, 0)
            emb = emb.astype(np.float32)

            try:
                self._faiss_index.add(emb)
            except Exception:
                logger.exception("upsert_documents: FAISS add failed")
                raise

            # update metadata and id list
            for m in metas:
                self._metadata[m["id"]] = {"text": m["text"], **(m["meta"] or {})}
                self._ids_list.append(m["id"])

            # persist
            try:
                self._save_index()
                self._save_meta()
            except Exception:
                logger.exception("upsert_documents: failed to persist index/meta (continuing)")

            logger.info("upsert_documents: inserted %d docs (total=%d)", len(ids_added), len(self._ids_list))
            return ids_added

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Call the wrapped embedding function and coerce result to an (n, dim) float32 numpy array.
        """
        res = None
        try:
            res = self.embedding_fn(texts)
        except Exception:
            logger.exception("Embedding function failed during _embed_texts; attempting per-item fallback")
            per = [self.embedding_fn(t) for t in texts]
            res = per

        arr = np.asarray(res, dtype=np.float32)
        # If embedding function returned single vector for the whole list, call per-item fallback
        if arr.ndim == 1 and len(texts) > 1:
            per = [np.asarray(self.embedding_fn(t), dtype=np.float32) for t in texts]
            arr = np.vstack([self._adjust_and_normalize(a) for a in per])
        # ensure proper shape
        if arr.ndim == 1:
            arr = np.expand_dims(self._adjust_and_normalize(arr), 0)
        else:
            arr = np.vstack([self._adjust_and_normalize(a) for a in arr])
        return arr

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search by text query. Returns list of dicts: {"id","score","text","meta"}.
        """
        with self._io_lock:
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                logger.debug("search: index empty or not initialized")
                return []
            try:
                q_vec = np.asarray(self.embedding_fn(query), dtype=np.float32)
                q_vec = self._adjust_and_normalize(q_vec)
            except Exception:
                logger.exception("search: embedding for query failed")
                return []

            try:
                D, I = self._faiss_index.search(np.expand_dims(q_vec, 0), top_k)
            except Exception:
                logger.exception("search: faiss.search failed")
                return []

            results: List[Dict[str, Any]] = []
            for score, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self._ids_list):
                    continue
                doc_id = self._ids_list[idx]
                md = self._metadata.get(doc_id, {})
                results.append({"id": doc_id, "score": float(score), "text": md.get("text"), "meta": md})
            logger.debug("search: query=%r returned %d results", query, len(results))
            return results

    # ---------------------------
    # Utilities
    # ---------------------------
    def list_ids(self) -> List[str]:
        with self._io_lock:
            return list(self._ids_list)

    def remove_index_files(self) -> None:
        """Remove persisted faiss + meta files from disk (does not change in-memory index)."""
        with self._io_lock:
            try:
                if os.path.exists(self.index_path):
                    os.remove(self.index_path)
                    logger.info("Removed faiss index file %s", self.index_path)
                if os.path.exists(self.meta_path):
                    os.remove(self.meta_path)
                    logger.info("Removed meta file %s", self.meta_path)
            except Exception:
                logger.exception("remove_index_files: failed to remove files")

    def clear(self) -> None:
        """
        Clear in-memory state and delete persisted files.
        Note: this resets the IndexFlatIP in-memory and deletes disk files.
        """
        with self._io_lock:
            self._metadata = {}
            self._ids_list = []
            if faiss is None:
                raise CustomException(RuntimeError("faiss module not available"), sys)
            self._faiss_index = faiss.IndexFlatIP(self.dim)
            try:
                if os.path.exists(self.index_path):
                    os.remove(self.index_path)
                if os.path.exists(self.meta_path):
                    os.remove(self.meta_path)
                logger.info("VectorSearch: cleared in-memory and removed persisted files")
            except Exception:
                logger.exception("clear: failed to remove persisted files (ignored)")

    def remove_vector(self, doc_id: str) -> None:
        """
        Removing one vector from IndexFlatIP requires rebuilding the index (FAISS doesn't support deletions).
        We don't store embeddings on disk; therefore deletion is not implemented here.

        Suggested approach if you need deletion:
          1. Export all vectors (or re-embed original texts if available)
          2. Filter out the doc_id
          3. Rebuild a new FAISS index and overwrite index+meta

        This method intentionally raises NotImplementedError to avoid silent incorrect behavior.
        """
        raise NotImplementedError(
            "remove_vector is not implemented for IndexFlatIP. "
            "Rebuild the index excluding the unwanted id (see docstring for steps)."
        )
