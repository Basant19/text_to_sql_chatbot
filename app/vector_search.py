# app/vector_search.py
"""
Improved VectorSearch module.

Key improvements vs older version:
 - Singleton accessor (get_vector_search) to avoid repeated init.
 - Lazy embedding client initialization (prefers langchain_google_genai if available).
 - Stable embedding dimension detection and consistent resizing/padding.
 - FAISS-backed store (if available) with safe persistence; NumPy fallback otherwise.
 - Deterministic metadata ordering (ids list) so vector index positions are stable.
 - Thread-safe load/save operations.
 - Cleaner error handling and logging.
"""
from __future__ import annotations
import os
import sys
import json
import uuid
import threading
from typing import List, Dict, Any, Callable, Optional, Union, Tuple

import numpy as np

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("vector_search")

# ---------------------------
# Optional dependencies
# ---------------------------
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False
    logger.info("FAISS not available; using NumPy fallback for vector search.")

# Try multiple text-splitter import locations
_USE_TEXT_SPLITTER = False
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    _USE_TEXT_SPLITTER = True
    logger.info("Using langchain_text_splitters.RecursiveCharacterTextSplitter for chunking")
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
        _USE_TEXT_SPLITTER = True
        logger.info("Using langchain.text_splitter.RecursiveCharacterTextSplitter for chunking")
    except Exception:
        _USE_TEXT_SPLITTER = False
        logger.debug("LangChain text splitter unavailable; chunking disabled.")

# Embedding provider detection (do not import at module top-level heavy objects)
_EMB_PROVIDER = None
try:
    # prefer langchain_google_genai name when present
    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
    _EMB_PROVIDER = "langchain_google_genai"
    logger.info("langchain_google_genai.GoogleGenerativeAIEmbeddings detected")
except Exception:
    try:
        from langchain.embeddings import GoogleGenerativeAIEmbeddings  # type: ignore
        _EMB_PROVIDER = "langchain.langchain"
        logger.info("langchain.embeddings.GoogleGenerativeAIEmbeddings detected")
    except Exception:
        _EMB_PROVIDER = None
        logger.info("No GoogleGenerativeAIEmbeddings provider found; will use fallback embedding")


# ---------------------------
# Helpers
# ---------------------------
def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _safe_json_load(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _safe_json_dump(path: str, data: Any) -> None:
    _ensure_parent_dir(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _remove_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ---------------------------
# Module-level singleton + lock
# ---------------------------
_vector_search_singleton = None
_singleton_lock = threading.Lock()


def get_vector_search(index_path: Optional[str] = None, embedding_fn: Optional[Callable] = None, dim: int = 128):
    """
    Retrieve a single shared VectorSearch instance (singleton) for the process.
    Callers that need multiple independent indexes can instantiate VectorSearch() directly.
    """
    global _vector_search_singleton
    with _singleton_lock:
        if _vector_search_singleton is None:
            _vector_search_singleton = VectorSearch(index_path=index_path, embedding_fn=embedding_fn, dim=dim)
        return _vector_search_singleton


# ---------------------------
# VectorSearch class
# ---------------------------
class VectorSearch:
    """
    Vector search wrapper supporting:
      - FAISS or NumPy backend
      - Upserting documents with 'text' and optional metadata
      - Optional chunking via RecursiveCharacterTextSplitter
      - Optional GoogleGenerativeAIEmbeddings via LangChain
      - Persistent index & metadata (.meta.json + faiss file / .npy)
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_fn: Optional[Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]] = None,
        dim: int = 128,
    ):
        try:
            # config-aware defaults
            self.index_path = index_path or getattr(config, "VECTOR_INDEX_PATH", "./faiss/index.faiss")
            self.meta_path = f"{self.index_path}.meta.json"
            _ensure_parent_dir(self.index_path)
            _ensure_parent_dir(self.meta_path)

            # chunking params
            try:
                self.chunk_size = int(getattr(config, "VECTOR_CHUNK_SIZE", 1000))
                self.chunk_overlap = int(getattr(config, "VECTOR_CHUNK_OVERLAP", 200))
            except Exception:
                self.chunk_size = 1000
                self.chunk_overlap = 200

            # concurrency lock for index/meta persistence
            self._io_lock = threading.RLock()

            # embedding function initialization (lazy provider support)
            self._embedding_fn_user = embedding_fn
            self._embedding_client = None
            self._init_embedding_fn(dim)

            # storage/backing
            self._use_faiss = _FAISS_AVAILABLE
            self._faiss_index = None
            self._numpy_vectors: Optional[np.ndarray] = None

            # metadata and stable ordering (ids_list keeps vector order)
            self._metadata: Dict[str, Dict[str, Any]] = {}
            self._ids_list: List[str] = []
            self.dim = int(dim)  # target dimension; may be adjusted after provider inference

            # load existing state (meta must be loaded before index so we know ordering)
            self._load_meta()
            self._load_index()

            logger.info("VectorSearch initialized at %s (FAISS=%s, dim=%d, entries=%d)",
                        self.index_path, self._use_faiss, self.dim, len(self._ids_list))
        except Exception as e:
            logger.exception("Failed to initialize VectorSearch")
            raise CustomException(e, sys)

    # ---------------------------
    # Embedding initialization
    # ---------------------------
    def _init_embedding_fn(self, preferred_dim: int) -> None:
        """
        Prepare self.embedding_fn that accepts either str or list[str] and returns normalized vectors.
        If a langchain provider is available, instantiate it lazily and infer dimension.
        """
        # If user supplied a raw callable, wrap it
        if self._embedding_fn_user is not None:
            self.embedding_fn = self._wrap_embedding_fn(self._embedding_fn_user)
            self.dim = int(preferred_dim)
            return

        # Try provider
        if _EMB_PROVIDER is not None:
            try:
                if _EMB_PROVIDER == "langchain_google_genai":
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
                    emb_client = GoogleGenerativeAIEmbeddings(model=getattr(config, "EMBEDDING_MODEL", "models/gemini-embedding-001"))
                else:
                    from langchain.embeddings import GoogleGenerativeAIEmbeddings  # type: ignore
                    emb_client = GoogleGenerativeAIEmbeddings(model=getattr(config, "EMBEDDING_MODEL", "models/gemini-embedding-001"))

                logger.info("VectorSearch: initialized embedding client via %s", _EMB_PROVIDER)

                def _emb_fn(x: Union[str, List[str]]):
                    # provider methods may be named embed_documents / embed_query
                    if isinstance(x, (list, tuple)):
                        # embed_documents -> returns list[list[float]]
                        try:
                            out = emb_client.embed_documents(list(x))
                        except Exception:
                            out = [emb_client.embed_query(t) for t in x]
                        return [list(map(float, v)) for v in out]
                    else:
                        try:
                            out = emb_client.embed_query(x)
                        except Exception:
                            out = emb_client.embed_documents([x])[0]
                        return list(map(float, out))

                # temporarily call with a sample to infer dim
                sample = _emb_fn("hello world")
                inferred_dim = len(sample) if isinstance(sample, (list, tuple)) else preferred_dim
                self.dim = int(inferred_dim)
                self.embedding_fn = self._wrap_embedding_fn(_emb_fn)
                return
            except Exception as e:
                logger.warning("Failed to initialize provider embeddings (%s) — falling back: %s", _EMB_PROVIDER, e)

        # Default deterministic fallback embedding (wrapped)
        logger.warning("Using deterministic fallback embedding (dim=%d)", preferred_dim)
        self.dim = int(preferred_dim)
        self.embedding_fn = self._wrap_embedding_fn(self._default_embedding)

    def _wrap_embedding_fn(self, fn: Callable[[str], List[float]]) -> Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]:
        """
        Wrap a function that may accept a single string into one that accepts either a single
        string or a list of strings. The wrapper also ensures numeric type and proper vector dim.
        """
        def wrapper(x: Union[str, List[str]]):
            if isinstance(x, (list, tuple)):
                out_vectors = []
                for t in x:
                    try:
                        v = fn(t)
                        v_list = list(map(float, v))
                    except Exception:
                        v_list = [0.0] * self.dim
                    # adjust & normalize
                    v_arr = np.array(v_list, dtype=np.float32)
                    v_arr = self._adjust_dim(v_arr)
                    v_arr = self._normalize_vector(v_arr)
                    out_vectors.append(v_arr.tolist())
                return out_vectors
            else:
                try:
                    v = fn(x)
                    v_list = list(map(float, v))
                except Exception:
                    v_list = [0.0] * self.dim
                v_arr = np.array(v_list, dtype=np.float32)
                v_arr = self._adjust_dim(v_arr)
                v_arr = self._normalize_vector(v_arr)
                return v_arr.tolist()
        return wrapper

    # ---------------------------
    # Default deterministic embedding (single text only)
    # ---------------------------
    def _default_embedding(self, text: str) -> List[float]:
        """
        Deterministic fallback embedding for single text.
        """
        d = self.dim or 128
        vec = np.zeros(d, dtype=np.float32)
        s = str(text or "")
        for i, ch in enumerate(s):
            vec[i % d] += (ord(ch) % 97) * 0.001
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    # ---------------------------
    # Vector helpers
    # ---------------------------
    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _adjust_dim(self, vec: np.ndarray) -> np.ndarray:
        """
        Ensure vector is exactly self.dim: pad with zeros or truncate.
        """
        if vec.size == self.dim:
            return vec.astype(np.float32)
        if vec.size < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[: vec.size] = vec.astype(np.float32)
            return padded
        return vec[: self.dim].astype(np.float32)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Given list[str] -> return numpy array shape (len(texts), dim)
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        try:
            res = self.embedding_fn(texts)
            if isinstance(res, (list, tuple)) and res and isinstance(res[0], (list, tuple, np.ndarray)):
                arr = np.array([np.array(r, dtype=np.float32) for r in res], dtype=np.float32)
            else:
                # fallback: call per item
                vecs = []
                for t in texts:
                    v = self.embedding_fn(t)
                    vecs.append(np.array(v, dtype=np.float32))
                arr = np.vstack(vecs).astype(np.float32)
        except Exception:
            # last-resort per-item safe calls
            vecs = []
            for t in texts:
                try:
                    v = self.embedding_fn(t)
                    vecs.append(np.array(v, dtype=np.float32))
                except Exception:
                    vecs.append(np.zeros(self.dim, dtype=np.float32))
            arr = np.vstack(vecs).astype(np.float32)

        # adjust dims & normalize
        adjusted = []
        for v in arr:
            v = self._adjust_dim(v)
            v = self._normalize_vector(v)
            adjusted.append(v)
        return np.vstack(adjusted).astype(np.float32)

    # ---------------------------
    # Persistence helpers (meta + index)
    # ---------------------------
    def _load_meta(self) -> None:
        with self._io_lock:
            try:
                if os.path.exists(self.meta_path):
                    data = _safe_json_load(self.meta_path) or {}
                    # Expect structure: {"ids": [...], "metadata": {id: {...}}, "dim": N}
                    ids = data.get("ids", [])
                    metadata = data.get("metadata", {})
                    self._ids_list = list(ids)
                    self._metadata = dict(metadata)
                    # If dim stored, adopt it (but don't override a provider-inferred dim if present)
                    stored_dim = data.get("dim")
                    if stored_dim and (not hasattr(self, "dim") or int(stored_dim) != int(self.dim)):
                        try:
                            self.dim = int(stored_dim)
                        except Exception:
                            pass
                else:
                    self._ids_list = []
                    self._metadata = {}
            except Exception as e:
                logger.exception("Failed to load metadata")
                raise CustomException(e, sys)

    def _save_meta(self) -> None:
        with self._io_lock:
            try:
                data = {"ids": self._ids_list, "metadata": self._metadata, "dim": int(self.dim)}
                _safe_json_dump(self.meta_path, data)
            except Exception as e:
                logger.exception("Failed to save metadata")
                raise CustomException(e, sys)

    def _load_index(self) -> None:
        with self._io_lock:
            try:
                # If FAISS enabled
                if self._use_faiss:
                    if os.path.exists(self.index_path):
                        self._faiss_index = faiss.read_index(self.index_path)
                        # If faiss index dimension doesn't match self.dim, re-create (can't easily resize)
                        try:
                            idx_d = self._faiss_index.d
                            if idx_d != self.dim:
                                logger.warning("FAISS index dim (%d) != expected dim (%d); recreating empty index", idx_d, self.dim)
                                self._faiss_index = faiss.IndexFlatIP(self.dim)
                        except Exception:
                            # fallback to create newly
                            self._faiss_index = faiss.IndexFlatIP(self.dim)
                        logger.info("Loaded faiss index from disk")
                    else:
                        self._faiss_index = faiss.IndexFlatIP(self.dim)
                        logger.info("Created new faiss index (empty)")
                        # if there is existing metadata + vector files stored as numpy, try to load them
                        npy_path = f"{self.index_path}.npy"
                        if os.path.exists(npy_path):
                            try:
                                vecs = np.load(npy_path)
                                if vecs.shape[1] != self.dim:
                                    # attempt to adjust; pad/truncate cols
                                    adj = np.zeros((vecs.shape[0], self.dim), dtype=np.float32)
                                    adj[:, :min(vecs.shape[1], self.dim)] = vecs[:, :min(vecs.shape[1], self.dim)]
                                    vecs = adj
                                self._faiss_index.add(vecs)
                                logger.info("Imported numpy vectors into new faiss index (converted)")
                            except Exception:
                                logger.debug("No legacy numpy vectors to import for faiss")
                else:
                    npy_path = f"{self.index_path}.npy"
                    if os.path.exists(npy_path):
                        arr = np.load(npy_path)
                        # ensure dimension matches
                        if arr.ndim == 2 and arr.shape[1] != self.dim:
                            adj = np.zeros((arr.shape[0], self.dim), dtype=np.float32)
                            adj[:, :min(arr.shape[1], self.dim)] = arr[:, :min(arr.shape[1], self.dim)]
                            arr = adj
                        self._numpy_vectors = arr.astype(np.float32)
                        logger.info("Loaded numpy vectors from disk")
                    else:
                        # create empty
                        self._numpy_vectors = np.zeros((0, self.dim), dtype=np.float32)
                        logger.info("Initialized empty numpy vector store")
            except Exception as e:
                logger.exception("Failed to load index")
                raise CustomException(e, sys)

    def _save_index(self) -> None:
        with self._io_lock:
            try:
                if self._use_faiss:
                    _ensure_parent_dir(self.index_path)
                    faiss.write_index(self._faiss_index, self.index_path)
                    # also keep a numpy snapshot for portability
                    try:
                        arr = self._faiss_index.reconstruct_n(0, self._faiss_index.ntotal)
                        npy_path = f"{self.index_path}.npy"
                        _ensure_parent_dir(npy_path)
                        np.save(npy_path, arr)
                    except Exception:
                        # reconstruct_n may not be available on all index types; ignore
                        pass
                    logger.info("Saved faiss index to disk")
                else:
                    npy_path = f"{self.index_path}.npy"
                    _ensure_parent_dir(npy_path)
                    if self._numpy_vectors is None:
                        self._numpy_vectors = np.zeros((0, self.dim), dtype=np.float32)
                    np.save(npy_path, self._numpy_vectors)
                    logger.info("Saved numpy vectors to disk")
            except Exception as e:
                logger.exception("Failed to save index")
                raise CustomException(e, sys)

    # ---------------------------
    # Upsert / chunked add
    # ---------------------------
    def upsert_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Upsert a list of documents where each document is {"id": <opt>, "text": <str>, "meta": <dict>}
        Returns list of ids inserted/updated.

        Note: this implementation appends new vectors — it does not overwrite existing vectors in-place.
        Overwriting would require tracking & removing the specific vector from FAISS (complex).
        For now, if an id already exists we record duplicate id and it will remain mapped to the newest vector by index ordering.
        """
        with self._io_lock:
            try:
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

                vecs = self._embed_texts(texts)  # shape (N, dim)

                # Add vectors to backend
                if self._use_faiss:
                    if self._faiss_index is None:
                        self._faiss_index = faiss.IndexFlatIP(self.dim)
                    # FAISS expects contiguous float32 matrix
                    self._faiss_index.add(vecs.astype(np.float32))
                else:
                    if self._numpy_vectors is None or self._numpy_vectors.size == 0:
                        self._numpy_vectors = vecs.copy()
                    else:
                        self._numpy_vectors = np.vstack([self._numpy_vectors, vecs.astype(np.float32)])

                # update ids_list and metadata (append order corresponds to vector positions)
                for doc_id, meta, text in metas:
                    if doc_id in self._metadata:
                        # update metadata in place but also append new id so it's available as a latest vector
                        logger.debug("Upsert: existing id %s detected; metadata will be updated and duplicate vector appended", doc_id)
                        self._metadata[doc_id].update(meta or {})
                    else:
                        self._metadata[doc_id] = meta or {}
                    # keep canonical text in metadata
                    self._metadata[doc_id]["text"] = text
                    self._ids_list.append(doc_id)

                # persist
                self._save_index()
                self._save_meta()
                logger.info("Upserted %d documents into vector index (total_entries=%d)", len(ids_added), len(self._ids_list))
                return ids_added
            except Exception as e:
                logger.exception("Failed to upsert documents")
                raise CustomException(e, sys)

    def add_texts(self, texts: List[Dict[str, Any]]) -> List[str]:
        """
        Accepts list of {"id": <opt>, "text": <str>, "meta": <dict>} and:
          - splits each text into chunks using RecursiveCharacterTextSplitter (if available)
          - upserts each chunk as a separate vector with metadata containing source_id & chunk_index
        Returns list of chunk ids inserted.
        """
        docs = []
        for item in texts:
            src_id = item.get("id") or uuid.uuid4().hex
            txt = item.get("text") or ""
            meta = item.get("meta", {}) or {}

            if _USE_TEXT_SPLITTER:
                try:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
                    chunks = splitter.split_text(txt)
                except Exception:
                    chunks = [txt]
            else:
                chunks = [txt]

            for idx, chunk in enumerate(chunks):
                docs.append({
                    "id": f"{src_id}__chunk__{idx}",
                    "text": chunk,
                    "meta": {"source_id": src_id, "chunk_index": idx, **(meta or {})}
                })

        return self.upsert_documents(docs)

    # ---------------------------
    # Search
    # ---------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the top_k most relevant chunks/documents for the query.
        Returns list of {"id", "score", "text", "meta"} in descending relevance.
        """
        try:
            q_vec = np.array(self._embed_texts([query])[0], dtype=np.float32)
            q_vec = self._normalize_vector(self._adjust_dim(q_vec))

            results: List[Dict[str, Any]] = []
            if self._use_faiss:
                if self._faiss_index is None or getattr(self._faiss_index, "ntotal", 0) == 0:
                    return []
                D, I = self._faiss_index.search(np.expand_dims(q_vec, axis=0), top_k)
                # I and D are arrays shaped (1, k)
                ids_ordered = self._ids_list
                for score, idx in zip(D[0], I[0]):
                    if idx < 0 or idx >= len(ids_ordered):
                        continue
                    doc_id = ids_ordered[idx]
                    md = self._metadata.get(doc_id, {})
                    results.append({"id": doc_id, "score": float(score), "text": md.get("text"), "meta": md})
                return results
            else:
                if self._numpy_vectors is None or self._numpy_vectors.shape[0] == 0:
                    return []
                sims = self._numpy_vectors @ q_vec  # inner product since vectors normalized
                idxs = np.argsort(-sims)[:top_k]
                ids_ordered = self._ids_list
                for idx in idxs:
                    if idx < 0 or idx >= len(ids_ordered):
                        continue
                    doc_id = ids_ordered[idx]
                    md = self._metadata.get(doc_id, {})
                    results.append({"id": doc_id, "score": float(sims[idx]), "text": md.get("text"), "meta": md})
                return results
        except Exception as e:
            logger.exception("Search failed")
            raise CustomException(e, sys)

    # ---------------------------
    # Utility / Clear / Info
    # ---------------------------
    def clear(self) -> None:
        """Remove all vectors and metadata (resets index)."""
        with self._io_lock:
            try:
                self._metadata = {}
                self._ids_list = []
                if self._use_faiss:
                    self._faiss_index = faiss.IndexFlatIP(self.dim)
                    _remove_file(self.index_path)
                    _remove_file(f"{self.index_path}.npy")
                else:
                    self._numpy_vectors = np.zeros((0, self.dim), dtype=np.float32)
                    _remove_file(f"{self.index_path}.npy")
                _remove_file(self.meta_path)
                logger.info("Cleared vector store and metadata")
            except Exception as e:
                logger.exception("Failed to clear vector store")
                raise CustomException(e, sys)

    def info(self) -> Dict[str, Any]:
        """Return basic diagnostic info."""
        with self._io_lock:
            return {
                "index_path": self.index_path,
                "meta_path": self.meta_path,
                "use_faiss": self._use_faiss,
                "dim": int(self.dim),
                "entries": len(self._ids_list),
            }

# ---------------------------
# End of file
# ---------------------------
