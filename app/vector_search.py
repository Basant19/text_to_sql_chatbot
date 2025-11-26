# app/vector_search.py
import os
import sys
import json
import uuid
from typing import List, Dict, Any, Callable, Optional, Union

import numpy as np

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("vector_search")

# ---------------------------
# Optional dependencies
# ---------------------------
_FAISS_AVAILABLE = True
try:
    import faiss  # type: ignore
except Exception:
    _FAISS_AVAILABLE = False
    logger.info("FAISS not available; using NumPy fallback for vector search.")

# Try multiple splitter/embeddings import locations (per user's note)
_USE_TEXT_SPLITTER = False
try:
    # preferred: lightweight separate package name shown in your note
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    _USE_TEXT_SPLITTER = True
    logger.info("Using langchain_text_splitters.RecursiveCharacterTextSplitter for chunking")
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
        _USE_TEXT_SPLITTER = True
        logger.info("Using langchain.text_splitter.RecursiveCharacterTextSplitter for chunking")
    except Exception:
        logger.info("LangChain text splitter unavailable; chunking disabled.")

_USE_LANGCHAIN_EMBEDDINGS = False
# try the langchain_google_genai package first (as you suggested), then fallback to langchain.embeddings
_emb_module = None
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
    _USE_LANGCHAIN_EMBEDDINGS = True
    _emb_module = "langchain_google_genai"
    logger.info("langchain_google_genai.GoogleGenerativeAIEmbeddings is available")
except Exception:
    try:
        from langchain.embeddings import GoogleGenerativeAIEmbeddings  # type: ignore
        _USE_LANGCHAIN_EMBEDDINGS = True
        _emb_module = "langchain.langchain"
        logger.info("langchain.embeddings.GoogleGenerativeAIEmbeddings is available")
    except Exception:
        logger.info("GoogleGenerativeAIEmbeddings unavailable; using default embedding function.")


# ---------------------------
# Helpers
# ---------------------------
def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _remove_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _safe_json_load(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


# ---------------------------
# VectorSearch class
# ---------------------------
class VectorSearch:
    """
    Vector search wrapper supporting:
      - FAISS or NumPy backend
      - Upserting documents with 'text' and optional metadata
      - RecursiveCharacterTextSplitter chunking (if available)
      - Google Generative Embeddings via LangChain (if available)
      - Persistent index & metadata (.meta.json + faiss file / .npy)
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_fn: Optional[Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]] = None,
        dim: int = 128,
    ):
        try:
            # paths
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

            # embedding function: must accept either str or list[str]
            if embedding_fn is None and _USE_LANGCHAIN_EMBEDDINGS:
                self._init_langchain_embeddings(dim)
            else:
                # if provided embedding_fn might accept only single str; wrap it to support list
                self.embedding_fn = self._wrap_embedding_fn(embedding_fn) if embedding_fn is not None else self._wrap_embedding_fn(self._default_embedding)
                self.dim = dim

            # metadata and backend
            self._metadata: Dict[str, Dict[str, Any]] = {}
            self._use_faiss = _FAISS_AVAILABLE
            self._faiss_index = None
            self._numpy_vectors: Optional[np.ndarray] = None
            self._id_to_idx: Dict[str, int] = {}
            self._next_idx = 0

            # load existing
            self._load_meta()
            self._load_index()

            logger.info(f"VectorSearch initialized at {self.index_path} (FAISS={self._use_faiss}, dim={self.dim})")
        except Exception as e:
            logger.exception("Failed to initialize VectorSearch")
            raise CustomException(e, sys)

    # ---------------------------
    # Embedding integration
    # ---------------------------
    def _init_langchain_embeddings(self, dim: int):
        """
        Initialize GoogleGenerativeAIEmbeddings (if available) and create an embedding_fn
        that accepts either str or List[str]. If initialization fails, fall back to default.
        """
        try:
            # instantiate provider based on earlier import success
            if _emb_module == "langchain_google_genai":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
                emb_client = GoogleGenerativeAIEmbeddings(model=getattr(config, "EMBEDDING_MODEL", "models/gemini-embedding-001"))
            else:
                from langchain.embeddings import GoogleGenerativeAIEmbeddings  # type: ignore
                emb_client = GoogleGenerativeAIEmbeddings(model=getattr(config, "EMBEDDING_MODEL", "models/gemini-embedding-001"))

            logger.info("VectorSearch: initialized GoogleGenerativeAIEmbeddings embedding client")

            def _emb_fn(x: Union[str, List[str]]):
                if isinstance(x, (list, tuple)):
                    # embed_documents returns list[list[float]]
                    res = emb_client.embed_documents(list(x))
                    return [list(map(float, r)) for r in res]
                else:
                    res = emb_client.embed_query(x)
                    return list(map(float, res)) if res is not None else [0.0] * dim

            # set wrapper (ensures consistent return types)
            self.embedding_fn = _emb_fn
            # infer dim
            sample = self.embedding_fn("hello world")
            self.dim = len(sample) if isinstance(sample, (list, tuple)) else dim
            logger.info("VectorSearch: inferred embedding dim=%d from provider", self.dim)
        except Exception as e:
            logger.warning("VectorSearch: failed to initialize GoogleGenerativeAIEmbeddings, falling back: %s", e)
            # fallback
            self.embedding_fn = self._wrap_embedding_fn(self._default_embedding)
            self.dim = dim

    def _wrap_embedding_fn(self, fn: Callable[[str], List[float]]) -> Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]:
        """
        Wrap a function that may accept a single string into one that accepts either
        a single string or a list of strings. The wrapped function returns either
        a single vector (list[float]) or a list of vectors.
        """
        def wrapper(x: Union[str, List[str]]):
            if isinstance(x, (list, tuple)):
                out = []
                for t in x:
                    try:
                        v = fn(t)
                        out.append(list(map(float, v)))
                    except Exception:
                        # ensure consistent shape on error
                        out.append([0.0] * getattr(self, "dim", 128))
                return out
            else:
                return list(map(float, fn(x)))
        return wrapper

    # ---------------------------
    # Default deterministic embedding (list-capable via wrapper)
    # ---------------------------
    def _single_default_embedding(self, text: str, dim: Optional[int] = None) -> List[float]:
        """
        Deterministic fallback embedding for single text.
        """
        d = dim or getattr(self, "dim", 128)
        vec = np.zeros(d, dtype=float)
        s = str(text or "")
        for i, ch in enumerate(s.lower()):
            vec[i % d] += (ord(ch) % 97) * 0.001
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    # keep old name for internal compatibility with previous code
    def _default_embedding(self, text: str) -> List[float]:
        return self._single_default_embedding(text, getattr(self, "dim", 128))

    # ---------------------------
    # Vector helpers
    # ---------------------------
    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _adjust_dim(self, vec: np.ndarray) -> np.ndarray:
        if vec.size != self.dim:
            if vec.size < self.dim:
                padded = np.zeros(self.dim, dtype=np.float32)
                padded[: vec.size] = vec
                return padded
            return vec[: self.dim]
        return vec

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Given list[str] -> return numpy array shape (len(texts), dim)
        embedding_fn must accept either a list or single string. This function handles:
          - embedding_fn(list) -> list[list[float]]
          - embedding_fn(str) -> list[float] (called per text)
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        try:
            # try calling embedding_fn with the whole list (preferred)
            res = self.embedding_fn(texts)
            if isinstance(res, (list, tuple)) and res and isinstance(res[0], (list, tuple, np.ndarray)):
                arr = np.array([np.array(r, dtype=np.float32) for r in res], dtype=np.float32)
            else:
                # embedding_fn returned something unexpected for list input; fall back to per-item
                vecs = []
                for t in texts:
                    v = self.embedding_fn(t)
                    vecs.append(np.array(v, dtype=np.float32))
                arr = np.vstack(vecs).astype(np.float32)
        except Exception:
            # safe fallback: call per-item
            vecs = []
            for t in texts:
                try:
                    v = self.embedding_fn(t)
                    vecs.append(np.array(v, dtype=np.float32))
                except Exception:
                    vecs.append(np.zeros(self.dim, dtype=np.float32))
            arr = np.vstack(vecs).astype(np.float32)

        # normalize & adjust dims
        adjusted = []
        for v in arr:
            v = self._adjust_dim(v)
            v = self._normalize_vector(v)
            adjusted.append(v)
        return np.vstack(adjusted).astype(np.float32)

    # ---------------------------
    # Metadata persistence
    # ---------------------------
    def _load_meta(self) -> None:
        try:
            if os.path.exists(self.meta_path):
                self._metadata = _safe_json_load(self.meta_path)
            else:
                self._metadata = {}
        except Exception as e:
            logger.exception("Failed to load metadata")
            raise CustomException(e, sys)

    def _save_meta(self) -> None:
        try:
            _ensure_parent_dir(self.meta_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.exception("Failed to save metadata")
            raise CustomException(e, sys)

    # ---------------------------
    # Index persistence
    # ---------------------------
    def _load_index(self) -> None:
        try:
            if self._use_faiss:
                if os.path.exists(self.index_path):
                    self._faiss_index = faiss.read_index(self.index_path)
                    logger.info("Loaded faiss index from disk")
                else:
                    self._faiss_index = faiss.IndexFlatIP(self.dim)
                    logger.info("Created new faiss index (empty)")
            else:
                npy_path = f"{self.index_path}.npy"
                if os.path.exists(npy_path):
                    self._numpy_vectors = np.load(npy_path)
                    self._id_to_idx = {doc_id: idx for idx, doc_id in enumerate(list(self._metadata.keys()))}
                    self._next_idx = int(self._numpy_vectors.shape[0])
                    logger.info("Loaded numpy vectors from disk")
                else:
                    self._numpy_vectors = np.zeros((0, self.dim), dtype=np.float32)
                    self._id_to_idx = {}
                    self._next_idx = 0
                    logger.info("Initialized empty numpy vector store")
        except Exception as e:
            logger.exception("Failed to load index")
            raise CustomException(e, sys)

    def _save_index(self) -> None:
        try:
            if self._use_faiss:
                _ensure_parent_dir(self.index_path)
                faiss.write_index(self._faiss_index, self.index_path)
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
        """
        try:
            ids, vecs = [], []
            for doc in documents:
                doc_id = doc.get("id") or uuid.uuid4().hex
                text = doc.get("text") or ""
                meta = doc.get("meta", {})

                emb = np.array(self._embed_texts([text])[0], dtype=np.float32)
                emb = self._normalize_vector(self._adjust_dim(emb))

                ids.append(doc_id)
                vecs.append(emb)
                self._metadata[doc_id] = {"text": text, **(meta or {})}

            if not vecs:
                return []

            vecs_arr = np.vstack(vecs).astype(np.float32)
            if self._use_faiss:
                if self._faiss_index is None:
                    self._faiss_index = faiss.IndexFlatIP(self.dim)
                self._faiss_index.add(vecs_arr)
            else:
                if self._numpy_vectors is None or self._numpy_vectors.size == 0:
                    self._numpy_vectors = vecs_arr.copy()
                else:
                    self._numpy_vectors = np.vstack([self._numpy_vectors, vecs_arr])
                start = self._next_idx
                for i, doc_id in enumerate(ids):
                    self._id_to_idx[doc_id] = start + i
                self._next_idx += len(ids)

            self._save_index()
            self._save_meta()
            logger.info("Upserted %d documents into vector index", len(ids))
            return ids
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
        try:
            chunk_docs = []
            splitter = None
            if _USE_TEXT_SPLITTER:
                try:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
                except Exception as e:
                    logger.warning("Failed to instantiate text splitter: %s", e)
                    splitter = None

            for item in texts:
                src_id = item.get("id") or uuid.uuid4().hex
                text = item.get("text") or ""
                meta = item.get("meta", {})
                if splitter:
                    chunks = splitter.split_text(text)
                else:
                    chunks = [text]
                for idx, chunk in enumerate(chunks):
                    chunk_docs.append({
                        "id": f"{src_id}__chunk__{idx}",
                        "text": chunk,
                        "meta": {"source_id": src_id, "chunk_index": idx, **(meta or {})}
                    })
            return self.upsert_documents(chunk_docs)
        except Exception as e:
            logger.exception("Failed to add_texts")
            raise CustomException(e, sys)

    # ---------------------------
    # Search
    # ---------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the top_k most relevant chunks/documents for the query.
        Returns list of {"id", "score", "text", "meta"}.
        """
        try:
            q_emb = np.array(self._embed_texts([query])[0], dtype=np.float32)
            q_emb = self._normalize_vector(self._adjust_dim(q_emb))

            if self._use_faiss:
                if self._faiss_index is None or getattr(self._faiss_index, "ntotal", 0) == 0:
                    return []
                D, I = self._faiss_index.search(np.expand_dims(q_emb, axis=0), top_k)
                ids_list = list(self._metadata.keys())
                results = []
                for idx, score in zip(I[0], D[0]):
                    if idx < 0 or idx >= len(ids_list):
                        continue
                    doc_id = ids_list[idx]
                    md = self._metadata.get(doc_id, {})
                    results.append({"id": doc_id, "score": float(score), "text": md.get("text"), "meta": md})
                return results
            else:
                if self._numpy_vectors is None or self._numpy_vectors.shape[0] == 0:
                    return []
                sims = self._numpy_vectors @ q_emb
                idxs = np.argsort(-sims)[:top_k]
                ids_list = list(self._metadata.keys())
                results = []
                for idx in idxs:
                    if idx < 0 or idx >= len(ids_list):
                        continue
                    doc_id = ids_list[idx]
                    md = self._metadata.get(doc_id, {})
                    results.append({"id": doc_id, "score": float(sims[idx]), "text": md.get("text"), "meta": md})
                return results
        except Exception as e:
            logger.exception("Search failed")
            raise CustomException(e, sys)

    # ---------------------------
    # Utility / Clear
    # ---------------------------
    def clear(self) -> None:
        try:
            self._metadata = {}
            if self._use_faiss:
                self._faiss_index = faiss.IndexFlatIP(self.dim)
                _remove_file(self.index_path)
            else:
                self._numpy_vectors = np.zeros((0, self.dim), dtype=np.float32)
                self._id_to_idx = {}
                self._next_idx = 0
                _remove_file(f"{self.index_path}.npy")
            _remove_file(self.meta_path)
            logger.info("Cleared vector store and metadata")
        except Exception as e:
            logger.exception("Failed to clear vector store")
            raise CustomException(e, sys)
