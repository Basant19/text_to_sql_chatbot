# app/vector_search.py

import os
import sys
import json
import uuid
from typing import List, Dict, Any, Callable, Optional

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
    import faiss
except Exception:
    _FAISS_AVAILABLE = False
    logger.info("FAISS not available; using NumPy fallback for vector search.")

_USE_TEXT_SPLITTER = False
_USE_LANGCHAIN_EMBEDDINGS = False
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    _USE_TEXT_SPLITTER = True
except Exception:
    logger.info("LangChain text splitter unavailable; chunking disabled.")

try:
    from langchain.embeddings import GoogleGenerativeAIEmbeddings
    _USE_LANGCHAIN_EMBEDDINGS = True
except Exception:
    logger.info("GoogleGenerativeAIEmbeddings unavailable; using default embedding function.")


# ---------------------------
# Helper functions
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


# ---------------------------
# VectorSearch class
# ---------------------------
class VectorSearch:
    """
    Vector search wrapper supporting:
    - FAISS or NumPy backend
    - Upserting documents with 'text' and optional metadata
    - RecursiveCharacterTextSplitter chunking
    - Google Generative Embeddings (if available)
    - Persistent index & metadata
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        dim: int = 128,
    ):
        try:
            # Paths
            self.index_path = index_path or getattr(config, "VECTOR_INDEX_PATH", "./faiss/index.faiss")
            self.meta_path = f"{self.index_path}.meta.json"
            _ensure_parent_dir(self.index_path)
            _ensure_parent_dir(self.meta_path)

            # Embeddings
            if embedding_fn is None and _USE_LANGCHAIN_EMBEDDINGS:
                self._init_langchain_embeddings(dim)
            else:
                self.embedding_fn = embedding_fn or self._default_embedding
                self.dim = dim

            # Chunking
            self.chunk_size = int(getattr(config, "VECTOR_CHUNK_SIZE", 1000))
            self.chunk_overlap = int(getattr(config, "VECTOR_CHUNK_OVERLAP", 200))

            # Metadata & backend
            self._metadata: Dict[str, Dict[str, Any]] = {}
            self._use_faiss = _FAISS_AVAILABLE
            self._faiss_index = None
            self._numpy_vectors: Optional[np.ndarray] = None
            self._id_to_idx: Dict[str, int] = {}
            self._next_idx = 0

            # Load existing index
            self._load_meta()
            self._load_index()
            logger.info(f"VectorSearch initialized at {self.index_path} (FAISS={self._use_faiss}, dim={self.dim})")
        except Exception as e:
            logger.exception("Failed to initialize VectorSearch")
            raise CustomException(e, sys)

    # ---------------------------
    # Embedding init
    # ---------------------------
    def _init_langchain_embeddings(self, dim: int):
        try:
            self._emb_client = GoogleGenerativeAIEmbeddings()
            def _emb_fn(text_or_texts):
                if isinstance(text_or_texts, str):
                    res = self._emb_client.embed_query(text_or_texts)
                    return list(map(float, res)) if res else [0.0]*dim
                elif isinstance(text_or_texts, (list, tuple)):
                    res = self._emb_client.embed_documents(list(text_or_texts))
                    return [list(map(float, r)) for r in res]
                else:
                    raise ValueError("Unsupported type for embedding_fn input")
            self.embedding_fn = _emb_fn
            # Infer dim
            sample = self.embedding_fn("hello world")
            self.dim = len(sample) if isinstance(sample, list) else dim
            logger.info(f"Inferred embedding dimension={self.dim} from LangChain provider")
        except Exception as e:
            logger.warning("Failed to initialize GoogleGenerativeAIEmbeddings; falling back: %s", e)
            self.embedding_fn = self._default_embedding
            self.dim = dim

    # ---------------------------
    # Default embedding
    # ---------------------------
    def _default_embedding(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=float)
        for i, ch in enumerate(text.lower()):
            vec[i % self.dim] += (ord(ch) % 97) * 0.001
        vec = self._normalize_vector(vec)
        return vec.tolist()

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
                padded[:vec.size] = vec
                return padded
            return vec[:self.dim]
        return vec

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        res = self.embedding_fn(texts)
        if isinstance(res[0], (list, tuple, np.ndarray)):
            arr = np.array([np.array(r, dtype=np.float32) for r in res], dtype=np.float32)
        else:
            arr = np.array([np.array(self.embedding_fn(t), dtype=np.float32) for t in texts], dtype=np.float32)
        # Normalize & adjust
        arr = np.vstack([self._normalize_vector(self._adjust_dim(v)) for v in arr])
        return arr.astype(np.float32)

    # ---------------------------
    # Metadata persistence
    # ---------------------------
    def _load_meta(self) -> None:
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {}

    def _save_meta(self) -> None:
        _ensure_parent_dir(self.meta_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)

    # ---------------------------
    # Index persistence
    # ---------------------------
    def _load_index(self) -> None:
        if self._use_faiss:
            if os.path.exists(self.index_path):
                self._faiss_index = faiss.read_index(self.index_path)
            else:
                self._faiss_index = faiss.IndexFlatIP(self.dim)
        else:
            npy_path = f"{self.index_path}.npy"
            if os.path.exists(npy_path):
                self._numpy_vectors = np.load(npy_path)
                self._id_to_idx = {doc_id: idx for idx, doc_id in enumerate(list(self._metadata.keys()))}
                self._next_idx = int(self._numpy_vectors.shape[0])
            else:
                self._numpy_vectors = np.zeros((0, self.dim), dtype=np.float32)
                self._id_to_idx = {}
                self._next_idx = 0

    def _save_index(self) -> None:
        if self._use_faiss:
            _ensure_parent_dir(self.index_path)
            faiss.write_index(self._faiss_index, self.index_path)
        else:
            npy_path = f"{self.index_path}.npy"
            _ensure_parent_dir(npy_path)
            np.save(npy_path, self._numpy_vectors)

    # ---------------------------
    # Upsert / chunked add
    # ---------------------------
    def upsert_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        ids, vecs = [], []
        for doc in documents:
            doc_id = doc.get("id") or uuid.uuid4().hex
            text = doc.get("text") or ""
            meta = doc.get("meta", {})
            emb = np.array(self._embed_texts([text])[0], dtype=np.float32)
            emb = self._normalize_vector(self._adjust_dim(emb))
            ids.append(doc_id)
            vecs.append(emb)
            self._metadata[doc_id] = {"text": text, **meta}

        if not vecs:
            return []

        vecs_arr = np.vstack(vecs).astype(np.float32)
        if self._use_faiss:
            if self._faiss_index is None:
                self._faiss_index = faiss.IndexFlatIP(self.dim)
            self._faiss_index.add(vecs_arr)
        else:
            if self._numpy_vectors is None:
                self._numpy_vectors = vecs_arr.copy()
            else:
                self._numpy_vectors = np.vstack([self._numpy_vectors, vecs_arr])
            start = self._next_idx
            for i, doc_id in enumerate(ids):
                self._id_to_idx[doc_id] = start + i
            self._next_idx += len(ids)

        self._save_index()
        self._save_meta()
        return ids

    def add_texts(self, texts: List[Dict[str, Any]]) -> List[str]:
        chunk_docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap) if _USE_TEXT_SPLITTER else None
        for item in texts:
            src_id = item.get("id") or uuid.uuid4().hex
            text = item.get("text") or ""
            meta = item.get("meta", {})
            chunks = splitter.split_text(text) if splitter else [text]
            for idx, chunk in enumerate(chunks):
                chunk_docs.append({
                    "id": f"{src_id}__chunk__{idx}",
                    "text": chunk,
                    "meta": {"source_id": src_id, "chunk_index": idx, **meta}
                })
        return self.upsert_documents(chunk_docs)

    # ---------------------------
    # Search
    # ---------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = np.array(self._embed_texts([query])[0], dtype=np.float32)
        q_emb = self._normalize_vector(self._adjust_dim(q_emb))

        if self._use_faiss:
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                return []
            D, I = self._faiss_index.search(np.expand_dims(q_emb, axis=0), top_k)
            ids_list = list(self._metadata.keys())
            return [
                {"id": ids_list[idx], "score": float(score), "text": self._metadata[ids_list[idx]]["text"], "meta": self._metadata[ids_list[idx]]}
                for idx, score in zip(I[0], D[0]) if 0 <= idx < len(ids_list)
            ]
        else:
            if self._numpy_vectors is None or self._numpy_vectors.shape[0] == 0:
                return []
            sims = self._numpy_vectors @ q_emb
            idxs = np.argsort(-sims)[:top_k]
            ids_list = list(self._metadata.keys())
            return [
                {"id": ids_list[idx], "score": float(sims[idx]), "text": self._metadata[ids_list[idx]]["text"], "meta": self._metadata[ids_list[idx]]}
                for idx in idxs if 0 <= idx < len(ids_list)
            ]

    # ---------------------------
    # Clear
    # ---------------------------
    def clear(self) -> None:
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
