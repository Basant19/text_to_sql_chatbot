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

# Try to import faiss; fallback to numpy if unavailable
_FAISS_AVAILABLE = True
try:
    import faiss  # type: ignore
except Exception:
    _FAISS_AVAILABLE = False
    logger.info("faiss not available; using numpy fallback for vector search.")


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _remove_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


class VectorSearch:
    """
    Vector search wrapper supporting:
      - Upserting documents with 'text' and optional metadata
      - Index/metadata persistence
      - Search with top-k results
      - FAISS or NumPy backend
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        dim: int = 128,
    ):
        try:
            self.index_path = index_path or getattr(config, "VECTOR_INDEX_PATH", "./faiss/index.faiss")
            self.meta_path = f"{self.index_path}.meta.json"
            _ensure_parent_dir(self.index_path)
            _ensure_parent_dir(self.meta_path)

            self.embedding_fn = embedding_fn or self._default_embedding
            self.dim = dim

            self._metadata: Dict[str, Dict[str, Any]] = {}

            # Backend setup
            self._use_faiss = _FAISS_AVAILABLE
            self._faiss_index = None
            self._numpy_vectors: Optional[np.ndarray] = None
            self._id_to_idx: Dict[str, int] = {}
            self._next_idx = 0

            # Load existing index & metadata if present
            self._load_meta()
            self._load_index()
            logger.info(f"VectorSearch initialized at {self.index_path} (faiss={self._use_faiss})")
        except Exception as e:
            logger.exception("Failed to initialize VectorSearch")
            raise CustomException(e, sys)

    # ---------------------------
    # Helpers
    # ---------------------------
    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _adjust_dim(self, vec: np.ndarray) -> np.ndarray:
        if vec.size != self.dim:
            logger.warning(f"Embedding dimension mismatch ({vec.size} != {self.dim}); adjusting")
            if vec.size < self.dim:
                padded = np.zeros(self.dim, dtype=np.float32)
                padded[: vec.size] = vec
                return padded
            return vec[: self.dim]
        return vec

    # ---------------------------
    # Default embedding
    # ---------------------------
    def _default_embedding(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=float)
        if text:
            for i, ch in enumerate(text.lower()):
                vec[i % self.dim] += (ord(ch) % 97) * 0.001
            vec = self._normalize_vector(vec)
        return vec.tolist()

    # ---------------------------
    # Metadata persistence
    # ---------------------------
    def _load_meta(self) -> None:
        try:
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
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
    # Index persistence / building
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
                    self._next_idx = self._numpy_vectors.shape[0]
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
                if self._numpy_vectors is None:
                    self._numpy_vectors = np.zeros((0, self.dim), dtype=np.float32)
                _ensure_parent_dir(npy_path)
                np.save(npy_path, self._numpy_vectors)
                logger.info("Saved numpy vectors to disk")
        except Exception as e:
            logger.exception("Failed to save index")
            raise CustomException(e, sys)

    # ---------------------------
    # Upsert documents
    # ---------------------------
    def upsert_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        try:
            ids, vecs = [], []
            for doc in documents:
                doc_id = doc.get("id") or uuid.uuid4().hex
                text = doc.get("text") or ""
                meta = doc.get("meta", {})

                emb = np.array(self.embedding_fn(text), dtype=np.float32)
                emb = self._adjust_dim(emb)
                emb = self._normalize_vector(emb)

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
            logger.info(f"Upserted {len(ids)} documents")
            return ids
        except Exception as e:
            logger.exception("Failed to upsert documents")
            raise CustomException(e, sys)

    # ---------------------------
    # Search
    # ---------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            q_emb = np.array(self.embedding_fn(query), dtype=np.float32)
            q_emb = self._adjust_dim(q_emb)
            q_emb = self._normalize_vector(q_emb)

            if self._use_faiss:
                if self._faiss_index is None or self._faiss_index.ntotal == 0:
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
    # Utility
    # ---------------------------
    def clear(self) -> None:
        """Clear index and metadata (in memory and on disk)."""
        try:
            self._metadata = {}
            if self._use_faiss:
                self._faiss_index = faiss.IndexFlatIP(self.dim)
                _remove_file(self.index_path)
            else:
                self._numpy_vectors = np.zeros((0, self.dim), dtype=np.float32)
                _remove_file(f"{self.index_path}.npy")
                self._id_to_idx = {}
                self._next_idx = 0
            _remove_file(self.meta_path)
            logger.info("Cleared vector index and metadata")
        except Exception as e:
            logger.exception("Failed to clear vector store")
            raise CustomException(e, sys)
