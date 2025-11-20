import os
import sys
import json
import uuid
import logging
from typing import List, Dict, Any, Callable, Optional, Tuple

import numpy as np

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("vector_search")

# Try to import faiss; if unavailable, we'll use a numpy fallback
_FAISS_AVAILABLE = True
try:
    import faiss  # type: ignore
except Exception:
    _FAISS_AVAILABLE = False
    logger.info("faiss not available; using numpy fallback for vector search.")


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


class VectorSearch:
    """
    Simple vector search wrapper that supports:
      - building/upserting documents (documents have 'id' and 'text' + optional metadata)
      - saving/loading index and metadata
      - searching with a provided embedding function

    Storage:
      - Index path: config.VECTOR_INDEX_PATH (file). For FAISS we write the index file.
      - Metadata path: same path + '.meta.json', stores id -> metadata mapping.
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

            # metadata: id -> metadata dict (contains 'text' and optional extras)
            self._metadata: Dict[str, Dict[str, Any]] = {}

            # internal vector storage depending on backend:
            self._use_faiss = _FAISS_AVAILABLE
            self._faiss_index = None
            self._numpy_vectors: Optional[np.ndarray] = None
            self._id_to_idx: Dict[str, int] = {}  # mapping for numpy backend
            self._next_idx = 0

            # try to load existing index & metadata if present
            self._load_meta()
            self._load_index()
            logger.info(f"VectorSearch initialized at {self.index_path} (faiss={self._use_faiss})")
        except Exception as e:
            logger.exception("Failed to initialize VectorSearch")
            raise CustomException(e, sys)

    # ---------------------------
    # Default embedding (placeholder)
    # ---------------------------
    def _default_embedding(self, text: str) -> List[float]:
        """
        Very simple deterministic embedding fallback.
        Not for production: maps characters to trigram counts and then pads/normalizes.
        """
        vec = np.zeros(self.dim, dtype=float)
        if not text:
            return vec.tolist()
        s = text.lower()
        for i, ch in enumerate(s):
            vec[i % self.dim] += (ord(ch) % 97) * 0.001
        # normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    # ---------------------------
    # Metadata persistence
    # ---------------------------
    def _load_meta(self) -> None:
        try:
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                # build id->idx for numpy if we need to
                if not self._use_faiss and self._metadata:
                    # we will rebuild numpy vectors on load_index
                    pass
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
        """
        Load index from disk. For faiss we use read_index; for numpy we load a .npy if present.
        If index doesn't exist we initialize an empty index.
        """
        try:
            if self._use_faiss:
                if os.path.exists(self.index_path):
                    self._faiss_index = faiss.read_index(self.index_path)
                    logger.info("Loaded faiss index from disk")
                else:
                    # create empty index
                    self._faiss_index = faiss.IndexFlatIP(self.dim)  # inner-product (cosine w/ normalized vectors)
                    logger.info("Created new faiss index (empty)")
            else:
                # numpy fallback: store vectors in index_path + '.npy'
                npy_path = f"{self.index_path}.npy"
                if os.path.exists(npy_path):
                    self._numpy_vectors = np.load(npy_path)
                    # build id->idx mapping from metadata order
                    self._id_to_idx = {}
                    for idx, doc_id in enumerate(list(self._metadata.keys())):
                        self._id_to_idx[doc_id] = idx
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
        """
        Upsert a list of documents. Each doc must have 'text' and optionally 'id' and metadata.
        Returns list of doc ids upserted.
        """
        try:
            ids = []
            vecs = []
            for doc in documents:
                doc_id = doc.get("id") or uuid.uuid4().hex
                text = doc.get("text") or ""
                meta = doc.get("meta", {})
                # generate embedding
                emb = np.array(self.embedding_fn(text), dtype=np.float32)
                # ensure dim
                if emb.size != self.dim:
                    # try to pad or truncate
                    if emb.size < self.dim:
                        padded = np.zeros(self.dim, dtype=np.float32)
                        padded[: emb.size] = emb
                        emb = padded
                    else:
                        emb = emb[: self.dim]
                # normalize for inner product cosine-like similarity
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                ids.append(doc_id)
                vecs.append(emb)
                # update metadata
                self._metadata[doc_id] = {"text": text, **meta}

            if not vecs:
                return []

            vecs_arr = np.vstack(vecs).astype(np.float32)

            if self._use_faiss:
                # Ensure faiss index dimension matches
                if self._faiss_index is None:
                    self._faiss_index = faiss.IndexFlatIP(self.dim)
                # add
                self._faiss_index.add(vecs_arr)
            else:
                # append to numpy array
                if self._numpy_vectors is None:
                    self._numpy_vectors = vecs_arr.copy()
                else:
                    self._numpy_vectors = np.vstack([self._numpy_vectors, vecs_arr])
                # update id->idx mapping
                start = self._next_idx
                for i, doc_id in enumerate(ids):
                    self._id_to_idx[doc_id] = start + i
                self._next_idx += len(ids)

            # persist index and metadata
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
        """
        Search for query and return a list of result dicts:
          [{"id": id, "score": float, "text": ..., "meta": {...}}, ...]
        """
        try:
            q_emb = np.array(self.embedding_fn(query), dtype=np.float32)
            if q_emb.size != self.dim:
                if q_emb.size < self.dim:
                    padded = np.zeros(self.dim, dtype=np.float32)
                    padded[: q_emb.size] = q_emb
                    q_emb = padded
                else:
                    q_emb = q_emb[: self.dim]

            # normalize
            norm = np.linalg.norm(q_emb)
            if norm > 0:
                q_emb = q_emb / norm

            if self._use_faiss:
                if self._faiss_index is None or self._faiss_index.ntotal == 0:
                    return []
                # faiss expects (n, dim)
                D, I = self._faiss_index.search(np.expand_dims(q_emb, axis=0), top_k)
                scores = D[0].tolist()
                indices = I[0].tolist()
                results = []
                # We need to map index offsets to metadata ids. FAISS stores vectors in insertion order
                # Our metadata dict preserves insertion order since Python 3.7, so we can get id by position.
                ids_list = list(self._metadata.keys())
                for idx, sc in zip(indices, scores):
                    if idx < 0:
                        continue
                    if idx >= len(ids_list):
                        continue
                    doc_id = ids_list[idx]
                    md = self._metadata.get(doc_id, {})
                    results.append({"id": doc_id, "score": float(sc), "text": md.get("text"), "meta": md})
                return results
            else:
                if self._numpy_vectors is None or self._numpy_vectors.shape[0] == 0:
                    return []
                # compute cosine similarity via dot product (vectors normalized)
                dots = float(np.dot(self._numpy_vectors, q_emb))
                # For many rows we want vectorized
                sims = (self._numpy_vectors @ q_emb).astype(float)
                # get top_k indices
                idxs = np.argsort(-sims)[:top_k]
                results = []
                ids_list = list(self._metadata.keys())
                for idx in idxs:
                    if idx < 0 or idx >= len(ids_list):
                        continue
                    doc_id = ids_list[idx]
                    md = self._metadata.get(doc_id, {})
                    score = float(sims[idx])
                    results.append({"id": doc_id, "score": score, "text": md.get("text"), "meta": md})
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
                # remove index file if exists
                try:
                    if os.path.exists(self.index_path):
                        os.remove(self.index_path)
                except Exception:
                    pass
            else:
                self._numpy_vectors = np.zeros((0, self.dim), dtype=np.float32)
                npy_path = f"{self.index_path}.npy"
                try:
                    if os.path.exists(npy_path):
                        os.remove(npy_path)
                except Exception:
                    pass
                self._id_to_idx = {}
                self._next_idx = 0
            # remove meta file
            try:
                if os.path.exists(self.meta_path):
                    os.remove(self.meta_path)
            except Exception:
                pass
            logger.info("Cleared vector index and metadata")
        except Exception as e:
            logger.exception("Failed to clear vector store")
            raise CustomException(e, sys)
