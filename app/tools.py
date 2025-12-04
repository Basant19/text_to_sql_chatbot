# app/tools.py
"""
Tools - light dependency-adapter used across graph nodes and app logic.

Responsibilities:
 - Provide safe, list-capable embedding_fn wrappers (used by VectorSearch)
 - Lazily instantiate or accept injected backends (db, schema_store, vector_search, executor)
 - Provide simple adapter methods for common operations:
     * schema access (get_schema, get_sample_rows, list_csvs)
     * vector ops (search_vectors, upsert_vectors, add_texts, clear_vectors, get_vector_meta)
     * DB/sql execution (execute_sql, load_table, list_tables)
 - Fail early and cleanly with CustomException when required backends are missing.
"""
from __future__ import annotations

import sys
import uuid
from typing import Any, Dict, List, Optional, Callable, Union

from app.logger import get_logger
from app.exception import CustomException

logger = get_logger("tools")


class Tools:
    def __init__(
        self,
        db: Optional[Any] = None,
        schema_store: Optional[Any] = None,
        vector_search: Optional[Any] = None,
        executor: Optional[Any] = None,
        provider_client: Optional[Any] = None,
        tracer_client: Optional[Any] = None,
    ):
        """
        Initialize tool adapters. If a component is not provided, attempt to import a default
        implementation. Fail gracefully if defaults aren't available (set to None) — callers
        should handle missing backends via CustomException.
        """
        try:
            # Lazy-import config if present (not strictly required)
            try:
                from app import config as _config  # type: ignore
            except Exception:
                _config = None

            # Database backend
            if db is None:
                try:
                    # expects module or object with required interface
                    from app import database as _database  # type: ignore
                    self._db = _database
                except Exception:
                    logger.warning("Tools: default database backend not available; inject db for DB features.")
                    self._db = None
            else:
                self._db = db

            # SchemaStore backend
            if schema_store is None:
                try:
                    from app.schema_store import SchemaStore  # type: ignore
                    self._schema_store = SchemaStore()
                except Exception:
                    logger.warning("Tools: default SchemaStore not available; inject schema_store for schema features.")
                    self._schema_store = None
            else:
                self._schema_store = schema_store

            # VectorSearch backend
            if vector_search is None:
                try:
                    from app.vector_search import VectorSearch  # type: ignore

                    # Determine index path and inferred dim from config if possible
                    index_path = getattr(_config, "VECTOR_INDEX_PATH", None) if _config else None
                    index_path = index_path or "./faiss/index.faiss"
                    try:
                        inferred_dim = int(getattr(_config, "VECTOR_DIM", 128)) if _config else 128
                    except Exception:
                        inferred_dim = 128

                    # Build safe embedding wrapper (tries langchain embedding, falls back to deterministic)
                    embedding_fn_wrapper = self._build_embedding_wrapper(inferred_dim, _config)

                    self._vector_search = VectorSearch(index_path=index_path, embedding_fn=embedding_fn_wrapper, dim=inferred_dim)
                    logger.info("Tools: initialized default VectorSearch with safe embedding_fn")
                except Exception as e:
                    logger.warning("Tools: default VectorSearch initialization failed; vector_search must be injected. Error: %s", e)
                    self._vector_search = None
            else:
                self._vector_search = vector_search

            # Executor (SQL execution) backend
            if executor is None:
                try:
                    from app import sql_executor as _executor  # type: ignore
                    self._executor = _executor
                except Exception:
                    logger.warning("Tools: default sql_executor not available; inject executor for SQL execution.")
                    self._executor = None
            else:
                self._executor = executor

            # Provider / tracer clients (optional)
            self._provider_client = provider_client
            self._tracer_client = tracer_client

            logger.debug(
                "Tools initialized (has_db=%s, has_schema=%s, has_vector=%s, has_executor=%s, has_provider=%s, has_tracer=%s)",
                bool(self._db),
                bool(self._schema_store),
                bool(self._vector_search),
                bool(self._executor),
                bool(self._provider_client),
                bool(self._tracer_client),
            )
        except Exception as e:
            logger.exception("Tools initialization failed")
            raise CustomException(e, sys)

    # ---------------------
    # Embedding wrapper builder
    # ---------------------
    def _build_embedding_wrapper(self, dim: int, config_module: Optional[Any]) -> Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]:
        """
        Build an embedding function that accepts either a single string or a list of strings.

        Strategy:
          - Try LangChain GoogleGenerativeAIEmbeddings (if available) lazily.
          - Otherwise use a deterministic char-based fallback (stable, fast, no network).
        """
        # Try langchain provider lazily
        try:
            # try the "langchain_google_genai" package first, then fall back to "langchain.embeddings"
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
                emb_model = getattr(config_module, "EMBEDDING_MODEL", "models/gemini-embedding-001") if config_module else "models/gemini-embedding-001"
                emb_client = GoogleGenerativeAIEmbeddings(model=emb_model)
            except Exception:
                from langchain.embeddings import GoogleGenerativeAIEmbeddings  # type: ignore
                emb_model = getattr(config_module, "EMBEDDING_MODEL", "models/gemini-embedding-001") if config_module else "models/gemini-embedding-001"
                emb_client = GoogleGenerativeAIEmbeddings(model=emb_model)

            logger.info("Tools: using GoogleGenerativeAIEmbeddings for embedding_fn")

            def _lc_emb(x: Union[str, List[str]]):
                if isinstance(x, (list, tuple)):
                    res = emb_client.embed_documents(list(x))
                    return [list(map(float, r)) for r in res]
                else:
                    res = emb_client.embed_query(x)
                    return list(map(float, res)) if res is not None else [0.0] * dim

            return _lc_emb
        except Exception as e:
            logger.debug("Tools: LangChain embeddings not available or failed to initialize: %s", e)

        # Deterministic fallback
        try:
            import numpy as _np
        except Exception:
            # If numpy missing, raise clear error (VectorSearch expects numpy). Keep fallback but will likely fail later.
            _np = None

        logger.info("Tools: using deterministic fallback embedding for embedding_fn")

        def _single_default_emb(text: str) -> List[float]:
            d = dim or 128
            if _np is None:
                # last-resort simple Python-only fallback
                vec = [0.0] * d
                s = str(text or "")
                for i, ch in enumerate(s.lower()):
                    vec[i % d] += (ord(ch) % 97) * 0.001
                # no normalization
                return vec
            vec = _np.zeros(d, dtype=float)
            s = str(text or "")
            for i, ch in enumerate(s.lower()):
                vec[i % d] += (ord(ch) % 97) * 0.001
            norm = _np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()

        def _wrapper(input_texts: Union[str, List[str]]):
            if isinstance(input_texts, (list, tuple)):
                return [_single_default_emb(str(t)) for t in input_texts]
            return _single_default_emb(str(input_texts))

        return _wrapper

    # ---------------------
    # Provider / Tracer accessors
    # ---------------------
    def get_provider_client(self) -> Optional[Any]:
        return self._provider_client

    def get_tracer_client(self) -> Optional[Any]:
        return self._tracer_client

    # ---------------------
    # SchemaStore adapters
    # ---------------------
    def get_schema(self, csv_name: str) -> Optional[List[str]]:
        try:
            if not self._schema_store:
                raise CustomException("SchemaStore backend not configured", sys)
            return self._schema_store.get_schema(csv_name)
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.get_schema failed")
            raise CustomException(e, sys)

    def get_sample_rows(self, csv_name: str) -> Optional[List[Dict[str, Any]]]:
        try:
            if not self._schema_store:
                raise CustomException("SchemaStore backend not configured", sys)
            return self._schema_store.get_sample_rows(csv_name)
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.get_sample_rows failed")
            raise CustomException(e, sys)

    def list_csvs(self) -> List[str]:
        try:
            if not self._schema_store:
                return []
            return self._schema_store.list_csvs()
        except Exception as e:
            logger.exception("Tools.list_csvs failed")
            raise CustomException(e, sys)

    # ---------------------
    # VectorSearch adapters
    # ---------------------
    def search_vectors(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            if not self._vector_search:
                raise CustomException("VectorSearch backend not configured", sys)
            if not hasattr(self._vector_search, "search"):
                raise CustomException("VectorSearch backend missing 'search' method", sys)
            return self._vector_search.search(query, top_k=top_k)
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.search_vectors failed")
            raise CustomException(e, sys)

    def upsert_vectors(self, docs: List[Dict[str, Any]]) -> List[str]:
        """
        Upsert documents into vector store. Each doc: {"id": <opt>, "text": <str>, "meta": <dict>}
        Returns inserted ids.
        """
        try:
            if not self._vector_search:
                raise CustomException("VectorSearch backend not configured", sys)
            if not hasattr(self._vector_search, "upsert_documents") and not hasattr(self._vector_search, "upsert"):
                # support both possible method names
                raise CustomException("VectorSearch backend does not implement upsert_documents/upsert", sys)

            if hasattr(self._vector_search, "upsert_documents"):
                return self._vector_search.upsert_documents(docs)
            else:
                return self._vector_search.upsert(docs)
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.upsert_vectors failed")
            raise CustomException(e, sys)

    def add_texts(self, texts: List[Dict[str, Any]]) -> List[str]:
        """
        Helper to add large texts (with chunking) into vector store.
        Each item: {"id": <opt>, "text": <str>, "meta": <dict>}
        """
        try:
            if not self._vector_search:
                raise CustomException("VectorSearch backend not configured", sys)
            if not hasattr(self._vector_search, "add_texts") and not hasattr(self._vector_search, "add_documents"):
                raise CustomException("VectorSearch backend does not implement add_texts/add_documents", sys)

            if hasattr(self._vector_search, "add_texts"):
                return self._vector_search.add_texts(texts)
            else:
                return self._vector_search.add_documents(texts)
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.add_texts failed")
            raise CustomException(e, sys)

    def clear_vectors(self) -> None:
        try:
            if not self._vector_search:
                raise CustomException("VectorSearch backend not configured", sys)
            if not hasattr(self._vector_search, "clear"):
                raise CustomException("VectorSearch backend missing 'clear' method", sys)
            self._vector_search.clear()
            logger.info("Tools: cleared vector store")
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.clear_vectors failed")
            raise CustomException(e, sys)

    def get_vector_meta(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            if not self._vector_search:
                raise CustomException("VectorSearch backend not configured", sys)
            # many vector backends will expose metadata dict directly or via method
            if hasattr(self._vector_search, "_metadata"):
                return getattr(self._vector_search, "_metadata", {}).get(doc_id)
            if hasattr(self._vector_search, "get_metadata"):
                return self._vector_search.get_metadata(doc_id)
            return None
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.get_vector_meta failed")
            raise CustomException(e, sys)

    # ---------------------
    # Database / SQL execution adapters
    # ---------------------
    def load_table(self, csv_path: str, table_name: str, force_reload: bool = False) -> None:
        try:
            if not self._db:
                raise CustomException("Database backend not configured", sys)
            if not hasattr(self._db, "load_csv_table"):
                raise CustomException("Database backend missing 'load_csv_table' method", sys)
            self._db.load_csv_table(csv_path, table_name, force_reload=force_reload)
            logger.info("Tools: loaded table %s from %s", table_name, csv_path)
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.load_table failed")
            raise CustomException(e, sys)

    def list_tables(self) -> List[str]:
        try:
            if not self._db:
                raise CustomException("Database backend not configured", sys)
            if not hasattr(self._db, "list_tables"):
                raise CustomException("Database backend missing 'list_tables' method", sys)
            return self._db.list_tables()
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.list_tables failed")
            raise CustomException(e, sys)

    def execute_sql(
        self,
        sql: str,
        read_only: bool = True,
        limit: Optional[int] = None,
        as_dataframe: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute SQL using configured executor.

        Return shape is executor-dependent but expected to be a dict like:
          {"rows": [...], "columns": [...], "rowcount": N, "as_dataframe": <pd.DataFrame> (if requested)}
        """
        try:
            if not self._executor:
                raise CustomException("Executor backend not configured", sys)

            if hasattr(self._executor, "execute_sql"):
                return self._executor.execute_sql(sql, read_only=read_only, limit=limit, as_dataframe=as_dataframe)
            elif callable(self._executor):
                # support function-style executor
                return self._executor(sql, read_only=read_only, limit=limit, as_dataframe=as_dataframe)
            else:
                raise CustomException("Executor backend does not implement execute_sql", sys)
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.execute_sql failed")
            raise CustomException(e, sys)

    # ---------------------
    # Utilities
    # ---------------------
    def generate_short_id(self) -> str:
        """Produce a short unique id for temp keys."""
        return uuid.uuid4().hex[:8]
