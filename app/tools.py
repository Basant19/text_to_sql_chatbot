# app/tools.py

import sys
from typing import Any, Dict, List, Optional

from app.logger import get_logger
from app.exception import CustomException

logger = get_logger("tools")


class Tools:
    """
    Thin adapter layer exposing simple functions for Graph nodes or LLM flow.
    Supports dependency injection of underlying components for easier testing.
    """

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
        Initialize tools with optional injected dependencies. Defaults will be used if
        specific components are not provided (DB, SchemaStore, VectorSearch, SQL executor).
        """
        try:
            # Lazy import config if needed
            try:
                from app import config as _config
            except Exception:
                _config = None

            # -------- Database default --------
            if db is None:
                from app import database as _database
                self._db = _database
            else:
                self._db = db

            # -------- SchemaStore default --------
            if schema_store is None:
                from app.schema_store import SchemaStore
                self._schema_store = SchemaStore()
            else:
                self._schema_store = schema_store

            # -------- VectorSearch default --------
            if vector_search is None:
                try:
                    from app.vector_search import VectorSearch
                    index_path = getattr(_config, "VECTOR_INDEX_PATH", None) if _config else None
                    index_path = index_path or "./faiss/index.faiss"
                    self._vector_search = VectorSearch(index_path=index_path, embedding_fn=None)
                except Exception:
                    logger.warning(
                        "VectorSearch default initialization failed; vector search must be injected."
                    )
                    self._vector_search = None
            else:
                self._vector_search = vector_search

            # -------- Executor default --------
            if executor is None:
                try:
                    from app import sql_executor as _executor
                    self._executor = _executor
                except Exception:
                    logger.warning("Default sql_executor import failed; executor must be injected.")
                    self._executor = None
            else:
                self._executor = executor

            # -------- Provider / Tracer clients --------
            self._provider_client = provider_client
            self._tracer_client = tracer_client

            logger.debug(
                f"Tools initialized (has_provider={bool(self._provider_client)}, has_tracer={bool(self._tracer_client)})"
            )

        except Exception as e:
            logger.exception("Failed to initialize Tools")
            raise CustomException(e, sys)

    # -------- Provider / Tracer accessors --------
    def get_provider_client(self) -> Optional[Any]:
        """Return the injected provider client (e.g., GeminiClient) or None."""
        return self._provider_client

    def get_tracer_client(self) -> Optional[Any]:
        """Return the injected tracer client (e.g., LangSmithClient) or None."""
        return self._tracer_client

    # -------- Database adapters --------
    def load_table(self, csv_path: str, table_name: str, force_reload: bool = False) -> None:
        try:
            if not hasattr(self._db, "load_csv_table"):
                raise CustomException("Database backend does not implement load_csv_table", sys)
            self._db.load_csv_table(csv_path, table_name, force_reload=force_reload)
            logger.info(f"Tools: loaded table {table_name} from {csv_path}")
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.load_table failed")
            raise CustomException(e, sys)

    def list_tables(self) -> List[str]:
        try:
            if not hasattr(self._db, "list_tables"):
                raise CustomException("Database backend does not implement list_tables", sys)
            return self._db.list_tables()
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.list_tables failed")
            raise CustomException(e, sys)

    # -------- SchemaStore adapters --------
    def get_schema(self, csv_name: str) -> Optional[List[str]]:
        try:
            return self._schema_store.get_schema(csv_name)
        except Exception as e:
            logger.exception("Tools.get_schema failed")
            raise CustomException(e, sys)

    def get_sample_rows(self, csv_name: str) -> Optional[List[Dict[str, Any]]]:
        try:
            return self._schema_store.get_sample_rows(csv_name)
        except Exception as e:
            logger.exception("Tools.get_sample_rows failed")
            raise CustomException(e, sys)

    def list_csvs(self) -> List[str]:
        try:
            if hasattr(self._schema_store, "list_csvs"):
                return self._schema_store.list_csvs()
            return []
        except Exception as e:
            logger.exception("Tools.list_csvs failed")
            raise CustomException(e, sys)

    # -------- Vector search adapter --------
    def search_vectors(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            if self._vector_search is None:
                raise CustomException("VectorSearch backend is not configured", sys)
            if not hasattr(self._vector_search, "search"):
                raise CustomException("VectorSearch backend does not implement search()", sys)
            return self._vector_search.search(query, top_k=top_k)
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.search_vectors failed")
            raise CustomException(e, sys)

    # -------- SQL execution adapter --------
    def execute_sql(
        self,
        sql: str,
        read_only: bool = True,
        limit: Optional[int] = None,
        as_dataframe: bool = False,
    ) -> Dict[str, Any]:
        try:
            if self._executor is None:
                raise CustomException("Executor backend is not configured", sys)

            if hasattr(self._executor, "execute_sql"):
                return self._executor.execute_sql(sql, read_only=read_only, limit=limit, as_dataframe=as_dataframe)
            elif callable(self._executor):
                return self._executor(sql, read_only=read_only, limit=limit, as_dataframe=as_dataframe)
            else:
                raise CustomException("Executor does not implement execute_sql", sys)
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.execute_sql failed")
            raise CustomException(e, sys)
