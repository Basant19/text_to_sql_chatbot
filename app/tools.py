# app/tools.py
import sys
import time
from typing import Any, Dict, List, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("tools")


class Tools:
    """
    Thin adapter layer that exposes simple functions for the Graph nodes to call.
    Allows dependency injection of the underlying components for easy testing.

    Typical usage:
      t = Tools()
      t.load_table(path, table_name)
      schema = t.get_schema("people")
      docs = t.search_vectors("find people", top_k=5)
      result = t.execute_sql("SELECT ...", read_only=True)
    """

    def __init__(
        self,
        db: Optional[Any] = None,
        schema_store: Optional[Any] = None,
        vector_search: Optional[Any] = None,
        executor: Optional[Any] = None,
    ):
        try:
            # Lazy import default components to avoid heavy imports at module import time
            self._db = db or __import__("app.database", fromlist=[""]).database if False else db
            # Note: __import__ trick above is inert (keeps linter happy). We'll import inside try/except below.

            # Real defaults
            if db is None:
                from app import database as _database

                self._db = _database
            if schema_store is None:
                from app.schema_store import SchemaStore

                self._schema_store = SchemaStore()
            else:
                self._schema_store = schema_store
            if vector_search is None:
                from app.vector_search import VectorSearch

                # default index path taken from config
                try:
                    vs = VectorSearch(index_path=getattr(config, "VECTOR_INDEX_PATH", None), embedding_fn=None)
                except Exception:
                    # If faiss isn't available or something fails, still allow injection later
                    vs = VectorSearch(index_path=getattr(config, "VECTOR_INDEX_PATH", "./faiss/index.faiss"), embedding_fn=None)
                self._vector_search = vs
            else:
                self._vector_search = vector_search
            if executor is None:
                from app import sql_executor as _executor

                self._executor = _executor
            else:
                self._executor = executor
        except Exception as e:
            logger.exception("Failed to initialize Tools")
            raise CustomException(e, sys)

    # ---------- Database adapters ----------
    def load_table(self, csv_path: str, table_name: str, force_reload: bool = False) -> None:
        """
        Load a CSV into DuckDB (via database module). Returns None or raises CustomException.
        """
        try:
            # database.load_csv_table is expected to exist
            if not hasattr(self._db, "load_csv_table"):
                raise CustomException("database backend does not implement load_csv_table", sys)
            self._db.load_csv_table(csv_path, table_name, force_reload=force_reload)
            logger.info(f"Tools: loaded table {table_name} from {csv_path}")
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.load_table failed")
            raise CustomException(e, sys)

    def list_tables(self) -> List[str]:
        """
        Return list of tables known to the DuckDB database wrapper.
        """
        try:
            if not hasattr(self._db, "list_tables"):
                # fallback: database may expose other API; raise helpful error
                raise CustomException("database backend does not implement list_tables", sys)
            return self._db.list_tables()
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.list_tables failed")
            raise CustomException(e, sys)

    # ---------- Schema store adapters ----------
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

    # ---------- Vector search adapter ----------
    def search_vectors(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector DB and return list of docs (with id, score, text, meta).
        """
        try:
            if not hasattr(self._vector_search, "search"):
                raise CustomException("vector_search backend does not implement search()", sys)
            return self._vector_search.search(query, top_k=top_k)
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.search_vectors failed")
            raise CustomException(e, sys)

    # ---------- SQL execution adapter ----------
    def execute_sql(self, sql: str, read_only: bool = True, limit: Optional[int] = None, as_dataframe: bool = False) -> Dict[str, Any]:
        """
        Execute SQL safely via the sql_executor module and return execution result dict.
        """
        try:
            if hasattr(self._executor, "execute_sql"):
                return self._executor.execute_sql(sql, read_only=read_only, limit=limit, as_dataframe=as_dataframe)
            else:
                # If executor module was injected directly, it might be a callable/function
                if callable(self._executor):
                    return self._executor(sql, read_only=read_only, limit=limit, as_dataframe=as_dataframe)
                raise CustomException("executor does not implement execute_sql", sys)
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.execute_sql failed")
            raise CustomException(e, sys)
