# app/tools.py
"""
Tools - light dependency-adapter used across graph nodes and app logic.
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
        *,
        auto_init_vector: bool = False,
        vector_index_path: Optional[str] = None,
    ):
        try:
            # ---------------- Database ----------------
            if db is None:
                try:
                    from app import database as _database
                    self._db = _database
                except Exception:
                    self._db = None
                    logger.debug("Tools: DB backend not available")
            else:
                self._db = db

            # ---------------- Schema Store ----------------
            if schema_store is None:
                try:
                    from app.schema_store import SchemaStore
                    self._schema_store = SchemaStore()
                except Exception:
                    self._schema_store = None
            else:
                self._schema_store = schema_store

            # ---------------- Vector Search ----------------
            self._vector_search = vector_search

            # ---------------- Executor ----------------
            if executor is None:
                try:
                    from app import sql_executor as _executor
                    self._executor = _executor
                except Exception:
                    self._executor = None
            else:
                self._executor = executor

            self._provider_client = provider_client
            self._tracer_client = tracer_client

            logger.debug(
                "Tools initialized (db=%s, schema=%s, vector=%s, executor=%s)",
                bool(self._db),
                bool(self._schema_store),
                bool(self._vector_search),
                bool(self._executor),
            )
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # SQL EXECUTION  ✅ FIX IS HERE
    # ------------------------------------------------------------------
    def execute_sql(
        self,
        sql: str,
        *,
        table_map: Optional[Dict[str, str]] = None,
        read_only: bool = True,
        limit: Optional[int] = None,
        as_dataframe: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute SQL via executor.

        table_map:
            Optional mapping of logical table names → canonical DB table names.
            Required by graph ExecuteNode but optional for executor.
        """
        try:
            if not self._executor:
                raise CustomException("Executor backend not configured", sys)

            # Preferred: executor.execute_sql(...)
            if hasattr(self._executor, "execute_sql"):
                fn = self._executor.execute_sql

                # Check if executor supports table_map
                if "table_map" in fn.__code__.co_varnames:
                    return fn(
                        sql,
                        table_map=table_map,
                        read_only=read_only,
                        limit=limit,
                        as_dataframe=as_dataframe,
                    )

                # Backward-compatible call
                return fn(
                    sql,
                    read_only=read_only,
                    limit=limit,
                    as_dataframe=as_dataframe,
                )

            # Function-style executor fallback
            if callable(self._executor):
                return self._executor(
                    sql,
                    read_only=read_only,
                    limit=limit,
                    as_dataframe=as_dataframe,
                )

            raise CustomException("Executor backend does not support SQL execution", sys)

        except CustomException:
            raise
        except Exception as e:
            logger.exception("Tools.execute_sql failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------
    def get_schema(self, csv_name: str):
        if not self._schema_store:
            return None
        return self._schema_store.get_schema(csv_name)

    def list_csvs(self) -> List[str]:
        if not self._schema_store:
            return []
        return self._schema_store.list_csvs()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def generate_short_id(self) -> str:
        return uuid.uuid4().hex[:8]
