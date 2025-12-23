#D:\text_to_sql_bot\app\tools.py
from __future__ import annotations

import sys
import uuid
import os
from typing import Any, Dict, List, Optional

from app.logger import get_logger
from app.exception import CustomException

logger = get_logger("tools")


class Tools:
    """
    Central adapter for app-wide resources.

    🔑 CRITICAL RESPONSIBILITY:
    - Acts as the SINGLE SOURCE OF TRUTH for the LLM provider
    - GenerateNode MUST obtain its provider exclusively from here

    If get_provider_client() returns None → the system WILL fall back
    to deterministic SQL and natural-language understanding is LOST.
    """

    def __init__(
        self,
        db: Optional[Any] = None,
        schema_store: Optional[Any] = None,
        vector_search: Optional[Any] = None,
        executor: Optional[Any] = None,
        provider_client: Optional[Any] = None,
        tracer_client: Optional[Any] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ):
        try:
            # ---------------- Config ----------------
            self._config: Dict[str, Any] = config or {}

            # ---------------- Database ----------------
            if db is None:
                try:
                    from app import database as _database
                    self._db = _database
                except Exception:
                    self._db = None
            else:
                self._db = db

            # ---------------- Schema Store (SINGLETON) ----------------
            if schema_store is None:
                try:
                    from app.schema_store import SchemaStore
                    self._schema_store = SchemaStore.get_instance()
                except Exception:
                    self._schema_store = None
            else:
                self._schema_store = schema_store

            # ---------------- Vector Search ----------------
            self._vector_search = vector_search

            # ---------------- SQL Executor ----------------
            if executor is None:
                try:
                    from app import sql_executor as _executor
                    self._executor = _executor
                except Exception:
                    self._executor = None
            else:
                self._executor = executor

            # ---------------- LLM / Tracing ----------------
            self._provider_client = provider_client
            self._tracer_client = tracer_client

            # 🔥 AUTO-RESOLVE PROVIDER IF NOT INJECTED
            if self._provider_client is None:
                self._provider_client = self._resolve_provider_from_config()

            logger.info(
                "Tools initialized | provider=%s",
                self._provider_client.__class__.__name__
                if self._provider_client
                else None,
            )

        except Exception as e:
            logger.exception("Tools initialization failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # 🔐 SAFETY RULES
    # ------------------------------------------------------------------
    def get_safety_rules(self) -> Dict[str, Any]:
        default_rules = {
            "max_row_limit": 1000,
            "allow_write": False,
            "allow_drop": False,
        }
        return self._config.get("safety_rules", default_rules)

    # ------------------------------------------------------------------
    # SQL EXECUTION
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
        try:
            if not self._executor:
                raise CustomException("Executor backend not configured", sys)

            if hasattr(self._executor, "execute_sql"):
                fn = self._executor.execute_sql
                kwargs = {
                    "read_only": read_only,
                    "limit": limit,
                    "as_dataframe": as_dataframe,
                }
                if "table_map" in fn.__code__.co_varnames:
                    kwargs["table_map"] = table_map
                return fn(sql, **kwargs)

            if callable(self._executor):
                return self._executor(
                    sql,
                    read_only=read_only,
                    limit=limit,
                    as_dataframe=as_dataframe,
                )

            raise CustomException("Invalid executor backend", sys)

        except Exception as e:
            logger.exception("Tools.execute_sql failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------
    def get_schema(self, csv_name: str) -> Optional[List[str]]:
        if not self._schema_store:
            return None
        return self._schema_store.get_schema(csv_name)

    def list_csvs(self) -> List[str]:
        if not self._schema_store:
            return []
        return self._schema_store.list_csvs()

    def get_sample_rows(self, csv_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        if not self._schema_store:
            return []
        rows = self._schema_store.get_sample_rows(csv_name) or []
        return rows[:limit]

    # ------------------------------------------------------------------
    # Vector search (optional)
    # ------------------------------------------------------------------
    def search_vectors(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._vector_search or not hasattr(self._vector_search, "search"):
            logger.warning("Vector search not configured")
            return []
        try:
            return self._vector_search.search(query, top_k=top_k)
        except Exception:
            logger.exception("Vector search failed")
            return []

    # ------------------------------------------------------------------
    # 🔥 LLM PROVIDER (HARDENED)
    # ------------------------------------------------------------------
    def _resolve_provider_from_config(self) -> Optional[Any]:
        """
        Resolve and instantiate the LLM provider based on config/env.

        HARD GUARANTEES:
        - Never passes model twice
        - Never returns half-initialized client
        - Logs exact failure reason
        """
        try:
            provider = (
                self._config.get("llm_provider")
                or os.getenv("LLM_PROVIDER")
                or ""
            ).lower()

            if not provider:
                logger.warning("No LLM provider configured (LLM_PROVIDER missing)")
                return None

            if provider == "gemini":
                from app.gemini_client import GeminiClient

                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    logger.error("GEMINI_API_KEY not set")
                    return None

                client = GeminiClient(api_key=api_key)
                logger.info("Gemini LLM provider resolved via Tools")
                return client

            logger.error("Unsupported LLM provider: %s", provider)
            return None

        except Exception:
            logger.exception("Failed to resolve LLM provider")
            return None

    def get_provider_client(self) -> Optional[Any]:
        """
        Return the resolved LLM provider.

        ❗ If this returns None, GenerateNode WILL fall back to safe SQL.
        """
        return self._provider_client

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def generate_short_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def trace_event(self, name: str, data: Optional[Dict[str, Any]] = None):
        if not self._tracer_client or not hasattr(self._tracer_client, "trace"):
            return
        try:
            self._tracer_client.trace(name, data or {})
        except Exception:
            logger.exception("Tracing failed for event: %s", name)
