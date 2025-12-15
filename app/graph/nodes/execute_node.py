#D:\text_to_sql_bot\app\graph\nodes\execute_node.py
from __future__ import annotations

import sys
import time
import re
import logging
import os
from typing import Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools
from app import database

logger = get_logger("execute_node")
LOG = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Read-only SQL safety
# ------------------------------------------------------------------
_READONLY_DISALLOWED = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bUPDATE\b",
    r"\bINSERT\b",
    r"\bALTER\b",
    r"\bTRUNCATE\b",
    r"\bCREATE\b",
    r"\bATTACH\b",
    r"\bDETACH\b",
]


def _is_read_only_sql(sql: str) -> bool:
    """
    Returns True if the SQL contains only read-only operations.
    """
    for patt in _READONLY_DISALLOWED:
        if re.search(patt, sql, flags=re.IGNORECASE):
            return False
    return True


def _sanitize_sql(sql: str) -> str:
    """
    Normalize SQL:
    - trim whitespace
    - remove trailing semicolons
    - collapse repeated whitespace
    """
    sql = (sql or "").strip()
    sql = re.sub(r";+$", "", sql)
    sql = re.sub(r"\s+", " ", sql)
    return sql


class ExecuteNode:
    """
    ExecuteNode
    -----------
    Responsibilities:
    - Build canonical → csv_path table_map
    - Ensure CSV-backed tables are loaded into DuckDB
    - Pass table_map into sql_executor (CRITICAL FIX)
    - Enforce read-only SQL safety
    """

    def __init__(
        self,
        tools: Optional[Tools] = None,
        default_limit: Optional[int] = None,
    ):
        self._tools = tools or Tools()
        self._default_limit = default_limit

    def run(
        self,
        sql: str,
        *,
        read_only: bool = True,
        limit: Optional[int] = None,
        as_dataframe: bool = False,
        table_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:

        # --------------------------------------------------------------
        # Guard clauses
        # --------------------------------------------------------------
        if not sql:
            return {"rows": [], "columns": [], "rowcount": 0}

        sql = _sanitize_sql(sql)

        if read_only and not _is_read_only_sql(sql):
            raise CustomException("Non read-only SQL blocked", sys)

        effective_limit = limit if limit is not None else self._default_limit

        # --------------------------------------------------------------
        # 🔑 BUILD table_map (canonical_name → csv_path)
        # --------------------------------------------------------------
        table_map: Dict[str, str] = {}

        if table_schemas:
            for canonical, meta in table_schemas.items():
                if not isinstance(meta, dict):
                    continue

                path = meta.get("path") or meta.get("csv_path")
                if path and os.path.exists(path):
                    table_map[canonical] = os.path.abspath(path)

        LOG.debug("ExecuteNode: resolved table_map=%s", table_map)

        start = time.time()

        try:
            # ----------------------------------------------------------
            # ✅ CRITICAL FIX
            # table_map MUST be passed to sql_executor so that:
            # 1. CSVs are loaded into DuckDB
            # 2. Canonical names are rewritten to DuckDB-safe identifiers
            # ----------------------------------------------------------
            result = self._tools.execute_sql(
                sql,
                table_map=table_map if table_map else None,
                read_only=read_only,
                limit=effective_limit,
                as_dataframe=as_dataframe,
            )

            runtime = time.time() - start
            LOG.info("ExecuteNode: SQL executed in %.3fs", runtime)

            return {
                "rows": result.get("rows", []),
                "columns": result.get("columns", []),
                "rowcount": result.get("meta", {}).get("rowcount", 0),
            }

        except Exception as e:
            LOG.exception("ExecuteNode.run failed")
            raise CustomException(e, sys)
