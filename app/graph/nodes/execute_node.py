# File: app/graph/nodes/execute_node.py
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

logger = get_logger("execute_node")
LOG = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Read-only SQL safety patterns
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
    r"\bREPLACE\b",
    r"\bMERGE\b",
]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _sanitize_sql(sql: str) -> str:
    """
    Normalize SQL safely:
      - strip whitespace
      - remove trailing semicolons
      - collapse repeated spaces
    """
    sql = (sql or "").strip()
    sql = re.sub(r";+\s*$", "", sql)
    sql = re.sub(r"\s+", " ", sql)
    return sql


def _is_read_only_sql(sql: str) -> bool:
    """
    Return True if SQL is read-only and safe to execute.
    """
    if not sql:
        return False

    if not re.match(r"^\s*(SELECT|WITH)\b", sql, flags=re.IGNORECASE):
        return False

    for patt in _READONLY_DISALLOWED:
        if re.search(patt, sql, flags=re.IGNORECASE):
            return False

    return True


# ------------------------------------------------------------------
# ExecuteNode
# ------------------------------------------------------------------
class ExecuteNode:
    """
    ExecuteNode
    -----------

    Responsibilities:
      - Enforce strict read-only SQL execution
      - Resolve canonical table names → CSV paths
      - Execute SQL via Tools.execute_sql
      - Return normalized execution payload for Graph/FormatNode
    """

    def __init__(
        self,
        tools: Optional[Tools] = None,
        default_limit: Optional[int] = None,
    ):
        try:
            self._tools = tools or Tools()
            self._default_limit = default_limit

            LOG.info(
                "ExecuteNode initialized | default_limit=%s",
                self._default_limit,
            )

        except Exception as e:
            logger.exception("ExecuteNode initialization failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Run (Graph-safe, keyword-only)
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        sql: Optional[str] = None,
        table_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
        read_only: bool = True,
        limit: Optional[int] = None,
        as_dataframe: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute validated SQL safely.

        Returns:
          {
            "rows": list | DataFrame,
            "columns": list,
            "rowcount": int,
            "meta": dict
          }
        """

        # --------------------------------------------------------------
        # Guard: SQL presence
        # --------------------------------------------------------------
        if not sql or not isinstance(sql, str):
            LOG.warning("ExecuteNode.run called with empty or invalid SQL")
            return {
                "rows": [],
                "columns": [],
                "rowcount": 0,
                "meta": {"reason": "empty_sql"},
            }

        sql = _sanitize_sql(sql)

        # --------------------------------------------------------------
        # Enforce read-only SQL
        # --------------------------------------------------------------
        if read_only and not _is_read_only_sql(sql):
            LOG.error("Blocked non read-only SQL execution: %s", sql)
            raise CustomException("Blocked non read-only SQL", sys)

        effective_limit = limit if limit is not None else self._default_limit

        # --------------------------------------------------------------
        # Build table_map: canonical_name → absolute csv_path
        # --------------------------------------------------------------
        table_map: Dict[str, str] = {}
        if table_schemas:
            for canonical, meta in table_schemas.items():
                if not isinstance(meta, dict):
                    continue
                path = meta.get("path") or meta.get("csv_path")
                if not path:
                    continue
                abs_path = os.path.abspath(path)
                if not os.path.exists(abs_path):
                    LOG.warning(
                        "CSV path missing for table '%s': %s",
                        canonical,
                        abs_path,
                    )
                    continue
                table_map[canonical] = abs_path

        LOG.debug("ExecuteNode resolved table_map: %s", table_map)

        # --------------------------------------------------------------
        # Execute SQL via Tools
        # --------------------------------------------------------------
        start_time = time.time()

        try:
            result = self._tools.execute_sql(
                sql=sql,
                table_map=table_map or None,
                read_only=read_only,
                limit=effective_limit,
                as_dataframe=as_dataframe,
            )

            runtime = time.time() - start_time

            # ----------------------------------------------------------
            # Normalize executor output
            # ----------------------------------------------------------
            rows = result.get("rows", []) if isinstance(result, dict) else []
            columns = result.get("columns", []) if isinstance(result, dict) else []
            rowcount = result.get(
                "rowcount",
                len(rows) if isinstance(rows, list) else None,
            )

            meta = result.get("meta", {}) if isinstance(result, dict) else {}
            meta.update(
                {
                    "runtime_sec": round(runtime, 4),
                    "read_only": read_only,
                    "limit": effective_limit,
                    "tables_used": list(table_map.keys()),
                }
            )

            LOG.info(
                "ExecuteNode success | rows=%s | time=%.3fs",
                rowcount,
                runtime,
            )

            return {
                "rows": rows,
                "columns": columns,
                "rowcount": rowcount,
                "meta": meta,
            }

        except CustomException:
            raise
        except Exception as e:
            LOG.exception("ExecuteNode.run failed")
            raise CustomException(e, sys)
