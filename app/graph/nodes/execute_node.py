# File: app/graph/nodes/execute_node.py
from __future__ import annotations

import sys
import time
import re
import os
import logging
from typing import Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

logger = get_logger("execute_node")
LOG = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Mutating SQL keywords (STATEMENTS, not functions)
# ------------------------------------------------------------------
_MUTATING_KEYWORDS = {
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "MERGE",
    "CALL",
    "EXEC",
    "EXECUTE",
    "VACUUM",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "COPY",
}

_ALLOWED_START = {"SELECT", "WITH", "EXPLAIN"}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _normalize_sql(sql: str) -> str:
    """
    Normalize SQL safely (no semantic changes).
    """
    sql = (sql or "").strip()
    sql = re.sub(r";+\s*$", "", sql)     # trailing semicolons
    sql = re.sub(r"\s+", " ", sql)       # collapse whitespace
    return sql


def _first_keyword(sql: str) -> Optional[str]:
    """
    Extract the first SQL keyword.
    """
    match = re.match(r"^\s*([A-Z]+)", sql, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def _contains_mutation(sql: str) -> bool:
    """
    Detect mutating SQL statements safely.

    Notes:
    - Token-based
    - Ignores string literals
    - Does NOT block REPLACE() function
    """
    # Remove quoted strings to avoid false positives
    scrubbed = re.sub(r"'[^']*'|\"[^\"]*\"", "", sql)

    tokens = re.findall(r"\b[A-Z_]+\b", scrubbed.upper())

    for tok in tokens:
        if tok in _MUTATING_KEYWORDS:
            return True

    return False


def _is_read_only(sql: str) -> bool:
    """
    Final read-only gate.
    """
    if not sql:
        return False

    first = _first_keyword(sql)
    if first not in _ALLOWED_START:
        return False

    if _contains_mutation(sql):
        return False

    return True


# ------------------------------------------------------------------
# ExecuteNode
# ------------------------------------------------------------------
class ExecuteNode:
    """
    ExecuteNode
    ===========

    Final execution authority.

    Enforces:
      - Read-only SQL
      - CSV-backed execution
      - Single execution path (Tools)

    NEVER:
      - Generates SQL
      - Mutates data
      - Silently modifies queries
    """

    def __init__(
        self,
        tools: Optional[Tools] = None,
        *,
        default_limit: Optional[int] = None,
    ):
        try:
            self.tools = tools or Tools()
            self.default_limit = default_limit

            LOG.info(
                "ExecuteNode initialized | default_limit=%s",
                self.default_limit,
            )

        except Exception as e:
            logger.exception("ExecuteNode initialization failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Graph entrypoint
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
        Execute SQL safely.

        Return contract:
        {
            "valid": bool,
            "sql": str,
            "rows": list,
            "columns": list,
            "rowcount": int,
            "error": Optional[str],
            "meta": dict
        }
        """
        start_time = time.time()

        if not sql or not isinstance(sql, str):
            LOG.warning("ExecuteNode called with empty SQL")
            return {
                "valid": False,
                "sql": "",
                "rows": [],
                "columns": [],
                "rowcount": 0,
                "error": "empty_sql",
                "meta": {},
            }

        sql = _normalize_sql(sql)

        LOG.info("ExecuteNode received SQL: %s", sql)

        # --------------------------------------------------
        # Hard safety gate (executor-level)
        # --------------------------------------------------
        if read_only and not _is_read_only(sql):
            LOG.error("Blocked non-read-only SQL: %s", sql)
            return {
                "valid": False,
                "sql": sql,
                "rows": [],
                "columns": [],
                "rowcount": 0,
                "error": "non_read_only_sql",
                "meta": {
                    "reason": "mutating_or_invalid_statement",
                },
            }

        effective_limit = limit if limit is not None else self.default_limit

        # --------------------------------------------------
        # Resolve table → CSV paths
        # --------------------------------------------------
        table_map: Dict[str, str] = {}

        if table_schemas:
            for table, meta in table_schemas.items():
                if not isinstance(meta, dict):
                    continue

                path = meta.get("path") or meta.get("csv_path")
                if not path:
                    continue

                abs_path = os.path.abspath(path)
                if not os.path.exists(abs_path):
                    LOG.warning(
                        "Missing CSV for table '%s': %s",
                        table,
                        abs_path,
                    )
                    continue

                table_map[table] = abs_path

        LOG.debug("Resolved table_map=%s", table_map)

        # --------------------------------------------------
        # Execute (single authority)
        # --------------------------------------------------
        try:
            result = self.tools.execute_sql(
                sql=sql,
                table_map=table_map or None,
                read_only=read_only,
                limit=effective_limit,
                as_dataframe=as_dataframe,
            )

            rows = result.get("rows", []) or []
            columns = result.get("columns", []) or []
            meta = result.get("meta", {}) or {}

            runtime = round(time.time() - start_time, 4)

            meta.update(
                {
                    "runtime_sec": runtime,
                    "read_only": read_only,
                    "limit": effective_limit,
                    "tables_used": list(table_map.keys()),
                }
            )

            LOG.info(
                "ExecuteNode success | rows=%d | time=%.3fs",
                len(rows),
                runtime,
            )

            return {
                "valid": True,
                "sql": sql,
                "rows": rows,
                "columns": columns,
                "rowcount": meta.get("rowcount", len(rows)),
                "error": None,
                "meta": meta,
            }

        except CustomException as ce:
            LOG.exception("SQL execution failed")
            return {
                "valid": False,
                "sql": sql,
                "rows": [],
                "columns": [],
                "rowcount": 0,
                "error": str(ce),
                "meta": {},
            }

        except Exception as e:
            LOG.exception("Unexpected executor failure")
            return {
                "valid": False,
                "sql": sql,
                "rows": [],
                "columns": [],
                "rowcount": 0,
                "error": str(e),
                "meta": {},
            }
