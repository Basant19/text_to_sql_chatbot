# app/sql_executor.py
from __future__ import annotations

import sys
import time
import re
from typing import List, Dict, Any, Tuple, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import database

logger = get_logger("sql_executor")

# ------------------------------------------------------------------
# SQL Safety (read-only mode)
# ------------------------------------------------------------------
_READONLY_DISALLOWED = (
    "DROP",
    "DELETE",
    "ALTER",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "COPY",
    "CALL",
    "EXECUTE",
    "UPDATE",
    "INSERT",
    "CREATE",
)


def _validate_sql(sql: str, read_only: bool) -> None:
    """
    Block dangerous SQL keywords when in read-only mode.
    """
    if not read_only:
        return

    upper_sql = sql.upper()
    for kw in _READONLY_DISALLOWED:
        if kw in upper_sql:
            raise CustomException(
                f"Disallowed SQL for read-only mode: {kw}",
                sys,
            )


def _strip_trailing_semicolon(sql: str) -> str:
    """
    Remove trailing semicolon (DuckDB subquery-safe).
    """
    s = sql.rstrip()
    return s[:-1] if s.endswith(";") else s


def _rows_to_dicts(
    rows: List[Tuple[Any, ...]],
    columns: List[str],
) -> List[Dict[str, Any]]:
    """
    Convert rows + columns into list[dict].
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not columns:
            out.append({f"col{i}": v for i, v in enumerate(r)})
        else:
            out.append({col: v for col, v in zip(columns, r)})
    return out


# ------------------------------------------------------------------
# Identifier rewriting (CRITICAL FIX)
# ------------------------------------------------------------------

def _rewrite_sql_table_identifiers(
    sql: str,
    table_map: Dict[str, str],
) -> str:
    """
    Rewrite canonical table identifiers in SQL to sanitized DuckDB
    physical table names.

    Handles:
      - bare identifiers
      - backticks:  `table`
      - double quotes: "table"

    Example:
      FROM 7abc_table
      FROM `7abc_table`
      FROM "7abc_table"

      -> FROM t_7abc_table
    """
    rewritten = sql

    # Build canonical -> physical mapping
    mapping: Dict[str, str] = {}
    for canonical in table_map.keys():
        try:
            physical = database._sanitize_table_name(canonical)
        except Exception:
            # defensive fallback
            cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", canonical)
            if cleaned and cleaned[0].isdigit():
                cleaned = f"t_{cleaned}"
            physical = cleaned or "table"

        mapping[canonical] = physical

    # Replace longest first to avoid partial collisions
    for canonical in sorted(mapping.keys(), key=len, reverse=True):
        physical = mapping[canonical]
        esc = re.escape(canonical)

        # backticks
        rewritten = re.sub(
            rf"`\s*{esc}\s*`",
            physical,
            rewritten,
            flags=re.IGNORECASE,
        )

        # double quotes
        rewritten = re.sub(
            rf"\"\s*{esc}\s*\"",
            physical,
            rewritten,
            flags=re.IGNORECASE,
        )

        # bare identifiers
        rewritten = re.sub(
            rf"(?<![A-Za-z0-9_\.]){esc}(?![A-Za-z0-9_\.])",
            physical,
            rewritten,
            flags=re.IGNORECASE,
        )

    if rewritten != sql:
        logger.debug(
            "SQL table rewrite applied:\n  BEFORE: %s\n  AFTER : %s",
            sql,
            rewritten,
        )

    return rewritten


# ------------------------------------------------------------------
# Main Executor
# ------------------------------------------------------------------

def execute_sql(
    sql: str,
    *,
    table_map: Optional[Dict[str, str]] = None,
    read_only: bool = True,
    limit: Optional[int] = None,
    as_dataframe: bool = False,
) -> Dict[str, Any]:
    """
    Execute SQL safely with canonical → physical table rewrite.

    Parameters
    ----------
    sql : str
        SQL query (LLM-generated).
    table_map : dict | None
        canonical_table_name -> csv_path
    read_only : bool
        Enforce read-only SQL safety.
    limit : int | None
        Optional LIMIT wrapper.
    as_dataframe : bool
        Return pandas DataFrame if True.

    Returns
    -------
    dict with keys: rows, columns, meta
    """
    start = time.time()

    try:
        if not isinstance(sql, str):
            raise CustomException("SQL must be a string", sys)

        sql = sql.strip()
        if not sql:
            raise CustomException("Empty SQL provided", sys)

        _validate_sql(sql, read_only)

        # --------------------------------------------------------------
        # Ensure CSV-backed tables are loaded (PHYSICAL TABLES)
        # --------------------------------------------------------------
        if table_map:
            database.ensure_tables_loaded(table_map)

        exec_sql = _strip_trailing_semicolon(sql)

        # --------------------------------------------------------------
        # ✅ CRITICAL FIX:
        # Rewrite canonical identifiers → sanitized DuckDB tables
        # --------------------------------------------------------------
        if table_map:
            exec_sql = _rewrite_sql_table_identifiers(exec_sql, table_map)

        # Optional LIMIT wrapper
        if limit is not None:
            exec_sql = (
                f"SELECT * FROM ({exec_sql}) AS _texttosql_sub "
                f"LIMIT {int(limit)}"
            )

        logger.info("Executing SQL (final): %s", exec_sql)

        rows_or_df, columns, meta = database.execute_query(
            exec_sql,
            read_only=read_only,
            as_dataframe=as_dataframe,
        )

        meta = meta or {}
        runtime = meta.get("runtime", time.time() - start)

        if as_dataframe:
            try:
                import pandas as pd
                if isinstance(rows_or_df, pd.DataFrame):
                    rows = rows_or_df.to_dict(orient="records")
                else:
                    rows = _rows_to_dicts(rows_or_df, columns)
            except Exception:
                rows = _rows_to_dicts(rows_or_df, columns)
        else:
            rows = _rows_to_dicts(rows_or_df, columns)

        return {
            "rows": rows,
            "columns": list(columns or []),
            "meta": {
                "rowcount": meta.get("rowcount", len(rows)),
                "runtime": runtime,
            },
        }

    except CustomException:
        raise
    except Exception as e:
        logger.exception("SQL execution failed")
        raise CustomException(e, sys)
