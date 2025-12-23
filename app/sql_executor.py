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
# SQL SAFETY (READ-ONLY)
# ------------------------------------------------------------------
# NOTE:
# - SELECT is ALWAYS allowed
# - WITH / CTE is allowed
# - EXPLAIN is allowed
# - Only state-mutating commands are blocked
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
    "MERGE",
    "REPLACE",
    "TRUNCATE",
)


def _validate_sql(sql: str, read_only: bool) -> None:
    """
    Block dangerous SQL keywords when in read-only mode.
    SELECT queries are ALWAYS allowed.
    """
    if not read_only:
        return

    upper = sql.upper()

    # Allow EXPLAIN SELECT ...
    if upper.lstrip().startswith(("SELECT", "WITH", "EXPLAIN")):
        return

    for kw in _READONLY_DISALLOWED:
        if re.search(rf"\b{kw}\b", upper):
            raise CustomException(
                f"Disallowed SQL keyword in read-only mode: {kw}",
                sys,
            )


def _strip_trailing_semicolon(sql: str) -> str:
    """
    DuckDB-safe SQL cleanup.
    """
    sql = sql.strip()
    if sql.endswith(";"):
        return sql[:-1]
    return sql


def _rows_to_dicts(
    rows: List[Tuple[Any, ...]],
    columns: List[str],
) -> List[Dict[str, Any]]:
    """
    Convert DB rows to list-of-dicts.
    """
    output: List[Dict[str, Any]] = []
    for row in rows:
        if columns:
            output.append(dict(zip(columns, row)))
        else:
            output.append({f"col{i}": v for i, v in enumerate(row)})
    return output


# ------------------------------------------------------------------
# CANONICAL → PHYSICAL TABLE REWRITE
# ------------------------------------------------------------------
def _rewrite_sql_table_identifiers(
    sql: str,
    table_map: Dict[str, str],
) -> str:
    """
    Rewrite canonical schema names → DuckDB physical tables.

    Handles:
    - bare identifiers
    - `backticks`
    - "double quotes"
    """
    rewritten = sql
    mapping: Dict[str, str] = {}

    for canonical in table_map.keys():
        try:
            physical = database._sanitize_table_name(canonical)
        except Exception:
            cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", canonical)
            if cleaned and cleaned[0].isdigit():
                cleaned = f"t_{cleaned}"
            physical = cleaned or "table"

        mapping[canonical] = physical

    # Replace longest identifiers first
    for canonical in sorted(mapping, key=len, reverse=True):
        physical = mapping[canonical]
        esc = re.escape(canonical)

        rewritten = re.sub(rf"`\s*{esc}\s*`", physical, rewritten, flags=re.I)
        rewritten = re.sub(rf"\"\s*{esc}\s*\"", physical, rewritten, flags=re.I)
        rewritten = re.sub(
            rf"(?<![\w\.]){esc}(?![\w\.])",
            physical,
            rewritten,
            flags=re.I,
        )

    if rewritten != sql:
        logger.debug("SQL rewrite\nBEFORE: %s\nAFTER : %s", sql, rewritten)

    return rewritten


# ------------------------------------------------------------------
# MAIN EXECUTOR
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
    Safe, resume-proof SQL execution.

    Returns:
    {
        rows: list[dict],
        columns: list[str],
        meta: {
            rowcount: int,
            runtime: float
        }
    }
    """
    start = time.time()

    try:
        # -----------------------------
        # Guard clauses
        # -----------------------------
        if not isinstance(sql, str):
            raise CustomException("SQL must be a string", sys)

        sql = sql.strip()
        if not sql:
            raise CustomException("Empty SQL provided", sys)

        _validate_sql(sql, read_only)

        # -----------------------------
        # ALWAYS reload tables (resume-safe)
        # -----------------------------
        if table_map:
            database.ensure_tables_loaded(table_map)

        # -----------------------------
        # Normalize SQL
        # -----------------------------
        exec_sql = _strip_trailing_semicolon(sql)

        if table_map:
            exec_sql = _rewrite_sql_table_identifiers(exec_sql, table_map)

        # -----------------------------
        # Optional LIMIT wrapper
        # -----------------------------
        if limit is not None:
            exec_sql = (
                "SELECT * FROM ("
                f"{exec_sql}"
                ") AS _text_to_sql_subquery "
                f"LIMIT {int(limit)}"
            )

        logger.info("Executing SQL: %s", exec_sql)

        rows_or_df, columns, meta = database.execute_query(
            exec_sql,
            read_only=read_only,
            as_dataframe=as_dataframe,
        )

        meta = meta or {}
        runtime = round(time.time() - start, 4)

        # -----------------------------
        # Normalize rows
        # -----------------------------
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
