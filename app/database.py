from __future__ import annotations

import os
import sys
import time
import re
from typing import List, Tuple, Any, Dict, Optional

import duckdb

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("database")

# ============================================================
# Module-level DuckDB connection cache
# ============================================================
_CONN: Optional[duckdb.DuckDBPyConnection] = None


# ============================================================
# Identifier sanitization (CRITICAL)
# ============================================================
def _sanitize_table_name(canonical_name: str) -> str:
    """
    Convert canonical table name → valid DuckDB identifier.

    Rules:
    - only [a-z0-9_]
    - lowercased
    - cannot start with digit → prefix with `t_`
    - empty names become `t_table`
    """
    if not isinstance(canonical_name, str) or not canonical_name.strip():
        return "t_table"

    name = canonical_name.strip()
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_").lower()

    if not name:
        name = "table"

    if name[0].isdigit():
        name = f"t_{name}"

    return name


# ============================================================
# Connection management
# ============================================================
def get_connection() -> duckdb.DuckDBPyConnection:
    global _CONN

    try:
        db_path = getattr(config, "DATABASE_PATH", None)
        if not db_path:
            db_path = os.path.join(os.getcwd(), "data", "text_to_sql.duckdb")

        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        if _CONN is None:
            logger.info("Opening DuckDB connection: %s", db_path)
            _CONN = duckdb.connect(database=db_path, read_only=False)

        return _CONN

    except Exception as e:
        logger.exception("Failed to open DuckDB connection")
        raise CustomException(e, sys)


def close_connection() -> None:
    global _CONN
    if _CONN:
        try:
            _CONN.close()
        finally:
            _CONN = None


# ============================================================
# SQL Safety (last line of defense)
# ============================================================
_DISALLOWED = (
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "TRUNCATE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "COPY",
    "CALL",
    "EXECUTE",
    "CREATE",
    "MERGE",
    "REPLACE",
)


def _validate_read_only_sql(sql: str) -> None:
    upper = sql.upper()
    for kw in _DISALLOWED:
        if re.search(rf"\b{kw}\b", upper):
            raise PermissionError(f"Blocked SQL keyword: {kw}")


# ============================================================
# DuckDB helpers
# ============================================================
def _duckdb_has_table(
    con: duckdb.DuckDBPyConnection,
    physical_name: str,
) -> bool:
    rows = con.execute("SHOW TABLES").fetchall()
    return physical_name.lower() in {r[0].lower() for r in rows}


_TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z0-9_\"`]+)",
    flags=re.IGNORECASE,
)


def _extract_table_refs(sql: str) -> List[str]:
    """
    Extract raw (canonical) table references from SQL.
    """
    found: List[str] = []
    for m in _TABLE_REF_RE.finditer(sql):
        raw = m.group(1)
        if raw:
            found.append(raw.strip("`\""))
    return list(dict.fromkeys(found))


# ============================================================
# CSV loading (STABLE + IDEMPOTENT)
# ============================================================
def load_csv_table(
    csv_path: str,
    canonical_name: str,
    *,
    force_reload: bool = False,
) -> str:
    """
    Load CSV into DuckDB as a VIEW using a sanitized physical name.

    Returns
    -------
    str
        Physical DuckDB table/view name
    """
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        con = get_connection()
        physical_name = _sanitize_table_name(canonical_name)

        if _duckdb_has_table(con, physical_name) and not force_reload:
            logger.debug("DuckDB table already exists: %s", physical_name)
            return physical_name

        safe_path = csv_path.replace("'", "''")

        sql = f"""
        CREATE OR REPLACE VIEW {physical_name} AS
        SELECT *
        FROM read_csv_auto(
            '{safe_path}',
            all_varchar = true,
            null_padding = true,
            ignore_errors = true
        )
        """

        logger.info("Loading CSV %s → %s", csv_path, physical_name)
        con.execute(sql)

        return physical_name

    except Exception as e:
        logger.exception("Failed to load CSV")
        raise CustomException(e, sys)


def ensure_tables_loaded(
    table_map: Dict[str, str],
    *,
    force_reload: bool = False,
) -> Dict[str, str]:
    """
    Ensure all canonical tables are loaded.

    Input
    -----
    canonical_name -> csv_path

    Output
    ------
    canonical_name -> physical_table_name
    """
    resolved: Dict[str, str] = {}

    for canonical, path in table_map.items():
        physical = load_csv_table(
            path,
            canonical,
            force_reload=force_reload,
        )
        resolved[canonical] = physical

    return resolved


# ============================================================
# Query execution (FINAL EXECUTION LAYER)
# ============================================================
def execute_query(
    sql: str,
    *,
    read_only: bool = True,
    as_dataframe: bool = False,
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    Execute SQL against DuckDB.

    Assumes:
    - canonical → physical rewrite already happened
    - required tables already loaded
    """
    start = time.time()

    try:
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("SQL must be a non-empty string")

        if read_only:
            _validate_read_only_sql(sql)

        con = get_connection()
        cur = con.execute(sql)

        rows = cur.fetchall()
        columns = [c[0] for c in cur.description] if cur.description else []

        meta = {
            "rowcount": len(rows),
            "runtime": round(time.time() - start, 4),
        }

        if as_dataframe:
            import pandas as pd
            return pd.DataFrame(rows, columns=columns), columns, meta

        return rows, columns, meta

    except Exception as e:
        logger.exception("DuckDB query execution failed")
        raise CustomException(e, sys)


# ============================================================
# Table utilities
# ============================================================
def table_exists(canonical_name: str) -> bool:
    con = get_connection()
    return _duckdb_has_table(con, _sanitize_table_name(canonical_name))


def list_tables() -> List[str]:
    con = get_connection()
    return [r[0] for r in con.execute("SHOW TABLES").fetchall()]
