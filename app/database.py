# app/database.py
from __future__ import annotations

import os
import sys
import time
import re
from typing import List, Tuple, Any, Dict, Optional, Union

import duckdb

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("database")

# ============================================================
# Module-level DuckDB connection cache
# ============================================================
_conn: Optional[duckdb.DuckDBPyConnection] = None


# ============================================================
# Identifier sanitization (🔥 CRITICAL FIX)
# ============================================================
def sanitize_table_name(canonical_name: str) -> str:
    """
    Convert canonical table name → valid DuckDB identifier.

    Rules:
    - only [a-z0-9_]
    - lowercased
    - cannot start with digit → prefix with `t_`
    - empty names become `t_table`

    Examples
    --------
    "2023_sales"   -> "t_2023_sales"
    "Sales Data"   -> "sales_data"
    "1-table.csv"  -> "t_1_table_csv"
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
# SQL normalization helpers
# ============================================================
def _normalize_sql_variants(sql: str) -> List[Tuple[str, str]]:
    """
    DuckDB is picky about quoting.
    Try multiple variants safely.
    """
    variants = [("original", sql)]

    if "`" in sql:
        variants.append(("backticks_to_double", sql.replace("`", '"')))

    stripped = re.sub(r"[`\"]", "", sql)
    if stripped != sql:
        variants.append(("remove_quotes", stripped))

    return variants


_TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z0-9_`\"]+)",
    flags=re.IGNORECASE,
)


def _extract_table_refs(sql: str) -> List[str]:
    """Extract raw table references from SQL."""
    tables: List[str] = []
    for m in _TABLE_REF_RE.finditer(sql):
        raw = m.group(1)
        if raw:
            tables.append(raw.strip("`\""))
    return list(dict.fromkeys(tables))


def _duckdb_has_table(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    rows = con.execute("SHOW TABLES").fetchall()
    return table.lower() in {r[0].lower() for r in rows}


# ============================================================
# Connection management
# ============================================================
def get_connection() -> duckdb.DuckDBPyConnection:
    global _conn
    try:
        db_path = getattr(config, "DATABASE_PATH", None) or os.path.join(
            os.getcwd(), "data", "text_to_sql.duckdb"
        )
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        if _conn is None:
            logger.info("Opening DuckDB connection: %s", db_path)
            _conn = duckdb.connect(database=db_path, read_only=False)

        return _conn
    except Exception as e:
        logger.exception("DuckDB connection failed")
        raise CustomException(e, sys)


def close_connection() -> None:
    global _conn
    if _conn:
        try:
            _conn.close()
        finally:
            _conn = None


# ============================================================
# SQL Safety
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
)


def _is_safe_sql(sql: str) -> Tuple[bool, Optional[str]]:
    upper = sql.upper()
    for kw in _DISALLOWED:
        if kw in upper:
            return False, kw
    return True, None


# ============================================================
# CSV loading (🔥 FIXED)
# ============================================================
def load_csv_table(
    csv_path: str,
    canonical_name: str,
    *,
    force_reload: bool = False,
) -> str:
    """
    Load CSV into DuckDB using a sanitized physical name.

    Returns
    -------
    str
        Physical DuckDB table/view name
    """
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        con = get_connection()
        physical_name = sanitize_table_name(canonical_name)

        if _duckdb_has_table(con, physical_name) and not force_reload:
            logger.info("Table %s already exists", physical_name)
            return physical_name

        safe_path = csv_path.replace("'", "''")

        sql = f"""
        CREATE OR REPLACE VIEW {physical_name} AS
        SELECT *
        FROM read_csv_auto(
            '{safe_path}',
            all_varchar=true,
            null_padding=true,
            ignore_errors=true
        )
        """

        logger.info("Loading CSV %s → %s", csv_path, physical_name)
        con.execute(sql)

        return physical_name

    except Exception as e:
        logger.exception("CSV load failed")
        raise CustomException(e, sys)


def ensure_tables_loaded(
    table_map: Dict[str, str],
    *,
    force_reload: bool = False,
) -> Dict[str, str]:
    """
    canonical_name -> csv_path
    returns canonical_name -> physical_table_name
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
# Query execution
# ============================================================
def execute_query(
    sql: str,
    *,
    table_map: Optional[Dict[str, str]] = None,
    read_only: bool = True,
    as_dataframe: bool = False,
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    Execute SQL safely with automatic CSV loading.
    """
    start = time.time()

    try:
        if read_only:
            safe, bad = _is_safe_sql(sql)
            if not safe:
                raise PermissionError(f"Blocked keyword: {bad}")

        con = get_connection()

        # Auto-load referenced tables
        if table_map:
            referenced = _extract_table_refs(sql)
            to_load = {t: table_map[t] for t in referenced if t in table_map}
            ensure_tables_loaded(to_load)

        last_exc = None

        for label, variant in _normalize_sql_variants(sql):
            try:
                logger.debug("Executing SQL variant=%s", label)
                cur = con.execute(variant)
                rows = cur.fetchall()
                columns = [c[0] for c in cur.description] if cur.description else []

                meta = {
                    "rowcount": len(rows),
                    "runtime": time.time() - start,
                }

                if as_dataframe:
                    import pandas as pd
                    return pd.DataFrame(rows, columns=columns), columns, meta

                return rows, columns, meta

            except Exception as e:
                last_exc = e

        raise RuntimeError(f"All SQL variants failed: {last_exc}")

    except Exception as e:
        logger.exception("Query execution failed")
        raise CustomException(e, sys)


# ============================================================
# Table utilities
# ============================================================
def table_exists(canonical_name: str) -> bool:
    con = get_connection()
    return _duckdb_has_table(con, sanitize_table_name(canonical_name))


def list_tables() -> List[str]:
    con = get_connection()
    return [r[0] for r in con.execute("SHOW TABLES").fetchall()]
