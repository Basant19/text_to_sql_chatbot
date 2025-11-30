# app/database.py
import os
import sys
import time
from typing import List, Tuple, Any, Dict, Optional

import duckdb

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("database")

# ---------------------------
# Module-level connection cache
# ---------------------------
_conn: Optional[duckdb.DuckDBPyConnection] = None


# ---------------------------
# Helpers
# ---------------------------
def _sanitize_table_name(name: str) -> str:
    """
    Ensure table name is a safe DuckDB identifier:
    - only letters, digits and underscores
    - cannot start with digit (prefix with t_ if it does)
    - lowercased for consistency
    """
    import re

    if not isinstance(name, str) or not name:
        raise ValueError("Invalid table name")
    # replace non-alnum/_ with underscore
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    sanitized = sanitized.lower()
    if sanitized[0].isdigit():
        sanitized = f"t_{sanitized}"
    return sanitized


# ---------------------------
# Connection Management
# ---------------------------
def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Return a singleton DuckDB connection, file-backed at config.DATABASE_PATH.

    Returns
    -------
    duckdb.DuckDBPyConnection
    """
    global _conn
    try:
        db_path = getattr(config, "DATABASE_PATH", None) or os.path.join(os.getcwd(), "data", "text_to_sql.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if _conn is None:
            logger.info(f"Opening DuckDB connection at {db_path}")
            # use read_only=False to allow CREATE OR REPLACE TABLE for CSV loads
            _conn = duckdb.connect(database=db_path, read_only=False)
        return _conn
    except Exception as e:
        logger.exception("Failed to get DuckDB connection")
        raise CustomException(e, sys)


def close_connection() -> None:
    """
    Close the module-level DuckDB connection if it exists.
    Useful for cleanup in tests or shutdown.
    """
    global _conn
    try:
        if _conn:
            try:
                _conn.close()
                logger.info("DuckDB connection closed")
            except Exception as inner:
                logger.warning(f"Error closing DuckDB connection: {inner}")
            finally:
                _conn = None
    except Exception as e:
        logger.exception("Failed to close DuckDB connection")


# ---------------------------
# SQL Safety
# ---------------------------
_DISALLOWED_KEYWORDS = (
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
    "CREATE",  # prevent schema changes in read-only mode
)


def _is_safe_sql(sql: str, read_only: bool) -> Tuple[bool, Optional[str]]:
    """
    Heuristic check for unsafe SQL queries.

    Returns
    -------
    Tuple[bool, Optional[str]]: (is_safe, offending_keyword_or_none)
    """
    if not read_only:
        return True, None
    upper_sql = sql.upper()
    for kw in _DISALLOWED_KEYWORDS:
        if kw in upper_sql:
            return False, kw
    return True, None


# ---------------------------
# CSV Loading
# ---------------------------
def load_csv_table(path: str, table_name: str, force_reload: bool = False) -> None:
    """
    Load a CSV into DuckDB as a table.

    Parameters
    ----------
    path : str
        Path to CSV file.
    table_name : str
        DuckDB table name (canonical). This will be sanitized.
    force_reload : bool
        Replace table if exists.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        con = get_connection()
        sanitized = _sanitize_table_name(table_name)

        # skip load if exists (unless forced)
        if table_exists(sanitized) and not force_reload:
            logger.info(f"Table {sanitized} exists. Skipping load (force_reload=False).")
            return

        # Use read_csv_auto which detects csv format; escape single quotes in path
        safe_path = path.replace("'", "''")
        sql = f"CREATE OR REPLACE TABLE {sanitized} AS SELECT * FROM read_csv_auto('{safe_path}')"
        logger.info(f"Loading CSV into table {sanitized}: {path}")
        con.execute(sql)
        logger.info(f"CSV loaded into table {sanitized}")
    except Exception as e:
        logger.exception("Failed to load CSV into DuckDB")
        raise CustomException(e, sys)


def ensure_tables_loaded(table_map: Dict[str, str], force_reload: bool = False) -> None:
    """
    Ensure all tables in table_map (canonical_table_name -> csv_path) are loaded into DuckDB.
    """
    try:
        for canonical, path in table_map.items():
            sanitized = _sanitize_table_name(canonical)
            # call loader which will skip if present unless force_reload=True
            load_csv_table(path, sanitized, force_reload=force_reload)
    except Exception as e:
        logger.exception("Failed ensuring tables loaded")
        raise


# ---------------------------
# Query Execution
# ---------------------------
def execute_query(
    sql: str,
    read_only: bool = True,
    as_dataframe: bool = False
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    Execute a SQL query safely.

    Returns
    -------
    Tuple[result, columns, metadata]
    """
    start = time.time()
    try:
        safe, bad_kw = _is_safe_sql(sql, read_only)
        if not safe:
            msg = f"Disallowed SQL keyword in read-only mode: {bad_kw}"
            logger.warning(msg)
            raise PermissionError(msg)

        con = get_connection()
        logger.info("Executing SQL (read_only=%s): %s", read_only, sql)
        res = con.execute(sql)

        # Column names (duckdb cursor description)
        try:
            columns = [c[0] for c in res.description] if res.description else []
        except Exception:
            columns = []

        rows = res.fetchall()
        runtime = time.time() - start
        meta = {"rowcount": len(rows), "runtime": runtime}

        if as_dataframe:
            try:
                import pandas as pd
                df = pd.DataFrame(rows, columns=columns)
                logger.info(f"Query returned {len(rows)} rows in {runtime:.3f}s")
                return df, columns, meta
            except Exception:
                logger.warning("pandas not available or conversion failed; returning list of rows instead")
                return rows, columns, meta

        logger.info(f"Query returned {len(rows)} rows in {runtime:.3f}s")
        return rows, columns, meta
    except Exception as e:
        logger.exception("Failed to execute query")
        raise CustomException(e, sys)


# ---------------------------
# Table Utilities
# ---------------------------
def table_exists(table_name: str) -> bool:
    """Check if a table exists in DuckDB (sanitized name)."""
    try:
        con = get_connection()
        sanitized = _sanitize_table_name(table_name)
        # DuckDB's SHOW TABLES returns list of table names
        tables = {r[0].lower() for r in con.execute("SHOW TABLES").fetchall()}
        return sanitized.lower() in tables
    except Exception as e:
        logger.exception("Failed to check table existence")
        raise CustomException(e, sys)


def list_tables() -> List[str]:
    """List all tables in DuckDB."""
    try:
        con = get_connection()
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        return tables
    except Exception as e:
        logger.exception("Failed to list tables")
        raise CustomException(e, sys)
