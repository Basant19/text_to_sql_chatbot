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
        DuckDB table name.
    force_reload : bool
        Replace table if exists.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        con = get_connection()

        if table_exists(table_name) and not force_reload:
            logger.info(f"Table {table_name} exists. Skipping load (force_reload=False).")
            return

        safe_path = path.replace("'", "''")
        sql = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{safe_path}')"
        logger.info(f"Loading CSV into table {table_name}: {path}")
        con.execute(sql)
        logger.info(f"CSV loaded into table {table_name}")
    except Exception as e:
        logger.exception("Failed to load CSV into DuckDB")
        raise CustomException(e, sys)


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
        logger.info(f"Executing SQL (read_only={read_only}): {sql}")
        res = con.execute(sql)

        # Column names
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
                logger.warning("pandas not available; returning list of rows instead")
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
    """Check if a table exists in DuckDB."""
    try:
        con = get_connection()
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        return table_name in tables
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
