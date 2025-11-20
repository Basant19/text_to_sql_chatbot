import os
import sys
import time
from typing import List, Tuple, Any, Dict, Optional

import duckdb

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("database")

# Module-level connection cache
_conn: Optional[duckdb.DuckDBPyConnection] = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Return a singleton DuckDB connection, file-backed at config.DATABASE_PATH.
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
    Useful for cleanup in tests or when shutting down the app.
    """
    global _conn
    try:
        if _conn is not None:
            try:
                _conn.close()
                logger.info("DuckDB connection closed")
            except Exception as inner:
                logger.warning(f"Error while closing DuckDB connection: {inner}")
            finally:
                _conn = None
    except Exception as e:
        logger.exception("Failed to close DuckDB connection")


# Basic safety: keywords we disallow when read_only=True
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
    "CREATE",  # disallow CREATE by default to avoid schema changes
)


def _is_safe_sql(sql: str, read_only: bool) -> Tuple[bool, Optional[str]]:
    """
    Very small heuristic to detect unsafe SQL.
    Returns (is_safe, offending_keyword_or_none)
    """
    if not read_only:
        return True, None
    upper = sql.upper()
    for kw in _DISALLOWED_KEYWORDS:
        if kw in upper:
            return False, kw
    return True, None


def load_csv_table(path: str, table_name: str, force_reload: bool = False) -> None:
    """
    Load (or reload) a CSV file into DuckDB as `table_name`.
    Uses read_csv_auto for schema inference.
    If force_reload is True, existing table will be replaced.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        con = get_connection()

        # If table exists and no reload requested, skip
        if table_exists(table_name) and not force_reload:
            logger.info(f"Table {table_name} already exists and force_reload=False. Skipping load.")
            return

        # Escape single quotes in path for SQL string literal
        safe_path = path.replace("'", "''")

        # Create or replace table
        # DuckDB supports: CREATE OR REPLACE TABLE ... AS SELECT * FROM read_csv_auto('path')
        sql = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{safe_path}')"
        logger.info(f"Loading CSV into table {table_name}: {path}")
        con.execute(sql)
        logger.info(f"Loaded CSV into table {table_name}")
    except Exception as e:
        logger.exception("Failed to load CSV into DuckDB")
        raise CustomException(e, sys)


def execute_query(sql: str, read_only: bool = True, as_dataframe: bool = False) -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    Execute a SQL query against DuckDB.
    Returns:
      - result: either list of tuples (rows) or pandas.DataFrame if as_dataframe=True and pandas available
      - columns: list of column names
      - meta: dict with metadata (rowcount, runtime, error if any)
    Raises CustomException for errors.
    """
    start = time.time()
    try:
        safe, bad = _is_safe_sql(sql, read_only)
        if not safe:
            msg = f"Disallowed SQL keyword found in read-only mode: {bad}"
            logger.warning(msg)
            raise PermissionError(msg)

        con = get_connection()

        logger.info(f"Executing SQL (read_only={read_only}): {sql}")
        # Use DuckDB's execute and fetch
        res = con.execute(sql)
        try:
            # Try to fetch column names and rows
            columns = [c[0] for c in res.description] if res.description else []
        except Exception:
            columns = []

        # Fetch all rows
        rows = res.fetchall()
        runtime = time.time() - start

        meta = {"rowcount": len(rows), "runtime": runtime}

        if as_dataframe:
            try:
                import pandas as pd  # local import
                df = pd.DataFrame(rows, columns=columns)
                logger.info(f"Query returned {len(rows)} rows in {runtime:.3f}s")
                return df, columns, meta
            except Exception:
                # fallback to rows if pandas not available
                logger.warning("pandas not available or failed to create DataFrame; returning list of rows")
                return rows, columns, meta

        logger.info(f"Query returned {len(rows)} rows in {runtime:.3f}s")
        return rows, columns, meta
    except Exception as e:
        logger.exception("Failed to execute query")
        # If it's our PermissionError, wrap it for consistency
        if isinstance(e, PermissionError):
            raise CustomException(e, sys)
        raise CustomException(e, sys)


def table_exists(table_name: str) -> bool:
    """
    Check if table exists in the current DuckDB database (main schema).
    """
    try:
        con = get_connection()
        # SHOW TABLES returns tuples of table names
        res = con.execute("SHOW TABLES").fetchall()
        tables = {r[0] for r in res}
        return table_name in tables
    except Exception as e:
        logger.exception("Failed to check table existence")
        raise CustomException(e, sys)


def list_tables() -> List[str]:
    """
    List tables in the current DuckDB database.
    """
    try:
        con = get_connection()
        rows = con.execute("SHOW TABLES").fetchall()
        tables = [r[0] for r in rows]
        return tables
    except Exception as e:
        logger.exception("Failed to list tables")
        raise CustomException(e, sys)
