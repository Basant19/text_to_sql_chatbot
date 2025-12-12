#app\database.py
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
    if not isinstance(name, str) or not name:
        raise ValueError("Invalid table name")
    # replace non-alnum/_ with underscore
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    sanitized = sanitized.lower()
    if sanitized[0].isdigit():
        sanitized = f"t_{sanitized}"
    return sanitized


def _normalize_sql_quotes_for_duckdb(sql: str) -> List[Tuple[str, str]]:
    """
    Produce candidate normalized SQLs to try against DuckDB.
    Returns a list of tuples (label, sql_variant) in the order they should be tried.
    - 'original': original SQL
    - 'backticks_to_double': replace `...` with "..."
    - 'remove_quotes': remove both ` and " entirely
    """
    variants = []
    try:
        variants.append(("original", sql))
        if "`" in sql:
            v1 = sql.replace("`", '"')
            variants.append(("backticks_to_double", v1))
        # always include a no-quotes variant as a last resort
        v2 = re.sub(r'[`\"]', "", sql)
        if v2 != sql:
            variants.append(("remove_quotes", v2))
    except Exception:
        variants = [("original", sql)]
    return variants


# regex to extract simple table names after FROM/JOIN/INTO
_TABLE_NAME_RE = re.compile(r"\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+([A-Za-z0-9_`\"]+)", flags=re.IGNORECASE)


def _extract_tables_from_sql(sql: str) -> List[str]:
    """Extract candidate table identifiers from simple SQL. Returns raw identifiers (may include quotes/backticks)."""
    if not sql:
        return []
    found = []
    for m in _TABLE_NAME_RE.finditer(sql):
        raw = m.group(1)
        if not raw:
            continue
        # strip quotes/backticks
        raw_clean = raw.strip().strip('"').strip("`")
        if raw_clean:
            found.append(raw_clean)
    # preserve order, uniquify
    return list(dict.fromkeys(found))


def _duckdb_has_table(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    try:
        rows = con.execute("SHOW TABLES").fetchall()
        names = {r[0].lower() for r in rows}
        return table_name.lower() in names
    except Exception:
        try:
            rows = con.execute("SELECT table_name FROM information_schema.tables").fetchall()
            names = {r[0].lower() for r in rows}
            return table_name.lower() in names
        except Exception:
            return False


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
            logger.info("Opening DuckDB connection at %s", db_path)
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
                logger.warning("Error closing DuckDB connection: %s", inner)
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
    # Keep CREATE blocked for user SQL; we will perform controlled CREATEs when auto-loading CSVs.
    "CREATE",
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
        if _duckdb_has_table(con, sanitized) and not force_reload:
            logger.info("Table %s exists. Skipping load (force_reload=False).", sanitized)
            return

        # Use read_csv_auto which detects csv format; escape single quotes in path
        safe_path = path.replace("'", "''")
        # Create or replace a view to avoid permanently duplicating data in DB file.
        sql = f"CREATE OR REPLACE VIEW {sanitized} AS SELECT * FROM read_csv_auto('{safe_path}')"
        logger.info("Loading CSV into view %s: %s", sanitized, path)
        con.execute(sql)
        logger.info("CSV loaded into view %s", sanitized)
    except Exception as e:
        logger.exception("Failed to load CSV into DuckDB for path=%s table=%s", path, table_name)
        raise CustomException(e, sys)


def ensure_tables_loaded(table_map: Dict[str, str], force_reload: bool = False) -> None:
    """
    Ensure all tables in table_map (canonical_table_name -> csv_path) are loaded into DuckDB.
    """
    try:
        for canonical, path in table_map.items():
            sanitized = _sanitize_table_name(canonical)
            load_csv_table(path, sanitized, force_reload=force_reload)
    except Exception as e:
        logger.exception("Failed ensuring tables loaded")
        raise


# ---------------------------
# Query Execution
# ---------------------------
def execute_query(
    sql: str,
    read_only: Union[bool, Dict[str, Any]] = True,
    as_dataframe: bool = False,
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    Execute a SQL query safely.

    Parameters
    ----------
    sql : str
        SQL to run.
    read_only : bool | dict
        If a dict mapping canonical_table_name->metadata is provided, the function will
        attempt to auto-load referenced CSVs before executing the SQL.
    as_dataframe : bool
        Return a pandas DataFrame when True (if pandas available).

    Returns
    -------
    Tuple[result, columns, metadata]
    """
    start = time.time()
    try:
        # Basic safety check (read-only mode). If read_only is a mapping, still treat SQL as read-only.
        safe, bad_kw = _is_safe_sql(sql, True if isinstance(read_only, dict) or read_only else True)
        if not safe:
            msg = f"Disallowed SQL keyword in read-only mode: {bad_kw}"
            logger.warning(msg)
            raise PermissionError(msg)

        con = get_connection()

        # If read_only provides schema metadata mapping, attempt to auto-load referenced CSVs
        if isinstance(read_only, dict):
            try:
                referenced = _extract_tables_from_sql(sql)
                # Build a map canonical->path for referenced tables only
                to_load: Dict[str, str] = {}
                for tbl in referenced:
                    # try exact key, then sanitized key variants
                    if tbl in read_only:
                        meta = read_only.get(tbl) or {}
                        path = meta.get("path") or meta.get("csv_path") or meta.get("file") or meta.get("source")
                        if path:
                            to_load[tbl] = os.path.abspath(path)
                            continue
                    # try sanitized variant
                    san = _sanitize_table_name(tbl)
                    if san in read_only:
                        meta = read_only.get(san) or {}
                        path = meta.get("path") or meta.get("csv_path") or meta.get("file") or meta.get("source")
                        if path:
                            to_load[san] = os.path.abspath(path)
                            continue
                    # try case-insensitive match in keys
                    for k in list(read_only.keys()):
                        if k.lower() == tbl.lower() or _sanitize_table_name(k) == san:
                            meta = read_only.get(k) or {}
                            path = meta.get("path") or meta.get("csv_path") or meta.get("file") or meta.get("source")
                            if path:
                                to_load[k] = os.path.abspath(path)
                                break
                if to_load:
                    logger.info("Auto-loading CSVs for referenced tables: %s", list(to_load.keys()))
                    ensure_tables_loaded(to_load, force_reload=False)
            except Exception:
                logger.exception("Auto-registering referenced tables failed")

        # Try variants: original -> backticks->double -> remove quotes
        variants = _normalize_sql_quotes_for_duckdb(sql)
        last_exc = None
        for label, candidate in variants:
            try:
                logger.debug("Attempting SQL execute variant=%s", label)
                logger.info("Executing SQL (read_only=%s): %s", isinstance(read_only, dict) or read_only, candidate)
                res = con.execute(candidate)

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
                        logger.info("Query returned %d rows in %.3fs (variant=%s)", len(rows), runtime, label)
                        return df, columns, meta
                    except Exception:
                        logger.warning("pandas not available or conversion failed; returning list of rows instead")
                        return rows, columns, meta

                logger.info("Query returned %d rows in %.3fs (variant=%s)", len(rows), runtime, label)
                return rows, columns, meta

            except Exception as e:
                # record and try next variant
                logger.debug("Variant %s failed: %s", label, str(e))
                last_exc = e
                continue

        # if we reach here all variants failed
        logger.error("All SQL execute variants failed. original_sql=%s", sql)
        raise CustomException(
            RuntimeError(f"Failed to execute SQL after normalization attempts. last_error={last_exc}"),
            sys,
        )

    except CustomException:
        raise
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
