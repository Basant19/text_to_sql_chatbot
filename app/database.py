# app/database.py
"""
Database Layer — DuckDB Execution Engine
=======================================

This module is the FINAL execution boundary for all SQL queries.

────────────────────────────────────────────────────────────
HARD CONTRACT (NON-NEGOTIABLE)
────────────────────────────────────────────────────────────
1. This layer NEVER sees canonical table names
2. This layer NEVER rewrites SQL
3. This layer NEVER guesses identifiers
4. This layer NEVER inspects user intent

All SQL reaching this layer MUST:
- Be read-only (SELECT / WITH / EXPLAIN)
- Reference ONLY runtime_table names
- Have been validated and rewritten upstream
- Target tables that are already loaded

If SQL fails here → it is a SYSTEM ERROR (not user error).

────────────────────────────────────────────────────────────
RESPONSIBILITIES
────────────────────────────────────────────────────────────
✔ Manage DuckDB connection lifecycle
✔ Load CSVs into DuckDB as runtime views
✔ Enforce LAST-LINE read-only safety (statement-level)
✔ Execute SQL and return structured results

────────────────────────────────────────────────────────────
ANTI-GOALS
────────────────────────────────────────────────────────────
✘ No SQL generation
✘ No schema inference
✘ No SQL rewriting
✘ No retry logic
✘ No UI / formatting logic

────────────────────────────────────────────────────────────
SECURITY MODEL
────────────────────────────────────────────────────────────
This layer blocks ONLY true mutating SQL statements.
Read-only SQL functions (REPLACE, CAST, SUBSTR, etc.)
are explicitly allowed.
"""

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

# ============================================================================
# Process-wide DuckDB connection (INTENTIONAL SINGLETON)
# ============================================================================
_CONN: Optional[duckdb.DuckDBPyConnection] = None


# ============================================================================
# Connection management
# ============================================================================
def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Get or create the cached DuckDB connection.

    Guarantees
    ----------
    - Exactly one DuckDB connection per process
    - Connection reused across queries
    - Thread-safe at DuckDB engine level
    """
    global _CONN

    try:
        db_path = getattr(config, "DATABASE_PATH", None)
        if not db_path:
            db_path = os.path.join(os.getcwd(), "data", "text_to_sql.duckdb")

        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        if _CONN is None:
            logger.info("Opening DuckDB connection | path=%s", db_path)
            _CONN = duckdb.connect(database=db_path, read_only=False)

        return _CONN

    except Exception as e:
        logger.exception("Failed to initialize DuckDB connection")
        raise CustomException(e, sys)


def close_connection() -> None:
    """
    Close the cached DuckDB connection (best-effort).

    Safe to call multiple times.
    """
    global _CONN
    if _CONN is not None:
        try:
            logger.info("Closing DuckDB connection")
            _CONN.close()
        finally:
            _CONN = None


# ============================================================================
# SQL Safety — LAST LINE OF DEFENSE
# ============================================================================
"""
IMPORTANT
---------
This safety check is INTENTIONALLY MINIMAL and STATEMENT-BASED.

- Semantic validation belongs in ValidateNode
- This layer blocks ONLY true mutating SQL statements
- SQL functions (REPLACE, CAST, etc.) are explicitly allowed
"""

# Explicit mutating SQL statements (case-insensitive)
_MUTATING_PATTERNS = [
    r"^\s*DELETE\s+FROM\b",
    r"^\s*INSERT\s+INTO\b",
    r"^\s*UPDATE\s+\w+\b",
    r"^\s*DROP\s+TABLE\b",
    r"^\s*ALTER\s+TABLE\b",
    r"^\s*TRUNCATE\s+TABLE\b",
    r"^\s*CREATE\s+TABLE\b",
    r"^\s*CREATE\s+VIEW\b",
    r"^\s*REPLACE\s+INTO\b",
]


def _validate_read_only_sql(sql: str) -> None:
    """
    Enforce read-only SQL at the FINAL execution boundary.

    This function blocks ONLY true mutating SQL statements.
    It DOES NOT block read-only SQL functions.

    Parameters
    ----------
    sql : str

    Raises
    ------
    PermissionError
        If SQL attempts to mutate data
    """
    if not isinstance(sql, str):
        raise PermissionError("SQL must be a string")

    # Remove SQL comments
    scrubbed = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    scrubbed = re.sub(r"/\*.*?\*/", "", scrubbed, flags=re.DOTALL)

    for pattern in _MUTATING_PATTERNS:
        if re.search(pattern, scrubbed, flags=re.IGNORECASE):
            logger.error(
                "Blocked mutating SQL detected | pattern=%s | sql=%s",
                pattern,
                sql.strip()[:200],
            )
            raise PermissionError(
                "Mutating SQL statements are not allowed"
            )


# ============================================================================
# DuckDB helpers
# ============================================================================
def _duckdb_has_table(
    con: duckdb.DuckDBPyConnection,
    runtime_table: str,
) -> bool:
    """
    Check whether a runtime table/view exists in DuckDB.
    """
    rows = con.execute("SHOW TABLES").fetchall()
    return runtime_table.lower() in {r[0].lower() for r in rows}


# ============================================================================
# CSV loading (IDEMPOTENT, RUNTIME TABLES ONLY)
# ============================================================================
def load_csv_table(
    *,
    csv_path: str,
    runtime_table: str,
    force_reload: bool = False,
) -> str:
    """
    Load a CSV file into DuckDB as a VIEW.

    Guarantees
    ----------
    - Uses runtime_table name verbatim
    - Never infers schema
    - Idempotent unless force_reload=True
    """
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        if not runtime_table:
            raise ValueError("runtime_table must be provided")

        con = get_connection()

        if _duckdb_has_table(con, runtime_table) and not force_reload:
            logger.debug(
                "DuckDB runtime table already exists | table=%s",
                runtime_table,
            )
            return runtime_table

        safe_path = csv_path.replace("'", "''")

        sql = f"""
        CREATE OR REPLACE VIEW {runtime_table} AS
        SELECT *
        FROM read_csv_auto(
            '{safe_path}',
            all_varchar = true,
            null_padding = true,
            ignore_errors = true
        )
        """

        logger.info(
            "Loading CSV into DuckDB | csv=%s | runtime_table=%s",
            csv_path,
            runtime_table,
        )

        con.execute(sql)
        return runtime_table

    except Exception as e:
        logger.exception("Failed to load CSV into DuckDB")
        raise CustomException(e, sys)


def ensure_tables_loaded(
    runtime_map: Dict[str, str],
    *,
    force_reload: bool = False,
) -> None:
    """
    Ensure all runtime tables exist in DuckDB.
    """
    for runtime_table, path in runtime_map.items():
        load_csv_table(
            csv_path=path,
            runtime_table=runtime_table,
            force_reload=force_reload,
        )


# ============================================================================
# Query execution (FINAL EXECUTION BOUNDARY)
# ============================================================================
def execute_query(
    sql: str,
    *,
    read_only: bool = True,
    as_dataframe: bool = False,
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    Execute SQL against DuckDB.

    ASSUMPTIONS (ENFORCED UPSTREAM)
    -------------------------------
    - SQL references ONLY runtime tables
    - SQL passed ValidateNode
    - Required tables are already loaded
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
            "runtime_sec": round(time.time() - start, 4),
        }

        logger.info(
            "DuckDB query executed | rows=%d | time=%.4fs",
            meta["rowcount"],
            meta["runtime_sec"],
        )

        if as_dataframe:
            import pandas as pd
            return pd.DataFrame(rows, columns=columns), columns, meta

        return rows, columns, meta

    except Exception as e:
        logger.exception("DuckDB query execution failed")
        raise CustomException(e, sys)


# ============================================================================
# Debug / inspection helpers (SAFE)
# ============================================================================
def table_exists(runtime_table: str) -> bool:
    """
    Check whether a runtime table exists.
    """
    con = get_connection()
    return _duckdb_has_table(con, runtime_table)


def list_tables() -> List[str]:
    """
    List all runtime tables in DuckDB.
    """
    con = get_connection()
    return [r[0] for r in con.execute("SHOW TABLES").fetchall()]
