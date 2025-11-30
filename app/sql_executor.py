# app/sql_executor.py
import sys
import time
from typing import List, Dict, Any, Tuple, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import database

logger = get_logger("sql_executor")

# ---------------------------
# SQL Safety (read-only mode)
# ---------------------------
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
    Validate SQL query for safety in read-only mode.
    Raises CustomException if disallowed keywords are found.
    """
    if not read_only:
        return
    upper_sql = sql.upper()
    for kw in _READONLY_DISALLOWED:
        if kw in upper_sql:
            raise CustomException(f"Disallowed SQL for read-only mode: {kw}", sys)


def _rows_to_dicts(rows: List[Tuple[Any, ...]], columns: List[str]) -> List[Dict[str, Any]]:
    """
    Convert list of tuple rows to list of dicts using column names.
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not columns:
            out.append({f"col{i}": v for i, v in enumerate(r)})
        else:
            out.append({col: v for col, v in zip(columns, r)})
    return out


def _strip_trailing_semicolon(sql: str) -> str:
    """Remove a single trailing semicolon so wrapping queries remain valid."""
    if not sql:
        return sql
    s = sql.rstrip()
    if s.endswith(";"):
        return s[:-1]
    return s


# ---------------------------
# Main SQL Executor
# ---------------------------
def execute_sql(
    sql: str,
    *,
    table_map: Optional[Dict[str, str]] = None,
    read_only: bool = True,
    limit: Optional[int] = None,
    as_dataframe: bool = False
) -> Dict[str, Any]:
    """
    Execute SQL query safely with optional pre-loading of CSVs into DuckDB.

    Parameters
    ----------
    sql : str
        SQL query to execute.
    table_map : Optional[Dict[str,str]]
        Optional mapping of canonical_table_name -> csv_path.
        If provided, each csv will be loaded into DuckDB as table canonical_table_name before execution.
    read_only : bool
        If True, prevent unsafe operations.
    limit : Optional[int]
        Maximum number of rows to return (applied as wrapping SELECT * FROM (sql) LIMIT n).
    as_dataframe : bool
        If True, attempt to return a pandas.DataFrame (if pandas available). Otherwise returns rows list.

    Returns
    -------
    Dict[str, Any]
        {
            "rows": list of dicts (or dataframe rows),
            "columns": list of column names,
            "meta": {"rowcount": int, "runtime": float}
        }

    Raises
    ------
    CustomException
        If execution fails or SQL is unsafe.
    """
    start = time.time()
    try:
        if not isinstance(sql, str):
            raise CustomException("SQL must be a string", sys)

        # Basic safety
        _validate_sql(sql, read_only)

        # If user provided table_map, ensure those CSVs are loaded into DuckDB
        if table_map:
            try:
                database.ensure_tables_loaded(table_map)
            except Exception as e:
                logger.exception("Failed to ensure tables loaded before execution")
                raise CustomException(e, sys)

        # Prepare SQL for execution (strip trailing semicolon then optionally wrap)
        sql_clean = _strip_trailing_semicolon(sql)
        exec_sql = sql_clean
        if limit is not None:
            exec_sql = f"SELECT * FROM ({sql_clean}) AS _texttosql_sub LIMIT {int(limit)}"

        # Execute via database module
        rows_or_df, columns, meta = database.execute_query(
            exec_sql,
            read_only=read_only,
            as_dataframe=as_dataframe
        )

        # Normalize meta
        meta = meta or {}
        runtime_from_db = meta.get("runtime", None)

        # Convert results to list-of-dicts for UI consumption (unless user explicitly requested DF)
        if as_dataframe:
            try:
                import pandas as pd  # local import
                if isinstance(rows_or_df, pd.DataFrame):
                    rows_list = [r.to_dict() for _, r in rows_or_df.iterrows()]
                else:
                    rows_list = _rows_to_dicts(rows_or_df, columns)
                result_meta = {
                    "rowcount": meta.get("rowcount", len(rows_list)),
                    "runtime": runtime_from_db if runtime_from_db is not None else (time.time() - start),
                }
                logger.info(f"Executed SQL in {time.time() - start:.3f}s (as_dataframe)")
                return {"rows": rows_list, "columns": list(columns or []), "meta": result_meta}
            except Exception:
                # fallback to list-of-dicts
                rows_list = _rows_to_dicts(rows_or_df, columns)
                result_meta = {
                    "rowcount": meta.get("rowcount", len(rows_list)),
                    "runtime": runtime_from_db if runtime_from_db is not None else (time.time() - start),
                }
                logger.info(f"Executed SQL in {time.time() - start:.3f}s (as_dataframe-fallback)")
                return {"rows": rows_list, "columns": list(columns or []), "meta": result_meta}

        # Normal path (list of tuples -> list of dicts)
        rows_list = _rows_to_dicts(rows_or_df, columns)
        meta_out = {
            "rowcount": meta.get("rowcount", len(rows_list)),
            "runtime": runtime_from_db if runtime_from_db is not None else (time.time() - start)
        }
        logger.info(f"Executed SQL; rows={meta_out['rowcount']} runtime={meta_out['runtime']:.3f}s")
        return {"rows": rows_list, "columns": list(columns or []), "meta": meta_out}

    except CustomException:
        raise
    except Exception as e:
        logger.exception("Failed to execute SQL in sql_executor")
        raise CustomException(e, sys)
