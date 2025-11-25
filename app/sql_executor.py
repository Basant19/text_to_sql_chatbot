# app/sql_executor.py

import sys
import time
from typing import List, Dict, Any, Tuple, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import database

logger = get_logger("sql_executor")

# ---------------------------
# SQL Safety (readonly mode)
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
    Raises CustomException if disallowed keywords found.
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


# ---------------------------
# Main Executor
# ---------------------------
def execute_sql(
    sql: str,
    read_only: bool = True,
    limit: Optional[int] = None,
    as_dataframe: bool = False
) -> Dict[str, Any]:
    """
    Execute SQL query safely with optional row limit and return structured results.

    Parameters
    ----------
    sql : str
        SQL query to execute.
    read_only : bool
        If True, prevent unsafe operations.
    limit : Optional[int]
        Maximum number of rows to return.
    as_dataframe : bool
        If True, return pandas.DataFrame if available.

    Returns
    -------
    Dict[str, Any]
        {
            "rows": list of dicts or DataFrame rows,
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
        # Safety check
        _validate_sql(sql, read_only)

        # Apply row limit if requested
        exec_sql = sql
        if limit is not None:
            exec_sql = f"SELECT * FROM ({sql}) AS _sub LIMIT {int(limit)}"

        # Execute via database module
        rows_or_df, columns, meta = database.execute_query(exec_sql, read_only=read_only, as_dataframe=as_dataframe)

        # Convert results to list-of-dicts
        if as_dataframe:
            try:
                import pandas as pd  # local import
                if isinstance(rows_or_df, pd.DataFrame):
                    rows_list = [row._asdict() if hasattr(row, "_asdict") else row.to_dict() for _, row in rows_or_df.iterrows()]
                else:
                    rows_list = _rows_to_dicts(rows_or_df, columns)
                result = {"rows": rows_list, "columns": list(columns), "meta": meta}
                logger.info(f"Executed SQL in {time.time() - start:.3f}s (as_dataframe)")
                return result
            except Exception:
                rows_list = _rows_to_dicts(rows_or_df, columns)
                result = {"rows": rows_list, "columns": list(columns), "meta": meta}
                logger.info(f"Executed SQL in {time.time() - start:.3f}s (as_dataframe-fallback)")
                return result

        # Normal path (list of tuples)
        rows_list = _rows_to_dicts(rows_or_df, columns)
        meta_out = {"rowcount": meta.get("rowcount", len(rows_list)), "runtime": meta.get("runtime", time.time() - start)}
        logger.info(f"Executed SQL; rows={meta_out['rowcount']} runtime={meta_out['runtime']:.3f}s")
        return {"rows": rows_list, "columns": list(columns), "meta": meta_out}

    except CustomException:
        raise
    except Exception as e:
        logger.exception("Failed to execute SQL in sql_executor")
        raise CustomException(e, sys)
