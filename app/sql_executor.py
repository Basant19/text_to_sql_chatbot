import sys
import time
from typing import List, Dict, Any, Tuple, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import config
from app import database

logger = get_logger("sql_executor")


# Simple SQL safety layer (basic keyword heuristic). Database also checks this,
# but we provide a clearer wrapper and standard output format.
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
    if not read_only:
        return
    up = sql.upper()
    for kw in _READONLY_DISALLOWED:
        if kw in up:
            raise CustomException(f"Disallowed SQL for read-only mode: {kw}", sys)


def _rows_to_dicts(rows: List[Tuple[Any, ...]], columns: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        # If columns missing (e.g., no results), create generic names
        if not columns:
            out.append({f"col{i}": v for i, v in enumerate(r)})
        else:
            out.append({col: v for col, v in zip(columns, r)})
    return out


def execute_sql(
    sql: str, read_only: bool = True, limit: Optional[int] = None, as_dataframe: bool = False
) -> Dict[str, Any]:
    """
    Execute SQL query safely and return a result dict:
    {
      "rows": [ {col: value, ...}, ... ] or pandas.DataFrame if as_dataframe=True & pandas available,
      "columns": [...],
      "meta": {"rowcount": int, "runtime": float}
    }

    Raises CustomException on error.
    """
    start = time.time()
    try:
        _validate_sql(sql, read_only)
        if limit is not None:
            # Basic approach: wrap query in SELECT * FROM (<sql>) LIMIT N
            wrapped = f"SELECT * FROM ({sql}) AS _sub LIMIT {int(limit)}"
            exec_sql = wrapped
        else:
            exec_sql = sql

        rows_or_df, columns, meta = database.execute_query(exec_sql, read_only=read_only, as_dataframe=as_dataframe)

        # If DataFrame requested and returned, try to serialize rows as list of dicts as well.
        if as_dataframe:
            try:
                import pandas as pd  # type: ignore
                if isinstance(rows_or_df, pd.DataFrame):
                    rows_list = [row._asdict() if hasattr(row, "_asdict") else row.to_dict() for _, row in rows_or_df.iterrows()]
                else:
                    # fallback
                    rows_list = _rows_to_dicts(rows_or_df, columns)
                result = {"rows": rows_list, "columns": list(columns), "meta": meta}
                runtime = time.time() - start
                logger.info(f"Executed SQL in {runtime:.3f}s (as_dataframe)")
                return result
            except Exception:
                # fallback to list of tuples processing
                rows = rows_or_df
                rows_list = _rows_to_dicts(rows, columns)
                result = {"rows": rows_list, "columns": list(columns), "meta": meta}
                runtime = time.time() - start
                logger.info(f"Executed SQL in {runtime:.3f}s (as_dataframe-fallback)")
                return result

        # Normal path (list of tuples)
        rows = rows_or_df
        rows_list = _rows_to_dicts(rows, columns)
        meta_out = {"rowcount": meta.get("rowcount", len(rows_list)), "runtime": meta.get("runtime", time.time() - start)}
        logger.info(f"Executed SQL; rows={meta_out['rowcount']} runtime={meta_out['runtime']:.3f}s")
        return {"rows": rows_list, "columns": list(columns), "meta": meta_out}
    except CustomException:
        # re-raise unchanged
        raise
    except Exception as e:
        logger.exception("Failed to execute SQL in sql_executor")
        raise CustomException(e, sys)
