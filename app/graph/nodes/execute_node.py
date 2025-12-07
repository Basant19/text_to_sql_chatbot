# app/graph/nodes/execute_node.py
from __future__ import annotations
import sys
import logging
from typing import Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

logger = get_logger("execute_node")
LOG = logging.getLogger(__name__)


class ExecuteNode:
    """
    ExecuteNode:
      - Executes SQL using Tools.execute_sql or Tools._executor as fallback.
      - Enforces read_only by default and supports limit injection by caller.
      - Normalizes results into {'rows': [...], 'columns': [...], 'rowcount': N}
    """

    def __init__(self, tools: Optional[Tools] = None, default_limit: Optional[int] = None):
        try:
            self._tools = tools or Tools()
            self._default_limit = default_limit
        except Exception as e:
            logger.exception("Failed to initialize ExecuteNode")
            raise CustomException(e, sys)

    def run(self, sql: str, read_only: bool = True, limit: Optional[int] = None, as_dataframe: bool = False) -> Dict[str, Any]:
        try:
            if not sql:
                return {"rows": [], "columns": [], "rowcount": 0, "as_dataframe": None}

            # prefer explicit limit param
            effective_limit = limit if limit is not None else self._default_limit

            # If limit is requested, naive injection of LIMIT at end if not already present.
            # (Downstream SQLExecutor may do a safer parse.)
            if effective_limit is not None and "limit" not in sql.lower():
                if sql.strip().endswith(";"):
                    sql = sql.rstrip(";")
                sql = f"{sql} LIMIT {effective_limit}"

            try:
                if hasattr(self._tools, "execute_sql"):
                    return self._tools.execute_sql(sql, read_only=read_only, limit=effective_limit, as_dataframe=as_dataframe)
                # fallback to executor attr
                executor = getattr(self._tools, "_executor", None)
                if executor and hasattr(executor, "execute_sql"):
                    return executor.execute_sql(sql, read_only=read_only, limit=effective_limit, as_dataframe=as_dataframe)
                # last resort: if executor is callable
                if callable(executor):
                    return executor(sql, read_only=read_only, limit=effective_limit, as_dataframe=as_dataframe)
            except Exception:
                LOG.exception("Execution failed for SQL: %s", sql)
                raise

            raise CustomException("No executor available to run SQL", sys)
        except CustomException:
            raise
        except Exception as e:
            logger.exception("ExecuteNode.run failed")
            raise CustomException(e, sys)
