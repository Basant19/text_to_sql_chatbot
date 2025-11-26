# app/graph/nodes/execute_node.py

import sys
import time
from typing import Optional, Dict, Any

from app.logger import get_logger
from app.exception import CustomException
from app import sql_executor

logger = get_logger("execute_node")


class ExecuteNode:
    """
    Node responsible for executing validated SQL queries against DuckDB (via sql_executor).

    Features:
    - Supports optional row limits and read-only mode.
    - Returns structured results with metadata.
    - Logs execution time and row count.
    - Fully testable and modular.
    """

    def __init__(self):
        try:
            # No heavy initialization required
            logger.info("ExecuteNode initialized")
        except Exception as e:
            logger.exception("Failed to initialize ExecuteNode")
            raise CustomException(e, sys)

    def run(
        self,
        sql: str,
        limit: Optional[int] = None,
        read_only: bool = True,
        capture_runtime: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the provided SQL using the project's sql_executor.

        Parameters
        ----------
        sql : str
            Validated SQL query string to execute.
        limit : Optional[int]
            Row limit to enforce at execution time (executor will wrap SQL if needed).
        read_only : bool
            Whether to execute in read-only mode (default True).
        capture_runtime : bool
            Whether to include runtime info in logs (default True).

        Returns
        -------
        Dict[str, Any]
            {
                "data": <list of rows>,
                "meta": {
                    "rowcount": <number of rows returned>,
                    "runtime": <seconds, optional>,
                    ...
                }
            }

        Raises
        ------
        CustomException
            If execution fails.
        """
        start_time = time.time()
        try:
            logger.info("ExecuteNode: executing SQL%s",
                        f" with limit={limit}" if limit else "")

            # Execute the query via centralized executor
            result = sql_executor.execute_sql(sql, limit=limit, read_only=read_only)

            if capture_runtime:
                runtime = time.time() - start_time
                # Include runtime in metadata for downstream tracking
                if "meta" not in result:
                    result["meta"] = {}
                result["meta"]["runtime"] = round(runtime, 3)
                logger.info("ExecuteNode: execution finished in %.3fs, rows=%s",
                            runtime, result.get("meta", {}).get("rowcount"))

            return result

        except CustomException:
            # propagate CustomException directly
            raise
        except Exception as e:
            logger.exception("ExecuteNode.run failed")
            raise CustomException(e, sys)
