import sys
import time
from typing import Optional, Dict, Any

from app.logger import get_logger
from app.exception import CustomException
from app import sql_executor

logger = get_logger("execute_node")


class ExecuteNode:
    """
    Node responsible for executing validated SQL against DuckDB (via sql_executor).

    Usage:
        node = ExecuteNode()
        result = node.run(sql, limit=100)

    Returns dict produced by sql_executor.execute_sql(), or raises CustomException on failure.
    """

    def __init__(self):
        try:
            # no heavy initialization required
            pass
        except Exception as e:
            logger.exception("Failed to initialize ExecuteNode")
            raise CustomException(e, sys)

    def run(self, sql: str, limit: Optional[int] = None, read_only: bool = True) -> Dict[str, Any]:
        """
        Execute the provided SQL using the project's sql_executor.

        Parameters:
            sql: The SQL string to execute (assumed validated).
            limit: Optional row limit to pass to executor (executor will wrap the SQL).
            read_only: Whether to run in read-only mode (default True).

        Returns:
            The execution result dict returned by sql_executor.execute_sql()

        Raises:
            CustomException on execution failure.
        """
        start = time.time()
        try:
            logger.info("ExecuteNode: executing SQL")
            result = sql_executor.execute_sql(sql, read_only=read_only, limit=limit)
            runtime = time.time() - start
            logger.info(f"ExecuteNode: execution finished in {runtime:.3f}s rows={result.get('meta', {}).get('rowcount')}")
            return result
        except CustomException:
            # propagate
            raise
        except Exception as e:
            logger.exception("ExecuteNode.run failed")
            raise CustomException(e, sys)
