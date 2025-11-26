# app/graph/nodes/format_node.py

import sys
import time
from typing import Dict, Any, List, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import utils

logger = get_logger("format_node")


class FormatNode:
    """
    Node to format SQL execution results for UI or downstream consumption.

    Input:
        execution_result as returned by sql_executor.execute_sql:
        {
            "rows": [ {col: val, ...}, ... ],
            "columns": [...],
            "meta": {"rowcount": int, "runtime": float}
        }

    Output:
        {
            "preview_text": "<human readable preview>",
            "columns": [...],
            "rowcount": int,
            "runtime": float,
            "rows": [ ... ]  # truncated to max_preview
        }
    """

    def __init__(self):
        try:
            logger.info("FormatNode initialized")
        except Exception as e:
            logger.exception("Failed to initialize FormatNode")
            raise CustomException(e, sys)

    def run(
        self,
        execution_result: Optional[Dict[str, Any]],
        max_preview: int = 5
    ) -> Dict[str, Any]:
        """
        Format an execution result dict for display.

        Parameters
        ----------
        execution_result : Optional[Dict[str, Any]]
            Output from sql_executor.execute_sql
        max_preview : int
            Maximum number of rows to include in preview_text and 'rows'

        Returns
        -------
        Dict[str, Any]
            Formatted result including preview text, columns, rowcount, runtime, and preview rows.
        """
        start_time = time.time()
        try:
            if not execution_result:
                logger.info("FormatNode: empty execution_result provided")
                return {
                    "preview_text": "",
                    "columns": [],
                    "rowcount": 0,
                    "runtime": 0.0,
                    "rows": [],
                }

            rows = execution_result.get("rows", []) or []
            columns = execution_result.get("columns", []) or []
            meta = execution_result.get("meta", {}) or {}
            rowcount = int(meta.get("rowcount", len(rows)))
            runtime = float(meta.get("runtime", 0.0))

            # Prepare preview rows
            preview_rows: List[Dict[str, Any]] = []
            for r in rows[:max_preview]:
                if isinstance(r, dict):
                    preview_rows.append(r)
                elif columns:
                    # Convert tuple/list rows to dict using columns
                    preview_rows.append({col: val for col, val in zip(columns, r)})
                else:
                    # Fallback: represent row as single value
                    preview_rows.append({"value": r})

            # Generate human-readable preview text
            preview_text = utils.preview_sample_rows(preview_rows, max_preview)

            formatted_result = {
                "preview_text": preview_text,
                "columns": columns,
                "rowcount": rowcount,
                "runtime": runtime,
                "rows": preview_rows,
            }

            duration = time.time() - start_time
            logger.info(
                "FormatNode: formatted result in %.3fs, rows_preview=%d",
                duration,
                len(preview_rows),
            )

            return formatted_result

        except CustomException:
            raise
        except Exception as e:
            logger.exception("FormatNode.run failed")
            raise CustomException(e, sys)
