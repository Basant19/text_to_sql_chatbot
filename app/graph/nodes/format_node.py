import sys
import time
from typing import Dict, Any, List, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import utils

logger = get_logger("format_node")


class FormatNode:
    """
    Node to format execution results for UI consumption.

    Input: execution_result as returned by sql_executor.execute_sql:
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
            pass
        except Exception as e:
            logger.exception("Failed to initialize FormatNode")
            raise CustomException(e, sys)

    def run(self, execution_result: Optional[Dict[str, Any]], max_preview: int = 5) -> Dict[str, Any]:
        """
        Format an execution result dict.

        :param execution_result: dict returned from sql_executor.execute_sql
        :param max_preview: number of rows to include in preview_text and rows
        :return: formatted dict
        """
        start = time.time()
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

            # Prepare preview of sample rows using utils.preview_sample_rows
            # rows are expected to be list of dicts; if not, attempt to convert.
            preview_rows: List[Dict[str, Any]] = []
            for r in rows[:max_preview]:
                # if row is a tuple and columns exist, convert to dict
                if not isinstance(r, dict) and columns:
                    preview_rows.append({col: val for col, val in zip(columns, r)})
                elif isinstance(r, dict):
                    preview_rows.append(r)
                else:
                    # fallback: represent the row as {"value": str(r)}
                    preview_rows.append({"value": r})

            preview_text = utils.preview_sample_rows(preview_rows, max_preview)

            result = {
                "preview_text": preview_text,
                "columns": columns,
                "rowcount": rowcount,
                "runtime": runtime,
                "rows": preview_rows,
            }
            duration = time.time() - start
            logger.info(f"FormatNode: formatted result in {duration:.3f}s rows_preview={len(preview_rows)}")
            return result
        except CustomException:
            raise
        except Exception as e:
            logger.exception("FormatNode.run failed")
            raise CustomException(e, sys)
