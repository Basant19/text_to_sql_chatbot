# app/graph/nodes/format_node.py
import sys
import json
from typing import Any, Dict, List, Optional

from app.logger import get_logger
from app.exception import CustomException

logger = get_logger("format_node")


class FormatNode:
    """
    FormatNode: produce a user-friendly formatted response.

    Preferred run signature:
        run(sql: str, schemas: dict, retrieved: Optional[list],
            execution: Optional[dict], raw: Optional[Any]) -> Dict[str, Any]

    Returns a dict:
    {
      "output": <str|dict>,    # rendered text or structured object
      "sql": <sql>,
      "explain": <str> (optional),
      "rows": <list> (optional),
      "meta": {...} (optional)
    }
    """

    def __init__(self, pretty: bool = True):
        self.pretty = pretty
        logger.info("FormatNode initialized (pretty=%s)", self.pretty)

    def run(self, sql: str, schemas: Optional[Dict[str, Any]] = None,
            retrieved: Optional[List[Dict[str, Any]]] = None,
            execution: Optional[Dict[str, Any]] = None,
            raw: Optional[Any] = None) -> Dict[str, Any]:
        try:
            # Basic formatted payload
            payload: Dict[str, Any] = {
                "sql": sql,
            }

            # Add execution results if present
            if execution is not None:
                # Accept execution in many shapes: dict with 'rows' or list of tuples
                if isinstance(execution, dict):
                    payload["rows"] = execution.get("rows") or execution.get("data") or execution.get("results")
                    payload["meta"] = {k: v for k, v in execution.items() if k != "rows"}
                else:
                    payload["rows"] = execution

            # Simple explanation constructed from raw and retrieved docs
            explain_parts = []
            if retrieved:
                explain_parts.append(f"{len(retrieved)} retrieved document(s) used for RAG context.")
            if raw:
                # try to pretty print the raw LLM output (non-sensitive)
                try:
                    explain_parts.append("Raw LLM output present.")
                except Exception:
                    pass

            if explain_parts:
                payload["explain"] = " ".join(explain_parts)

            # Human-friendly output
            if self.pretty:
                # Try to build a short textual summary
                output_lines = []
                output_lines.append(f"SQL: {sql}")
                if payload.get("rows") is not None:
                    r_preview = payload["rows"][:3] if isinstance(payload["rows"], list) else str(payload["rows"])
                    output_lines.append(f"Preview rows: {json.dumps(r_preview, default=str) if not isinstance(r_preview, str) else r_preview}")
                if payload.get("explain"):
                    output_lines.append(f"Notes: {payload['explain']}")
                payload["output"] = "\n".join(output_lines)
            else:
                payload["output"] = payload

            logger.info("FormatNode: formatted output ready (sql_len=%s)", len(sql or ""))
            return payload

        except Exception as e:
            logger.exception("FormatNode.run failed")
            raise CustomException(e, sys)


# backward-compatibility adapter that can wrap old signature formatters
class FormatAdapter:
    """
    Wraps a format_node instance that may accept a different signature.
    Adapter will attempt to call the underlying node with a modern signature,
    then try fewer args (positional), then finally fall back to calling with only sql.

    Useful if you have older FormatNode implementations in other repos.
    """

    def __init__(self, fmt_node):
        self._node = fmt_node

    def run(self, sql, schemas=None, retrieved=None, execution=None, raw=None):
        # Try the node directly with the modern args
        try:
            return self._node.run(sql, schemas, retrieved, execution, raw)
        except TypeError:
            pass

        # Try common older signature: run(sql, schemas, retrieved)
        try:
            return self._node.run(sql, schemas, retrieved)
        except TypeError:
            pass

        # Try minimal signature: run(sql)
        try:
            return self._node.run(sql)
        except TypeError:
            pass

        # Last resort: try with kwargs mapped
        try:
            return self._node.run(sql=sql, schemas=schemas, retrieved=retrieved, execution=execution, raw=raw)
        except Exception as e:
            # re-raise so builder's error handler can catch it
            raise
