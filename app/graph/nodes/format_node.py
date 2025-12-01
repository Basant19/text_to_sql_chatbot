# app/graph/nodes/format_node.py
import sys
import json
from typing import Any, Dict, List, Optional

from app.logger import get_logger
from app.exception import CustomException

logger = get_logger("format_node")

__all__ = ["FormatNode", "FormatAdapter"]


class FormatNode:
    """
    FormatNode: produce a user-friendly formatted response.

    Preferred run signature:
        run(sql: str, schemas: dict, retrieved: Optional[list],
            execution: Optional[dict], raw: Optional[Any]) -> Dict[str, Any]
    """

    def __init__(self, pretty: bool = True):
        self.pretty = pretty
        logger.info("FormatNode initialized (pretty=%s)", self.pretty)

    def run(self, sql: str, schemas: Optional[Dict[str, Any]] = None,
            retrieved: Optional[List[Dict[str, Any]]] = None,
            execution: Optional[Dict[str, Any]] = None,
            raw: Optional[Any] = None) -> Dict[str, Any]:
        try:
            payload: Dict[str, Any] = {"sql": sql}

            # normalize execution shapes
            if execution is not None:
                if isinstance(execution, dict):
                    payload["rows"] = execution.get("rows") or execution.get("data") or execution.get("results")
                    # copy everything else to meta (shallow)
                    meta = {k: v for k, v in execution.items() if k != "rows"}
                    if meta:
                        payload.setdefault("meta", {}).update(meta)
                else:
                    payload["rows"] = execution

            explain_parts: List[str] = []
            if retrieved:
                explain_parts.append(f"{len(retrieved)} retrieved document(s) used for RAG context.")
            if raw:
                explain_parts.append("Raw LLM output present.")

            if explain_parts:
                payload["explain"] = " ".join(explain_parts)

            # compose human-friendly output (answer-first)
            if self.pretty:
                output_lines: List[str] = []
                # If SQL is short, show inline, otherwise show a one-line summary
                if sql and len(sql) < 240:
                    output_lines.append(f"SQL used: {sql}")
                else:
                    # very long SQL — show single-line summary (first line)
                    output_lines.append(f"SQL (truncated): {sql[:240]}{'...' if len(sql) > 240 else ''}")

                if payload.get("rows") is not None:
                    # try to show a concise preview
                    try:
                        preview = payload["rows"][:3] if isinstance(payload["rows"], list) else str(payload["rows"])
                        output_lines.append(f"Preview rows: {json.dumps(preview, default=str)}")
                    except Exception:
                        output_lines.append("Preview rows: (unavailable)")

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


# Adapter kept for backward compatibility and for GraphBuilder imports
class FormatAdapter:
    """
    Wraps a format_node instance that may accept a different signature.
    Adapter will attempt to call the underlying node with the modern signature,
    then try fewer args (positional), then finally fall back to calling with only sql.
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
        except Exception:
            raise
