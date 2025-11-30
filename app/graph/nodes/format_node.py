# app/graph/nodes/format_node.py
import sys
import json
from typing import Any, Dict, List, Optional

from app.logger import get_logger
from app.exception import CustomException
import app.config as config_module

logger = get_logger("format_node")


def _truncate(obj: Any, max_chars: int = 1000) -> str:
    """Safe truncation for debug summaries."""
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        try:
            s = str(obj)
        except Exception:
            s = "<unserializable>"
    if len(s) > max_chars:
        return s[: max_chars - 3] + "..."
    return s


class FormatNode:
    """
    FormatNode: produce a user-friendly formatted response.

    Preferred run signature:
        run(sql: str, schemas: dict, retrieved: Optional[list],
            execution: Optional[dict], raw: Optional[Any]) -> Dict[str, Any]

    Returns a dict:
    {
      "output": <str|dict>,    # short assistant-visible answer
      "sql": <sql>,
      "explain": <str> (optional),
      "rows": <list> (optional),
      "meta": { "debug": {...}, ... } (optional)
    }
    """
    def __init__(self, pretty: bool = True):
        self.pretty = pretty
        # config flag controls whether to store full raw LLM blobs
        self._store_full_blobs = bool(getattr(config_module, "STORE_FULL_LLM_BLOBS", False))
        logger.info("FormatNode initialized (pretty=%s, store_full_blobs=%s)", self.pretty, self._store_full_blobs)

    def _build_concise_answer(self, sql: str, execution: Optional[Dict[str, Any]]):
        """
        Build short human-friendly content for assistant.message.content and a 1-line explanation.
        Heuristics:
          - If execution contains one scalar value (single cell), show the value as the answer.
          - Else show short "SQL used" explanation and number of rows returned if available.
        """
        answer = ""
        one_line_explain = ""

        # Execution present and has rows
        if execution and isinstance(execution, dict):
            rows = execution.get("rows")
            if isinstance(rows, list) and len(rows) > 0:
                # check single-row, single-column numeric result (e.g., COUNT(*))
                first = rows[0]
                if isinstance(first, dict) and len(first) == 1:
                    val = next(iter(first.values()))
                    # numeric / scalar heuristic
                    if isinstance(val, (int, float)) or (isinstance(val, str) and val.isdigit()):
                        answer = f"{val}"
                        one_line_explain = f"SQL used: {sql}"
                        return answer, one_line_explain
                # fallback: show number of rows
                answer = f"Returned {len(rows)} row(s)."
                one_line_explain = f"SQL used: {sql}"
                return answer, one_line_explain

        # No execution results — produce SQL-first answer
        answer = "Generated SQL (preview)."
        one_line_explain = f"SQL used: {sql}"
        return answer, one_line_explain

    def run(self, sql: str, schemas: Optional[Dict[str, Any]] = None,
            retrieved: Optional[List[Dict[str, Any]]] = None,
            execution: Optional[Dict[str, Any]] = None,
            raw: Optional[Any] = None) -> Dict[str, Any]:
        try:
            payload: Dict[str, Any] = {"sql": sql}

            # rows (if provided)
            if execution is not None:
                # Accept execution shapes: { rows: [...], columns: [...], meta: {...} } or raw rows list
                if isinstance(execution, dict) and "rows" in execution:
                    payload["rows"] = execution.get("rows")
                    payload["execution_meta"] = execution.get("meta", {})
                else:
                    payload["rows"] = execution

            # build short assistant-visible output + one-line explanation
            try:
                answer, explain = self._build_concise_answer(sql, execution)
            except Exception:
                answer = "Answer generated."
                explain = f"SQL used: {sql}"

            payload["output"] = answer
            payload["explain"] = explain

            # developer/debug meta
            debug: Dict[str, Any] = {}
            # include execution meta if present
            if execution and isinstance(execution, dict):
                debug["execution_meta"] = execution.get("meta", {})
                # include validation errors if present under execution.meta or execution itself
                vallike = execution.get("meta", {}) or {}
                if isinstance(vallike, dict) and vallike.get("validation_errors"):
                    debug["validation_errors"] = vallike.get("validation_errors")

                # also try to surface any top-level validation_errors
                if execution.get("validation_errors"):
                    debug["validation_errors"] = execution.get("validation_errors")

            # include retrieved doc summary (counts)
            if retrieved:
                debug["retrieved_count"] = len(retrieved)

            # raw LLM info: either full blob or truncated summary depending on config
            if raw is not None:
                if self._store_full_blobs:
                    # store full raw under raw_blob (careful: can be large)
                    debug["raw_blob"] = raw
                else:
                    debug["raw_summary"] = _truncate(raw, max_chars=1000)

            # always include sql in debug for easier lookup
            debug["sql"] = sql

            # attach timing/meta if present in execution.meta
            if execution and isinstance(execution, dict):
                exec_meta = execution.get("meta") or {}
                if isinstance(exec_meta, dict) and exec_meta.get("runtime") is not None:
                    debug["execution_runtime"] = exec_meta.get("runtime")

            payload["meta"] = {"debug": debug}

            # final payload: include pretty output if desired (text), else structured
            if self.pretty:
                # Compose a friendly textual block for display if UI wants one string
                display_lines = []
                # Use the short answer as first line
                display_lines.append(answer)
                # one-line explanation
                display_lines.append(explain)
                # small preview of rows if available
                if payload.get("rows"):
                    try:
                        preview = payload["rows"][:3] if isinstance(payload["rows"], list) else payload["rows"]
                        preview_str = _truncate(preview, max_chars=800)
                        display_lines.append(f"Preview: {preview_str}")
                    except Exception:
                        pass
                payload["display"] = "\n".join(display_lines)

            logger.info("FormatNode: formatted output ready (sql_len=%s)", len(sql or ""))
            return payload

        except Exception as e:
            logger.exception("FormatNode.run failed")
            raise CustomException(e, sys)
