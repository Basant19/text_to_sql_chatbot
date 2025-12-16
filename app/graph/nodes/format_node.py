import sys
import json
from typing import Any, Dict, List, Optional, Union

from app.logger import get_logger
from app.exception import CustomException

logger = get_logger("format_node")

__all__ = ["FormatNode", "FormatAdapter"]


def _safe_json_dumps(obj: Any, max_len: int = 500) -> str:
    """Safely stringify objects with length limit."""
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        try:
            s = str(obj)
        except Exception:
            s = "<unserializable>"
    return s if len(s) <= max_len else s[:max_len] + "..."


class FormatNode:
    """
    FormatNode
    ----------
    Final presentation layer for the Text-to-SQL pipeline.

    Responsibilities:
      - Normalize execution output
      - Surface validation feedback
      - Produce human-friendly text output
      - Remain GraphBuilder-safe (keyword-only run signature)
    """

    def __init__(self, pretty: bool = True):
        self.pretty = bool(pretty)
        logger.info("FormatNode initialized (pretty=%s)", self.pretty)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    def _normalize_execution(self, execution: Any) -> Dict[str, Any]:
        """Normalize execution into {rows, rowcount, meta}."""
        out: Dict[str, Any] = {"rows": None, "rowcount": None, "meta": {}}

        if execution is None:
            return out

        if isinstance(execution, dict):
            rows = execution.get("rows") or execution.get("data") or execution.get("results")
            out["rows"] = rows
            out["rowcount"] = execution.get("rowcount") or (
                len(rows) if isinstance(rows, list) else None
            )
            out["meta"] = {k: v for k, v in execution.items() if k not in ("rows", "data", "results")}
            return out

        if isinstance(execution, list):
            out["rows"] = execution
            out["rowcount"] = len(execution)
            return out

        out["meta"]["raw"] = execution
        return out

    def _extract_validation(self, obj: Any) -> Optional[Dict[str, Any]]:
        """Extract validation payload from arbitrary dicts."""
        if not isinstance(obj, dict):
            return None

        candidates = [
            obj,
            obj.get("validation") if isinstance(obj.get("validation"), dict) else None,
            obj.get("formatted") if isinstance(obj.get("formatted"), dict) else None,
            obj.get("meta") if isinstance(obj.get("meta"), dict) else None,
        ]

        for c in candidates:
            if not isinstance(c, dict):
                continue
            if any(k in c for k in ("valid", "errors", "fixes", "suggested_sql", "suggestions")):
                return {
                    "valid": c.get("valid"),
                    "errors": c.get("errors") or [],
                    "fixes": c.get("fixes") or [],
                    "suggested_sql": c.get("suggested_sql"),
                    "suggestions": c.get("suggestions") or {},
                }
        return None

    # ------------------------------------------------------------------
    # GraphBuilder-safe entrypoint
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        sql: Optional[str] = None,
        schemas: Optional[Dict[str, Any]] = None,
        retrieved: Optional[List[Dict[str, Any]]] = None,
        execution: Optional[Union[Dict[str, Any], List[Any]]] = None,
        raw: Optional[Any] = None,
    ) -> Dict[str, Any]:
        try:
            sql = (sql or "").strip()
            retrieved = retrieved if isinstance(retrieved, list) else []

            payload: Dict[str, Any] = {"sql": sql}

            # Normalize execution
            exec_norm = self._normalize_execution(execution)
            rows = exec_norm.get("rows")
            payload["rows"] = rows

            # -----------------------------
            # Meta block
            # -----------------------------
            meta: Dict[str, Any] = {}

            if exec_norm.get("meta"):
                meta["execution_meta"] = exec_norm["meta"]

            validation = None
            if exec_norm.get("meta"):
                validation = self._extract_validation(exec_norm["meta"])
            if not validation and raw is not None:
                validation = self._extract_validation(raw)

            if validation:
                meta["validation"] = validation

            if raw is not None:
                meta["raw_summary"] = _safe_json_dumps(raw, max_len=1000)

            if retrieved:
                meta["retrieved_count"] = len(retrieved)
                meta["retrieved_preview"] = [
                    _safe_json_dumps(r.get("text") if isinstance(r, dict) else r, max_len=200)
                    for r in retrieved[:3]
                ]

            # -----------------------------
            # Explanation
            # -----------------------------
            explain_parts: List[str] = []

            if retrieved:
                explain_parts.append(f"{len(retrieved)} retrieved document(s) used.")
            if validation:
                explain_parts.append(
                    "Validation passed." if validation.get("valid") else "Validation failed."
                )
            if raw is not None:
                explain_parts.append("Raw LLM output available.")

            if explain_parts:
                payload["explain"] = " ".join(explain_parts)

            # -----------------------------
            # Human-readable output
            # -----------------------------
            if self.pretty:
                lines: List[str] = []

                if sql:
                    lines.append(
                        f"SQL used: {sql}" if len(sql) <= 240 else f"SQL (truncated): {sql[:240]}..."
                    )

                if rows is not None:
                    preview = rows[:3] if isinstance(rows, list) else rows
                    lines.append(f"Preview rows: {_safe_json_dumps(preview, max_len=400)}")
                    lines.append(
                        f"Returned rows: {exec_norm.get('rowcount') or 'unknown'}"
                    )

                if validation:
                    if not validation.get("valid"):
                        lines.append(f"Errors: {', '.join(map(str, validation.get('errors', [])))}")
                        if validation.get("suggested_sql"):
                            lines.append("Suggested SQL rewrite available.")
                    else:
                        lines.append("Validation: OK")

                if payload.get("explain"):
                    lines.append(f"Notes: {payload['explain']}")

                payload["output"] = "\n".join(lines)
            else:
                payload["output"] = {
                    "sql": sql,
                    "rows": rows,
                    "explain": payload.get("explain"),
                }

            if meta:
                payload["meta"] = meta

            logger.info("FormatNode completed (rows=%s)", exec_norm.get("rowcount"))
            return payload

        except Exception as e:
            logger.exception("FormatNode.run failed")
            raise CustomException(e, sys)


class FormatAdapter:
    """
    Backward-compatible adapter for legacy pipelines.
    """

    def __init__(self, fmt_node: FormatNode):
        self._node = fmt_node

    def run(self, *args, **kwargs):
        try:
            return self._node.run(**kwargs)
        except TypeError:
            if args:
                return self._node.run(sql=args[0])
            raise
