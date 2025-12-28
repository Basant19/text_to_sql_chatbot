# File: app/graph/nodes/format_node.py
from __future__ import annotations

import sys
import json
from typing import Any, Dict, List, Optional, Union

from app.logger import get_logger
from app.exception import CustomException

logger = get_logger("format_node")

__all__ = ["FormatNode", "FormatAdapter"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _safe_json(obj: Any, max_len: int = 800) -> str:
    """
    Safely stringify objects with truncation.

    Guarantees:
      - NEVER raises
      - ALWAYS returns a string
    """
    try:
        text = json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        try:
            text = str(obj)
        except Exception:
            return "<unserializable>"

    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# ---------------------------------------------------------------------
# FormatNode
# ---------------------------------------------------------------------
class FormatNode:
    """
    FormatNode
    ==========

    Final presentation node in the graph.

    HARD GUARANTEES
    ---------------
    - SQL is ALWAYS preserved
    - rows are NEVER dropped (always list)
    - columns are preserved if present
    - formatted_text is UI-only (no row data)
    """

    def __init__(self, *, pretty: bool = True):
        self.pretty = bool(pretty)
        logger.info("FormatNode initialized | pretty=%s", self.pretty)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    def _normalize_execution(self, execution: Any) -> Dict[str, Any]:
        """
        Normalize execution output.

        Returns:
        {
            "rows": list,
            "columns": list,
            "rowcount": int,
            "meta": dict
        }
        """
        out = {
            "rows": [],
            "columns": [],
            "rowcount": 0,
            "meta": {},
        }

        if execution is None:
            return out

        if isinstance(execution, dict):
            rows = execution.get("rows") or []
            columns = execution.get("columns") or []

            if not isinstance(rows, list):
                rows = []

            if not isinstance(columns, list):
                columns = []

            out["rows"] = rows
            out["columns"] = columns
            out["rowcount"] = len(rows)

            # Everything else → meta
            out["meta"] = {
                k: v
                for k, v in execution.items()
                if k not in ("rows", "columns")
            }
            return out

        if isinstance(execution, list):
            out["rows"] = execution
            out["rowcount"] = len(execution)
            return out

        # Unknown type
        out["meta"]["raw_execution"] = execution
        return out

    def _extract_validation(self, source: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(source, dict):
            return None

        v = source.get("validation") if isinstance(source.get("validation"), dict) else source

        if not any(k in v for k in ("valid", "errors", "suggested_sql")):
            return None

        return {
            "valid": bool(v.get("valid")),
            "errors": v.get("errors") or [],
            "suggested_sql": v.get("suggested_sql"),
        }

    # ------------------------------------------------------------------
    # Graph entrypoint
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
        """
        Final UI-safe output.

        Returns:
        {
            "sql": str,
            "rows": list,
            "columns": list,
            "formatted_text": str,
            "meta": dict
        }
        """
        try:
            sql = (sql or "").strip()
            retrieved = retrieved if isinstance(retrieved, list) else []

            payload: Dict[str, Any] = {
                "sql": sql,
                "rows": [],
                "columns": [],
                "formatted_text": "",
            }

            # --------------------------------------------------
            # Normalize execution (🔥 CRITICAL FIX)
            # --------------------------------------------------
            exec_norm = self._normalize_execution(execution)
            payload["rows"] = exec_norm["rows"]
            payload["columns"] = exec_norm["columns"]

            # --------------------------------------------------
            # Meta (debug-only)
            # --------------------------------------------------
            meta: Dict[str, Any] = {}

            if exec_norm["meta"]:
                meta["execution_meta"] = exec_norm["meta"]

            validation = None
            if exec_norm["meta"]:
                validation = self._extract_validation(exec_norm["meta"])
            if not validation and raw is not None:
                validation = self._extract_validation(raw)

            if validation:
                meta["validation"] = validation

            if retrieved:
                meta["retrieved_count"] = len(retrieved)

            if raw is not None:
                meta["raw_summary"] = _safe_json(raw)

            # --------------------------------------------------
            # Human-readable summary (NO ROW DATA)
            # --------------------------------------------------
            lines: List[str] = []

            if sql:
                lines.append("### 🧾 SQL Query Used")
                lines.append(f"`{sql}`")

            lines.append(f"### 📊 Rows Returned: {exec_norm['rowcount']}")

            if validation:
                if validation.get("valid"):
                    lines.append("### ✅ Validation passed")
                else:
                    lines.append("### ❌ Validation failed")
                    for err in validation.get("errors", []):
                        lines.append(f"- {err}")

            if retrieved:
                lines.append(f"### 📄 Context used: {len(retrieved)} document(s)")

            payload["formatted_text"] = "\n\n".join(lines)

            if meta:
                payload["meta"] = meta

            logger.info(
                "FormatNode completed | sql_len=%d | rowcount=%d",
                len(sql),
                exec_norm["rowcount"],
            )

            return payload

        except Exception as e:
            logger.exception("FormatNode.run failed")
            raise CustomException(e, sys)


# ---------------------------------------------------------------------
# Adapter (legacy safety)
# ---------------------------------------------------------------------
class FormatAdapter:
    """
    Backward-compatible adapter.
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