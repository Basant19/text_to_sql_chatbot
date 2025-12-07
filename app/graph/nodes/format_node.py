import sys
import json
from typing import Any, Dict, List, Optional, Union

from app.logger import get_logger
from app.exception import CustomException

logger = get_logger("format_node")

__all__ = ["FormatNode", "FormatAdapter"]


def _safe_json_dumps(obj: Any, max_len: int = 500) -> str:
    """Helper to produce a short json-ish preview for complex objects."""
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        try:
            s = str(obj)
        except Exception:
            s = "<unserializable>"
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


class FormatNode:
    """
    FormatNode: produce a user-friendly formatted response.

    Preferred run signature:
        run(sql: str, schemas: dict, retrieved: Optional[list],
            execution: Optional[dict], raw: Optional[Any]) -> Dict[str, Any]

    Returned shape (typical):
    {
      "sql": "<sql-used-or-generated>",
      "rows": [...],                  # if execution provided
      "output": "<human-friendly text>",
      "explain": "<short explanation>",
      "meta": {
        "execution_meta": {...},
        "validation": {
            "valid": bool,
            "errors": [...],
            "fixes": [...],
            "suggested_sql": "...",
            "suggestions": {...}
        },
        "raw": <raw dict or summary>
      }
    }
    """
    def __init__(self, pretty: bool = True):
        self.pretty = pretty
        logger.info("FormatNode initialized (pretty=%s)", self.pretty)

    def _normalize_execution(self, execution: Any) -> Dict[str, Any]:
        """
        Normalize different execution return shapes into a dict with keys:
        - rows (list or None)
        - rowcount (int or None)
        - meta (other metadata)
        """
        out: Dict[str, Any] = {"rows": None, "rowcount": None, "meta": {}}
        if execution is None:
            return out

        if isinstance(execution, dict):
            # Typical keys: 'rows', 'data', 'results', 'rowcount', 'timings', 'meta'
            rows = execution.get("rows") or execution.get("data") or execution.get("results") or None
            out["rows"] = rows
            out["rowcount"] = execution.get("rowcount") or (len(rows) if isinstance(rows, list) else None)
            # copy everything except 'rows' to meta
            meta = {k: v for k, v in execution.items() if k not in ("rows", "data", "results")}
            out["meta"].update(meta)
            return out

        # If execution is a list -> treat as rows
        if isinstance(execution, list):
            out["rows"] = execution
            out["rowcount"] = len(execution)
            return out

        # fallback: just stringify
        out["meta"]["raw"] = execution
        return out

    def _extract_validation_from_any(self, candidate: Any) -> Optional[Dict[str, Any]]:
        """
        If candidate dict contains validation-like keys, extract them into a normalized dict:
        { valid, errors, fixes, suggested_sql, suggestions }
        """
        if not isinstance(candidate, dict):
            return None
        # possible places for validation info: top-level or candidate.get("validation") or candidate.get("formatted")
        sources = [candidate]
        if "validation" in candidate:
            sources.insert(0, candidate.get("validation"))
        if "formatted" in candidate and isinstance(candidate["formatted"], dict):
            sources.insert(0, candidate["formatted"])
        if "meta" in candidate and isinstance(candidate["meta"], dict):
            sources.insert(0, candidate["meta"])

        for s in sources:
            if not isinstance(s, dict):
                continue
            if any(k in s for k in ("valid", "errors", "fixes", "suggested_sql", "suggestions")):
                return {
                    "valid": s.get("valid"),
                    "errors": s.get("errors"),
                    "fixes": s.get("fixes"),
                    "suggested_sql": s.get("suggested_sql"),
                    "suggestions": s.get("suggestions"),
                }
        return None

    def run(
        self,
        sql: str,
        schemas: Optional[Dict[str, Any]] = None,
        retrieved: Optional[List[Dict[str, Any]]] = None,
        execution: Optional[Union[Dict[str, Any], List[Any]]] = None,
        raw: Optional[Any] = None,
    ) -> Dict[str, Any]:
        try:
            payload: Dict[str, Any] = {"sql": sql or ""}

            # Normalize execution to consistent shape
            exec_norm = self._normalize_execution(execution)
            rows = exec_norm.get("rows")
            payload["rows"] = rows

            # Build meta
            meta: Dict[str, Any] = {}
            if exec_norm.get("meta"):
                meta["execution_meta"] = exec_norm["meta"]

            # Pull validation info from raw or exec meta if present
            validation = None
            # check execution meta first
            if exec_norm.get("meta"):
                validation = self._extract_validation_from_any(exec_norm["meta"])
            if not validation and raw:
                validation = self._extract_validation_from_any(raw)
            # also check raw.get('formatted') or raw itself if dict
            if not validation and isinstance(raw, dict):
                validation = self._extract_validation_from_any(raw)

            if validation:
                # place into meta.validation
                meta["validation"] = validation

            # Also include raw summary for debugging (but avoid huge dumps)
            if raw is not None:
                try:
                    meta["raw_summary"] = _safe_json_dumps(raw, max_len=1000)
                except Exception:
                    meta["raw_summary"] = str(raw)

            # If retrieved context present, include a short summary/preview
            if retrieved:
                try:
                    meta["retrieved_count"] = len(retrieved)
                    # include a small preview of the first retrieved doc titles/snippets if available
                    try:
                        preview = []
                        for doc in retrieved[:3]:
                            if isinstance(doc, dict):
                                if "title" in doc:
                                    preview.append(doc.get("title"))
                                elif "text" in doc:
                                    preview.append((doc.get("text") or "")[:200])
                                else:
                                    preview.append(_safe_json_dumps(doc, max_len=160))
                            else:
                                preview.append(_safe_json_dumps(doc, max_len=160))
                        meta["retrieved_preview"] = preview
                    except Exception:
                        meta["retrieved_preview"] = "(unavailable)"
                except Exception:
                    pass

            # Compose explanation parts
            explain_parts: List[str] = []
            if retrieved:
                explain_parts.append(f"{len(retrieved)} retrieved document(s) used for context.")
            if raw is not None:
                explain_parts.append("LLM raw output available.")
            if validation:
                if validation.get("valid") is False:
                    explain_parts.append("Validation failed.")
                elif validation.get("valid") is True:
                    explain_parts.append("Validation passed.")
                if validation.get("suggested_sql"):
                    explain_parts.append("A suggested SQL rewrite is available.")

            if explain_parts:
                payload["explain"] = " ".join(explain_parts)

            # Build human-friendly output text
            if self.pretty:
                out_lines: List[str] = []
                # Show a brief natural answer if raw contains a short 'answer' or formatted 'output'
                shown_answer = None
                if isinstance(raw, dict):
                    # common places: raw['formatted']['output'] or raw['output'] or raw.get('answer')
                    formatted = raw.get("formatted") if isinstance(raw.get("formatted"), dict) else None
                    if formatted:
                        shown_answer = formatted.get("output") or formatted.get("explain") or None
                    if not shown_answer:
                        shown_answer = raw.get("output") or raw.get("answer")
                # Fallback to execution meta or rows
                if shown_answer and isinstance(shown_answer, str) and len(shown_answer.strip()) > 0:
                    # keep only a compact version
                    out_lines.append(shown_answer.strip())
                else:
                    # show SQL preview first (if present)
                    if sql:
                        if len(sql) < 240:
                            out_lines.append(f"SQL used: {sql}")
                        else:
                            out_lines.append(f"SQL (truncated): {sql[:240]}{'...' if len(sql) > 240 else ''}")

                # Rows preview
                if rows is not None:
                    try:
                        preview_rows = rows[:3] if isinstance(rows, list) else rows
                        out_lines.append(f"Preview rows: {_safe_json_dumps(preview_rows, max_len=400)}")
                        out_lines.append(f"Returned rows: {exec_norm.get('rowcount') or (len(rows) if isinstance(rows, list) else 'unknown')}")
                    except Exception:
                        out_lines.append("Preview rows: (unavailable)")

                # Validation notes
                if validation:
                    if validation.get("valid") is False:
                        errs = validation.get("errors") or []
                        out_lines.append(f"Validation failed: {', '.join([str(e) for e in errs])}")
                        fixes = validation.get("fixes") or []
                        if fixes:
                            out_lines.append(f"Fixes: {', '.join([str(f) for f in fixes])}")
                        # show suggestions mapping
                        suggestions = validation.get("suggestions") or {}
                        if suggestions:
                            out_lines.append(f"Suggested mapping: {_safe_json_dumps(suggestions, max_len=400)}")
                        if validation.get("suggested_sql"):
                            out_lines.append("A suggested SQL rewrite is available (see details).")
                    elif validation.get("valid") is True:
                        out_lines.append("Validation: OK.")

                # Additional small notes
                if payload.get("explain"):
                    out_lines.append(f"Notes: {payload['explain']}")

                payload["output"] = "\n".join(out_lines) if out_lines else ""
            else:
                payload["output"] = {
                    "sql": sql,
                    "rows": rows,
                    "explain": payload.get("explain"),
                }

            # Attach meta (structured)
            if meta:
                payload["meta"] = meta

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
