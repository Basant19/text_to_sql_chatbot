#D:\text_to_sql_bot\app\graph\nodes\error_node.py
import sys
import traceback
from typing import Any, Dict, Optional

from app.logger import get_logger
from app.exception import CustomException

logger = get_logger("error_node")


class ErrorNode:
    """
    Node that normalizes exceptions into a standard payload for the graph/UI.

    Usage:
        node = ErrorNode()
        payload = node.run(exc, step="generate", context={"user_query": "..."})

    Returned payload (GraphBuilder-friendly):
        {
            "prompt": None,
            "sql": None,
            "valid": False,
            "execution": None,
            "formatted": { "output": "<user-friendly msg>", "meta": {"debug": {...}} },
            "raw": { ... normalized error object ... },
            "error": "<short message>",
            "timings": context.get("timings", {})
        }
    """

    def __init__(self):
        try:
            logger.info("ErrorNode initialized")
        except Exception as e:
            logger.exception("Failed to initialize ErrorNode")
            raise CustomException(e, sys)

    def _normalize_exception(self, exc: Any) -> Dict[str, Any]:
        """Return a normalized dict with short message, details and stack trace (if available)."""
        try:
            short_msg = None
            details = None

            # If it's our CustomException, try to extract a friendly message
            try:
                if isinstance(exc, CustomException):
                    short_msg = getattr(exc, "error_message", None) or getattr(exc, "message", None) or str(exc)
                    details = repr(exc)
                else:
                    short_msg = str(exc) if exc is not None else "Unknown error"
                    details = repr(exc)
            except Exception:
                short_msg = str(exc) if exc is not None else "Unknown error"
                details = repr(exc)

            # Get trace if available
            trace_str = None
            try:
                if isinstance(exc, BaseException):
                    trace_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                else:
                    # If it's not an exception, we won't have a trace
                    trace_str = None
            except Exception:
                trace_str = None

            # include any extra attributes (non-private) if present
            extra = {}
            try:
                if hasattr(exc, "__dict__"):
                    d = {k: v for k, v in vars(exc).items() if not k.startswith("_") and k not in ("args", "__traceback__")}
                    if d:
                        extra = d
            except Exception:
                extra = {}

            result: Dict[str, Any] = {"error": short_msg or "Error", "details": details or short_msg, "trace": trace_str}
            if extra:
                result["extra"] = extra
            return result
        except Exception as e:
            # fallback
            logger.exception("ErrorNode._normalize_exception failed")
            return {"error": "Failed to normalize exception", "details": repr(e), "trace": None}

    def run(
        self,
        exc: Any,
        step: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Normalize the exception and return a GraphBuilder-compatible error payload.

        Parameters
        ----------
        exc : Exception | Any
            The exception or error-like object.
        step : Optional[str]
            Identifier for pipeline step (generate/validate/execute/etc.)
        context : Optional[Dict[str, Any]]
            Additional run-time context (may include timings).
        """
        try:
            ctx = context or {}
            timings = ctx.get("timings", {}) if isinstance(ctx, dict) else {}

            # Build normalized error dict for logs and developer debug
            norm = self._normalize_exception(exc)
            short_msg = norm.get("error") or "Error"
            details = norm.get("details")
            trace = norm.get("trace")

            # Log with trace for observability
            if trace:
                logger.error("ErrorNode caught exception at step=%s: %s\n%s", step, short_msg, trace)
            else:
                logger.error("ErrorNode caught exception at step=%s: %s", step, short_msg)

            # Build a concise user-facing formatted response
            user_message = "An error occurred while processing your request."
            if step:
                user_message += f" Step: {step}."
            # include a short reason if available (keep it user-friendly)
            if short_msg and short_msg.lower() not in ("", "error"):
                user_message += f" Reason: {short_msg}"

            # Truncate trace for debug payload to avoid huge dumps
            trace_preview = None
            try:
                if trace:
                    MAX_TRACE = 20000
                    trace_preview = trace if len(trace) <= MAX_TRACE else trace[:MAX_TRACE] + "..."
            except Exception:
                trace_preview = None

            formatted = {
                "output": user_message,
                "explain": "See developer details for full error information.",
                "meta": {
                    "debug": {
                        "step": step,
                        "error_short": short_msg,
                        "error_details": details,
                        "error_trace": trace_preview,
                        "context": {k: v for k, v in (ctx.items() if isinstance(ctx, dict) else [])},
                    }
                },
            }

            # GraphBuilder-compatible error payload
            return {
                "prompt": None,
                "sql": None,
                "valid": False,
                "execution": None,
                "formatted": formatted,
                "raw": norm,
                "error": short_msg,
                "timings": timings or {},
            }

        except Exception as e:
            # If normalization itself fails, return a minimal error payload
            fallback = {"error": "ErrorNode failed while handling another error", "details": repr(e)}
            logger.exception("ErrorNode.run failed while handling exception: %s", fallback)
            return {
                "prompt": None,
                "sql": None,
                "valid": False,
                "execution": None,
                "formatted": {"output": "Internal error", "meta": {"debug": {"fallback": fallback}}},
                "raw": fallback,
                "error": fallback.get("error"),
                "timings": context.get("timings") if isinstance(context, dict) else {},
            }
