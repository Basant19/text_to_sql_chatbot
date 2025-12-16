# File: app/graph/nodes/error_node.py
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
        """Return a normalized dict with short message, details, and stack trace."""
        try:
            short_msg = "Error"
            details = repr(exc)
            trace_str = None
            extra = {}

            # Extract message and details
            if exc is not None:
                short_msg = getattr(exc, "error_message", None) or getattr(exc, "message", None) or str(exc)
                details = repr(exc)

            # Extract traceback if possible
            try:
                if isinstance(exc, BaseException):
                    trace_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            except Exception:
                trace_str = None

            # Include extra non-private attributes
            try:
                if hasattr(exc, "__dict__"):
                    extra = {k: v for k, v in vars(exc).items()
                             if not k.startswith("_") and k not in ("args", "__traceback__")}
            except Exception:
                extra = {}

            result: Dict[str, Any] = {"error": short_msg, "details": details, "trace": trace_str}
            if extra:
                result["extra"] = extra
            return result
        except Exception as e:
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

            # Normalize exception
            norm = self._normalize_exception(exc)
            short_msg = norm.get("error") or "Error"
            details = norm.get("details")
            trace = norm.get("trace")

            # Log full trace for observability
            if trace:
                logger.error("ErrorNode caught exception at step=%s: %s\n%s", step, short_msg, trace)
            else:
                logger.error("ErrorNode caught exception at step=%s: %s", step, short_msg)

            # Build user-facing message
            user_message = "An error occurred while processing your request."
            if step:
                user_message += f" Step: {step}."
            if short_msg and short_msg.lower() not in ("", "error"):
                user_message += f" Reason: {short_msg}"

            # Limit trace length for debug
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
            # Fallback if normalization fails
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
