# app/graph/nodes/error_node.py

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
        payload = node.run(exc, step="generate", context={"user_query": "..."}).

    Returns:
        {
            "ok": False,
            "error": "<short message>",
            "details": "<detailed message or repr>",
            "step": "<step name where error occurred>",
            "trace": "<stack trace string>"
        }
    """

    def __init__(self):
        try:
            logger.info("ErrorNode initialized")
        except Exception as e:
            logger.exception("Failed to initialize ErrorNode")
            raise CustomException(e, sys)

    def run(
        self,
        exc: Exception,
        step: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Normalize the exception and return a dictionary suitable for UI/telemetry.

        Parameters
        ----------
        exc : Exception
            Exception instance (or anything convertible to str)
        step : Optional[str]
            Identifier for the step where the exception occurred
        context : Optional[Dict[str, Any]]
            Additional context information (not deeply serialized)

        Returns
        -------
        Dict[str, Any]
            Standardized error payload
        """
        try:
            # Short message
            short_msg = str(exc) or exc.__class__.__name__

            # Details: extract human-friendly message if CustomException
            if isinstance(exc, CustomException):
                details = getattr(exc, "error_message", None) or short_msg
            else:
                details = repr(exc)

            # Stack trace
            trace_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) \
                if isinstance(exc, BaseException) else None

            payload = {
                "ok": False,
                "error": short_msg,
                "details": details,
                "step": step,
                "trace": trace_str,
            }

            # Log for observability
            if trace_str:
                logger.error("ErrorNode caught exception at step=%s: %s\n%s", step, short_msg, trace_str)
            else:
                logger.error("ErrorNode caught exception at step=%s: %s", step, short_msg)

            return payload

        except Exception as e:
            # Failsafe: if normalization itself fails
            fallback_trace = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            logger.exception("ErrorNode failed while formatting exception: %s", fallback_trace)
            return {
                "ok": False,
                "error": "ErrorNode failed to normalize exception",
                "details": repr(e),
                "step": step,
                "trace": fallback_trace,
            }
