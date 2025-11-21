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
        payload = node.run(exc, step="generate", context={"user_query": "..."})
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
            # nothing special to init for now
            pass
        except Exception as e:
            logger.exception("Failed to initialize ErrorNode")
            raise CustomException(e, sys)

    def run(self, exc: Exception, step: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Normalize the exception and return a dictionary suitable for UI/telemetry.

        :param exc: Exception instance (or anything convertible to str)
        :param step: optional step identifier in which the exception happened
        :param context: optional context dict (will not be deeply serialized here)
        :return: normalized error dict
        """
        try:
            # Build short message and details
            short_msg = str(exc) or exc.__class__.__name__
            # If CustomException wraps details differently, attempt to pull meaningful info
            if isinstance(exc, CustomException):
                # CustomException in this project expects its message to be human-friendly
                details = getattr(exc, "error_message", None) or short_msg
            else:
                details = repr(exc)

            # Build stack trace
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if isinstance(exc, BaseException) else None

            payload = {
                "ok": False,
                "error": short_msg,
                "details": details,
                "step": step,
                "trace": tb,
            }

            # Log with traceback for visibility
            if tb:
                logger.error("ErrorNode caught exception at step=%s: %s\n%s", step, short_msg, tb)
            else:
                logger.error("ErrorNode caught exception at step=%s: %s", step, short_msg)

            return payload
        except Exception as e:
            # Failsafe: if error normalization itself fails, return a minimal payload
            tb2 = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            logger.exception("ErrorNode failed while formatting exception: %s", tb2)
            return {
                "ok": False,
                "error": "ErrorNode failed to normalize exception",
                "details": repr(e),
                "step": step,
                "trace": tb2,
            }
