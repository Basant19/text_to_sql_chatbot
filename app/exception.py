import sys
import traceback
from typing import Any

def error_message_detail(error: Any, error_detail: sys):
    """
    Build a friendly error message. Works when:
      - error is an Exception (with or without __traceback__)
      - error is a string
      - sys.exc_info() may or may not contain a traceback
    """
    try:
        exc_type, exc_obj, exc_tb = error_detail.exc_info()
    except Exception:
        exc_tb = None

    # Prefer traceback from sys.exc_info()
    if exc_tb:
        try:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        except Exception:
            file_name = "<unknown>"
            line_number = 0
    else:
        # Fall back to traceback on the exception object, if present
        tb = None
        if isinstance(error, BaseException) and getattr(error, "__traceback__", None):
            tb = error.__traceback__
        else:
            # Try extracting the last frame from stack as a last resort
            tb = None

        if tb:
            try:
                file_name = tb.tb_frame.f_code.co_filename
                line_number = tb.tb_lineno
            except Exception:
                file_name = "<unknown>"
                line_number = 0
        else:
            # No traceback available — use placeholders
            file_name = "<unknown>"
            line_number = 0

    return f"Error occurred in script [{file_name}] line [{line_number}] message [{str(error)}]"


class CustomException(Exception):
    def __init__(self, error_message: Any, error_details: sys = None):
        """
        error_message can be an Exception or a string.
        error_details should be the 'sys' module (or similar) so we can call exc_info().
        If error_details is not provided, we use the current sys module.
        """
        # keep original message in base Exception
        super().__init__(str(error_message))
        if error_details is None:
            import sys as _sys
            error_details = _sys

        # Build the more detailed message safely
        try:
            self.error_message = error_message_detail(error_message, error_detail=error_details)
        except Exception:
            # As a final fallback, just use the string form
            self.error_message = f"Error: {str(error_message)}"

    def __str__(self):
        return self.error_message
