import sys
import pytest
from app.exception import CustomException, error_message_detail

def test_error_message_detail_contains_file_and_line():
    try:
        raise ValueError("sample error")
    except Exception as e:
        msg = error_message_detail(e, sys)
        assert "Error occurred in script" in msg
        assert "sample error" in msg

def test_custom_exception_str_contains_details():
    try:
        try:
            raise KeyError("inner")
        except Exception as inner:
            raise CustomException(inner, sys)
    except CustomException as ce:
        s = str(ce)
        assert "Error occurred in script" in s
        assert "inner" in s
