# tests/test_error_node.py
import os
import sys

# Ensure project root on sys.path so imports work when running the test directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from app.graph.nodes.error_node import ErrorNode
from app.exception import CustomException


def test_error_node_with_value_error():
    node = ErrorNode()
    try:
        raise ValueError("something went wrong")
    except Exception as e:
        out = node.run(e, step="generate", context={"user": "alice"})
        assert out["ok"] is False
        assert "something went wrong" in out["error"]
        assert "ValueError" in out["details"] or "something went wrong" in out["details"]
        assert out["step"] == "generate"
        assert out["trace"] is not None and "ValueError" in out["trace"]


def test_error_node_with_custom_exception():
    node = ErrorNode()
    try:
        # Create a CustomException instance the same way code does
        raise CustomException("custom failure", sys)
    except Exception as e:
        out = node.run(e, step="validate")
        assert out["ok"] is False
        assert "custom failure" in out["error"]
        assert out["step"] == "validate"
        assert out["trace"] is not None


def test_error_node_failsafe_on_bad_input():
    node = ErrorNode()
    # Pass a non-exception value and ensure node still returns normalized payload
    out = node.run("plain string error", step="execute")
    assert out["ok"] is False
    assert "plain string error" in out["error"] or "plain string error" in out["details"]
    assert out["step"] == "execute"
