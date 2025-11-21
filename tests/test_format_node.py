import sys
import os

# allow direct execution of this test file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest

from app.graph.nodes.format_node import FormatNode


def test_format_node_normal_result():
    node = FormatNode()
    exec_result = {
        "rows": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Carol"},
        ],
        "columns": ["id", "name"],
        "meta": {"rowcount": 3, "runtime": 0.002},
    }
    out = node.run(exec_result, max_preview=2)
    assert isinstance(out, dict)
    assert out["rowcount"] == 3
    assert "Alice" in out["preview_text"]
    assert len(out["rows"]) == 2  # limited to max_preview
    assert out["columns"] == ["id", "name"]
    assert out["runtime"] == pytest.approx(0.002, rel=1e-2)


def test_format_node_empty_result():
    node = FormatNode()
    out = node.run(None)
    assert out["rowcount"] == 0
    assert out["columns"] == []
    assert out["rows"] == []
    assert out["preview_text"] == ""


def test_format_node_handles_tuple_rows():
    node = FormatNode()
    exec_result = {
        "rows": [
            (1, "Alice"),
            (2, "Bob"),
        ],
        "columns": ["id", "name"],
        "meta": {"rowcount": 2, "runtime": 0.001},
    }
    out = node.run(exec_result, max_preview=10)
    assert len(out["rows"]) == 2
    assert out["rows"][0]["name"] == "Alice"
    assert "id=1, name=Alice" in out["preview_text"]


if __name__ == "__main__":
    ok = pytest.main([__file__])
    if ok != 0:
        sys.exit(1)
