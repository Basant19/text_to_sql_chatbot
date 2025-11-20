import sys
import os

# allow direct test run (so `python tests/test_execute_node.py` works)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from unittest.mock import patch

from app.graph.nodes.execute_node import ExecuteNode
from app.exception import CustomException


def test_execute_node_success(monkeypatch):
    node = ExecuteNode()

    fake_result = {
        "rows": [{"id": 1, "name": "Alice"}],
        "columns": ["id", "name"],
        "meta": {"rowcount": 1, "runtime": 0.001},
    }

    # patch sql_executor.execute_sql to return fake_result
    monkeypatch.setattr("app.graph.nodes.execute_node.sql_executor.execute_sql", lambda sql, read_only, limit: fake_result)

    res = node.run("SELECT id, name FROM users", limit=10, read_only=True)
    assert isinstance(res, dict)
    assert res["meta"]["rowcount"] == 1
    assert res["rows"][0]["name"] == "Alice"


def test_execute_node_propagates_custom_exception(monkeypatch):
    node = ExecuteNode()

    def raise_ce(sql, read_only, limit):
        raise CustomException("boom", sys)

    monkeypatch.setattr("app.graph.nodes.execute_node.sql_executor.execute_sql", raise_ce)

    with pytest.raises(CustomException):
        node.run("SELECT 1", limit=5, read_only=True)


def test_execute_node_passes_limit(monkeypatch):
    node = ExecuteNode()

    called = {}

    def fake_exec(sql, read_only, limit):
        called["sql"] = sql
        called["limit"] = limit
        return {"rows": [], "columns": [], "meta": {"rowcount": 0}}

    monkeypatch.setattr("app.graph.nodes.execute_node.sql_executor.execute_sql", fake_exec)

    node.run("SELECT * FROM t", limit=123, read_only=True)
    assert called["limit"] == 123
    assert "SELECT" in called["sql"]


# Allow running this test file directly
if __name__ == "__main__":
    ok = pytest.main([__file__])
    if ok != 0:
        sys.exit(1)
