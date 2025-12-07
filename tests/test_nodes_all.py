# tests/test_nodes_all.py

import os
import uuid
import pytest

# nodes under test
from app.graph.nodes.validate_node import ValidateNode
from app.graph.nodes.format_node import FormatNode
from app.graph.nodes.error_node import ErrorNode
from app.graph.nodes.context_node import ContextNode

import app.graph.nodes.validate_node as validate_node_module
import app.graph.nodes.context_node as context_node_module
import app.graph.nodes.format_node as format_node_module
import app.graph.nodes.error_node as error_node_module


# --- fixtures / helpers -----------------------------------------------------
@pytest.fixture(autouse=True)
def patch_utils_monkey(monkeypatch):
    """
    Provide minimal implementations for the utils functions ValidateNode expects.
    Tests can override particular behaviors by monkeypatching validate_node_module.utils.*.
    """
    # minimal canonicalizer: lowercase & replace non-word with underscore
    monkeypatch.setattr(validate_node_module.utils, "_canonicalize_name", lambda s: (s or "").strip().lower().replace(" ", "_"))
    monkeypatch.setattr(validate_node_module.utils, "is_select_query", lambda sql: (sql or "").strip().lower().startswith("select"))
    # fallback extractors (not relied on when sqlglot present); safe defaults
    monkeypatch.setattr(validate_node_module.utils, "extract_table_names_from_sql", lambda sql: [])
    monkeypatch.setattr(validate_node_module.utils, "extract_column_names_from_sql", lambda sql: [])
    monkeypatch.setattr(validate_node_module.utils, "validate_tables_in_sql", lambda sql, schemas: [])
    monkeypatch.setattr(validate_node_module.utils, "validate_columns_in_sql", lambda sql, schemas: [])
    # default row-limit helpers (tests can override)
    monkeypatch.setattr(validate_node_module.utils, "exceeds_row_limit", lambda sql, limit: False)
    monkeypatch.setattr(validate_node_module.utils, "limit_sql_rows", lambda sql, limit: sql)
    yield


def make_sample_schemas():
    """
    Return a simple schema dict as expected by ValidateNode.
    Keys are store_keys; each value should contain 'columns' and 'columns_normalized' ideally.
    """
    return {
        "apps": {
            "canonical": "apps",
            "aliases": ["apps", "applications"],
            "path": "/tmp/apps.csv",
            "columns": ["App", "Rating"],
            "columns_normalized": ["app", "rating"]
        },
        "users": {
            "canonical": "users",
            "aliases": ["users"],
            "path": "/tmp/users.csv",
            "columns": ["Name", "Email"],
            "columns_normalized": ["name", "email"]
        },
    }


# ----------------- ValidateNode tests ---------------------------------------
def test_validate_node_valid_select_passes():
    schemas = make_sample_schemas()
    node = ValidateNode()
    sql = "SELECT App, Rating FROM apps"
    out = node.run(sql, schemas)
    assert out["valid"] is True
    assert out["errors"] == []
    assert isinstance(out.get("fixes"), list)


def test_validate_node_missing_table_detected():
    schemas = make_sample_schemas()
    node = ValidateNode()
    sql = "SELECT foo FROM unknown_table"
    out = node.run(sql, schemas)
    assert out["valid"] is False
    assert any("Tables not found" in e for e in out["errors"])


def test_validate_node_missing_column_detected():
    schemas = make_sample_schemas()
    node = ValidateNode()
    sql = "SELECT not_a_col FROM apps"
    out = node.run(sql, schemas)
    assert out["valid"] is False
    assert any("Columns not found" in e for e in out["errors"])


def test_validate_node_forbidden_table(monkeypatch):
    schemas = make_sample_schemas()
    node = ValidateNode(safety_rules={"forbidden_tables": ["users"]})
    sql = "SELECT name FROM users"
    out = node.run(sql, schemas)
    assert out["valid"] is False
    assert any("forbidden" in e.lower() for e in out["errors"])


def test_validate_node_row_limit_enforced(monkeypatch):
    schemas = make_sample_schemas()
    monkeypatch.setattr(validate_node_module.utils, "exceeds_row_limit", lambda sql, limit: True)
    monkeypatch.setattr(validate_node_module.utils, "limit_sql_rows", lambda sql, limit: sql.rstrip(";") + f" LIMIT {limit}")
    node = ValidateNode(safety_rules={"max_row_limit": 1})
    sql = "SELECT App FROM apps"
    out = node.run(sql, schemas)
    assert "LIMIT 1" in (out.get("sql") or "") or (out.get("suggested_sql") and "LIMIT 1" in out.get("suggested_sql"))


# ----------------- FormatNode tests -----------------------------------------
def test_format_node_basic_with_execution_and_validation():
    fmt = FormatNode(pretty=True)
    sql = "SELECT * FROM apps"
    execution = {"rows": [{"App": "x", "Rating": 5}], "rowcount": 1}
    raw = {"validation": {"valid": True, "errors": [], "fixes": []}, "formatted": {"output": "done"}}
    out = fmt.run(sql, schemas=None, retrieved=[{"text": "doc text"}], execution=execution, raw=raw)
    assert "sql" in out and out["sql"] == sql
    assert "output" in out and isinstance(out["output"], str)
    assert out.get("meta", {}).get("validation", {}).get("valid") is True or "Validation" in out.get("output", "")


# ----------------- ContextNode tests ----------------------------------------
def test_context_node_resolves_schema_and_samples(monkeypatch):
    # Provide a Tools-like stub with get_schema/get_sample_rows
    class ToolsStub:
        def __init__(self):
            self._called = {}
        def get_schema(self, name):
            # pretend canonicalization: remove path, extension etc.
            n = (name or "").split("/")[-1].split(".")[0]
            if n in ("apps", "users"):
                return ["app", "rating"] if n == "apps" else ["name", "email"]
            return []
        def get_sample_rows(self, name):
            if "apps" in (name or ""):
                return [{"App": "A", "Rating": 4}]
            return []

    tools = ToolsStub()
    node = ContextNode(tools=tools, sample_limit=2)
    out = node.run(["apps.csv", "users"])
    assert "apps" in out
    assert isinstance(out["apps"]["columns"], list)
    assert isinstance(out["apps"]["sample_rows"], list)


# ----------------- ErrorNode tests ------------------------------------------
def test_error_node_normalizes_basic_exception():
    node = ErrorNode()
    try:
        raise ValueError("boom")
    except Exception as exc:
        out = node.run(exc, step="test_step", context={"user_query": "select 1", "timings": {"a": 1}})
        assert out["valid"] is False
        assert out["error"] is not None
        assert "An error occurred" in out["formatted"]["output"]


def test_error_node_handles_non_exception_object():
    node = ErrorNode()
    # pass a non-exception object
    obj = {"code": 500, "msg": "bad"}
    out = node.run(obj, step="weird", context={})
    assert out["valid"] is False
    assert "error" in out
    assert isinstance(out["raw"], dict)
