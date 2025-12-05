# tests/test_validate_node.py
import os
import copy
import pytest
from app.graph.nodes.validate_node import ValidateNode
import app.graph.nodes.validate_node as validate_node_module

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

def test_valid_select_passes():
    schemas = make_sample_schemas()
    node = ValidateNode()
    sql = "SELECT App, Rating FROM apps"
    out = node.run(sql, schemas)
    assert out["valid"] is True
    assert out["errors"] == []
    assert "Applied automatic" not in " ".join(out["fixes"])

def test_missing_table_detected():
    schemas = make_sample_schemas()
    node = ValidateNode()
    sql = "SELECT foo FROM unknown_table"
    out = node.run(sql, schemas)
    assert out["valid"] is False
    assert any("Tables not found" in e for e in out["errors"])

def test_missing_column_detected():
    schemas = make_sample_schemas()
    node = ValidateNode()
    sql = "SELECT not_a_col FROM apps"
    out = node.run(sql, schemas)
    assert out["valid"] is False
    assert any("Columns not found" in e for e in out["errors"])

def test_forbidden_table():
    schemas = make_sample_schemas()
    node = ValidateNode(safety_rules={"forbidden_tables": ["users"]})
    sql = "SELECT name FROM users"
    out = node.run(sql, schemas)
    assert out["valid"] is False
    assert any("forbidden tables" in e.lower() for e in out["errors"])

def test_row_limit_enforced(monkeypatch):
    schemas = make_sample_schemas()
    # monkeypatch exceeds_row_limit to return True and limit_sql_rows to add LIMIT 1
    monkeypatch.setattr(validate_node_module.utils, "exceeds_row_limit", lambda sql, limit: True)
    monkeypatch.setattr(validate_node_module.utils, "limit_sql_rows", lambda sql, limit: sql.rstrip(";") + f" LIMIT {limit}")
    node = ValidateNode(safety_rules={"max_row_limit": 1})
    sql = "SELECT App FROM apps"
    out = node.run(sql, schemas)

    # Build safe strings to check for LIMIT presence
    sql_str = out.get("sql") or ""
    suggested_str = out.get("suggested_sql") or ""
    fixes_join = " ".join(out.get("fixes") or [])

    assert ("LIMIT 1" in sql_str) or ("LIMIT 1" in suggested_str) or ("LIMIT 1" in fixes_join)
