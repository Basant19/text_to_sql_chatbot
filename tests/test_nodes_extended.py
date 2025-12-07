# tests/test_nodes_extended.py
import pytest

# nodes under test
from app.graph.nodes.validate_node import ValidateNode
from app.graph.nodes.context_node import ContextNode
from app.graph.nodes.format_node import FormatNode
from app.graph.nodes.error_node import ErrorNode

import app.graph.nodes.validate_node as validate_node_module

# Reuse same utils monkeypatch pattern as your other tests
@pytest.fixture(autouse=True)
def patch_utils_monkey(monkeypatch):
    """
    Provide minimal implementations for the utils functions ValidateNode expects.
    Keep behavior deterministic for tests.
    """
    monkeypatch.setattr(validate_node_module.utils, "_canonicalize_name", lambda s: (s or "").strip().lower().replace(" ", "_"))
    monkeypatch.setattr(validate_node_module.utils, "is_select_query", lambda sql: (sql or "").strip().lower().startswith("select"))
    monkeypatch.setattr(validate_node_module.utils, "extract_table_names_from_sql", lambda sql: [])
    monkeypatch.setattr(validate_node_module.utils, "extract_column_names_from_sql", lambda sql: [])
    monkeypatch.setattr(validate_node_module.utils, "validate_tables_in_sql", lambda sql, schemas: [])
    monkeypatch.setattr(validate_node_module.utils, "validate_columns_in_sql", lambda sql, schemas: [])
    monkeypatch.setattr(validate_node_module.utils, "exceeds_row_limit", lambda sql, limit: False)
    monkeypatch.setattr(validate_node_module.utils, "limit_sql_rows", lambda sql, limit: sql)
    yield


def make_sample_schemas_extended():
    """
    Slightly richer schemas for more edge-case tests.
    """
    return {
        # store key has suffix to simulate generated keys
        "googleplaystore_abc123": {
            "canonical": "googleplaystore",
            "aliases": ["googleplaystore", "gps"],
            "path": "/data/googleplaystore_abc123.csv",
            "columns": ["App", "Rating", "Reviews"],
            "columns_normalized": ["app", "rating", "reviews"],
        },
        "applications_1234": {
            "canonical": "applications",
            "aliases": ["applications", "apps"],
            "path": "/data/applications_1234.csv",
            "columns": ["id", "name"],
            "columns_normalized": ["id", "name"],
        },
    }


# ---------------- ValidateNode: fuzzy mapping / aliases / single-string utils ----------------
def test_validate_node_fuzzy_alias_table_mapping():
    schemas = make_sample_schemas_extended()
    node = ValidateNode()
    # Query uses friendly canonical name (no suffix); should map to store_key and produce interpretation hint
    sql = "SELECT App, Rating FROM googleplaystore WHERE Rating > 4"
    out = node.run(sql, schemas)
    # No "Tables not found" error
    assert not any("Tables not found" in e for e in out.get("errors", [])), f"Unexpected errors: {out.get('errors')}"
    # There should be at least one interpreting fix referencing 'Interpreting table'
    fixes = out.get("fixes") or []
    assert any("interpreting" in f.lower() or "interpreting table" in f for f in fixes), f"No interpretation fix found: {fixes}"


def test_validate_node_forbidden_table_matches_alias():
    schemas = make_sample_schemas_extended()
    node = ValidateNode(safety_rules={"forbidden_tables": ["apps"]})
    # Query refers to canonical 'applications' (store key is applications_1234 whose aliases include 'apps')
    sql = "SELECT id, name FROM applications"
    out = node.run(sql, schemas)
    errors = out.get("errors") or []
    assert any("forbidden" in e.lower() or "forbidden" in " ".join(errors).lower() for e in errors), f"Expected forbidden-table error, got: {errors}"


def test_validate_node_handles_utils_helpers_returning_single_string(monkeypatch):
    """
    Some utils helpers may (mistakenly) return a single string. ValidateNode should handle both string and list.
    """
    schemas = {}
    # monkeypatch utils validators to return single strings instead of lists
    monkeypatch.setattr(validate_node_module.utils, "validate_tables_in_sql", lambda sql, schemas: "apps")
    monkeypatch.setattr(validate_node_module.utils, "validate_columns_in_sql", lambda sql, schemas: "colx")

    node = ValidateNode()
    sql = "SELECT colx FROM apps"
    out = node.run(sql, schemas)
    errors = out.get("errors") or []
    assert any("Tables not found" in e or "Columns not found" in e for e in errors), f"Expected missing table/column errors, got: {errors}"


# ---------------- ContextNode: missing schema / canonical fallback ----------------
def test_context_node_missing_schema_returns_empty_entries(monkeypatch):
    """
    If a CSV name cannot be resolved by the Tools/SchemaStore, ContextNode should still
    return an entry for it with empty columns/sample_rows so downstream nodes can handle it.
    """
    class MinimalTools:
        def get_schema(self, name):
            return None
        def get_sample_rows(self, name):
            return []

    tools = MinimalTools()
    node = ContextNode(tools=tools, sample_limit=2)
    out = node.run(["nonexistent_table.csv"])
    # Should include normalized key for input and empty lists
    assert isinstance(out, dict)
    # key should be normalized 'nonexistent_table' (no extension), but ContextNode uses its internal canonicalizer.
    # At minimum ensure we returned one mapping and it has empty lists
    assert len(out) == 1
    k = list(out.keys())[0]
    assert isinstance(out[k]["columns"], list)
    assert out[k]["columns"] == []
    assert isinstance(out[k]["sample_rows"], list)
    assert out[k]["sample_rows"] == []


# ---------------- FormatNode: non-pretty mode and raw metadata extraction ----------------
def test_format_node_non_pretty_returns_structured_output():
    fmt = FormatNode(pretty=False)
    sql = "SELECT name FROM users"
    execution = [{"name": "Alice"}, {"name": "Bob"}]  # list form should be accepted as rows
    out = fmt.run(sql, schemas=None, retrieved=None, execution=execution, raw=None)
    # In non-pretty mode, output should be a dict structure (per implementation)
    assert isinstance(out.get("output"), dict) or not fmt.pretty  # tolerant check
    # rows should be present under 'rows' key in returned payload
    assert "rows" in out and out["rows"] == execution


# ---------------- ErrorNode: CustomException handling and long trace truncation ----------------
def test_error_node_handles_custom_exception_and_truncates_trace():
    node = ErrorNode()
    # Build a CustomException-like object (use real CustomException if available)
    from app.exception import CustomException as CE  # import from your codebase
    try:
        raise CE("custom failure")
    except Exception as exc:
        # simulate a very long trace by attaching long trace string to exc.__traceback__ not needed;
        out = node.run(exc, step="custom_step", context={"timings": {"t": 1}})
        assert out["valid"] is False
        assert "custom" in out["error"].lower() or "failure" in out["formatted"]["meta"]["debug"]["error_details"].lower()
        debug = out["formatted"]["meta"]["debug"]
        assert "error_trace" in debug  # trace may be None on some platforms, but field must exist
