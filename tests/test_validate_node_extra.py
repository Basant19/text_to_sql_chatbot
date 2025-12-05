# tests/test_validate_node_extra.py
import pytest

# Import module & class under test
import app.graph.nodes.validate_node as validate_node_module
from app.graph.nodes.validate_node import ValidateNode


def make_sample_schemas():
    """
    Create sample schemas similar to what SchemaStore would provide.
    Keys are store keys (unique identifiers saved in the store).
    Each entry contains canonical name, aliases, columns and path.
    """
    return {
        # store key contains suffix to simulate generated key in store
        "googleplaystore_abc123": {
            "canonical": "googleplaystore",
            "aliases": ["googleplaystore", "gps"],
            "columns": ["App", "Rating", "Reviews"],
            "columns_normalized": ["app", "rating", "reviews"],
            "path": "/data/googleplaystore_abc123.csv",
        },
        "applications_1234": {
            "canonical": "applications",
            "aliases": ["applications", "apps"],
            "columns": ["id", "name"],
            "columns_normalized": ["id", "name"],
            "path": "/data/applications_1234.csv",
        },
    }


def test_fuzzy_alias_table_mapping():
    """
    Query references the friendly/canonical table name (without the generated suffix).
    ValidateNode should map it to the store key and provide a helpful 'interpreting' fix.
    """
    schemas = make_sample_schemas()
    node = ValidateNode()

    # Query uses "googleplaystore" (canonical friendly name) while store key is googleplaystore_abc123
    sql = "SELECT App, Rating FROM googleplaystore WHERE Rating > 4"
    out = node.run(sql, schemas)

    # There should be no "Tables not found" error because the node should map canonical -> store key
    assert not any("Tables not found" in e for e in out.get("errors", [])), f"Unexpected errors: {out.get('errors')}"

    # Node should have appended at least one interpreting fix (makes mapping explicit)
    fixes = out.get("fixes") or []
    assert any("Interpreting table" in f or "interpreting table" in f.lower() for f in fixes), f"No interpretation fix found: {fixes}"


def test_forbidden_table_matches_alias():
    """
    If a forbidden table is listed (by alias), and the query references an alias,
    the validator should detect it.
    """
    schemas = make_sample_schemas()

    # mark 'apps' as forbidden; the stored canonical for applications_1234 includes alias 'apps'
    node = ValidateNode(safety_rules={"forbidden_tables": ["apps"]})

    sql = "SELECT id, name FROM applications"
    out = node.run(sql, schemas)

    # Should report forbidden table usage in errors
    errors = out.get("errors") or []
    assert any("forbidden" in e.lower() or "forbidden" in " ".join(errors).lower() for e in errors), f"Expected forbidden-table error, got: {errors}"


def test_row_limit_enforced(monkeypatch):
    """
    Ensure max_row_limit enforcement uses utils.exceeds_row_limit / limit_sql_rows and
    that the returned SQL or suggested_sql contains the LIMIT applied.
    """
    schemas = make_sample_schemas()

    # Force utils.exceeds_row_limit to return True and limit_sql_rows to append LIMIT 1
    monkeypatch.setattr(validate_node_module.utils, "exceeds_row_limit", lambda sql, limit: True)
    monkeypatch.setattr(validate_node_module.utils, "limit_sql_rows", lambda sql, limit: sql.rstrip(";") + f" LIMIT {limit}")

    node = ValidateNode(safety_rules={"max_row_limit": 1})
    sql = "SELECT App FROM googleplaystore"
    out = node.run(sql, schemas)

    # The node should have applied the limit either by modifying sql or in suggested_sql, and added a fix message.
    applied_in_sql = "LIMIT 1" in (out.get("sql") or "")
    applied_in_suggested = "LIMIT 1" in (out.get("suggested_sql") or "")
    applied_in_fixes = any("LIMIT 1" in f for f in (out.get("fixes") or []))

    assert applied_in_sql or applied_in_suggested or applied_in_fixes, f"LIMIT not applied; output: {out}"


def test_utils_helpers_return_single_string(monkeypatch):
    """
    Some utils helpers may (mistakenly) return a single string instead of a list.
    ValidateNode should handle that gracefully (not crash) and produce appropriate errors.
    """
    # empty schemas so ValidateNode will try to use utils.validate_tables_in_sql / validate_columns_in_sql
    schemas = {}

    # Monkeypatch utils validators to return single string values
    monkeypatch.setattr(validate_node_module.utils, "validate_tables_in_sql", lambda sql, schemas: "apps")
    monkeypatch.setattr(validate_node_module.utils, "validate_columns_in_sql", lambda sql, schemas: "colx")

    node = ValidateNode()
    sql = "SELECT colx FROM apps"
    out = node.run(sql, schemas)

    # Should have errors mentioning missing tables or columns (handled as strings)
    errors = out.get("errors") or []
    assert any("Tables not found" in e or "Columns not found" in e for e in errors), f"Expected missing table/column errors, got: {errors}"
