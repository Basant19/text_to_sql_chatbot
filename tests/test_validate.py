# tests/test_validate_node.py
import pytest
from app.graph.nodes.validate_node import ValidateNode

# Dummy schema for testing
dummy_schemas = {
    "users": {"columns": ["id", "name"]},
    "orders": {"columns": ["id", "amount"]}
}

def test_validate_node_select_valid():
    node = ValidateNode()
    sql = "SELECT id, name FROM users;"
    res = node.run(sql, dummy_schemas)
    assert res["valid"] is True
    assert res["errors"] == []
    assert res["sql"] == sql

def test_validate_node_non_select():
    node = ValidateNode()
    sql = "DROP TABLE users;"
    res = node.run(sql, dummy_schemas)
    assert res["valid"] is False
    assert "Only SELECT statements are allowed." in res["errors"]

def test_validate_node_missing_table():
    node = ValidateNode()
    sql = "SELECT * FROM customers;"
    res = node.run(sql, dummy_schemas)
    assert res["valid"] is False
    assert "Tables not found in schema" in res["errors"][0]

def test_validate_node_empty_sql():
    node = ValidateNode()
    sql = "   "
    res = node.run(sql, dummy_schemas)
    assert res["valid"] is False
    assert "SQL query is empty." in res["errors"]
