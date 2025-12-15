import pytest
from app.sql_executor import _rewrite_sql_table_identifiers


def test_numeric_prefix_identifier_rewrite():
    sql = "SELECT app FROM 7c7782caaa9858b43e9969538c96f604_googleplaystore"
    table_map = {
        "7c7782caaa9858b43e9969538c96f604_googleplaystore": "dummy.csv"
    }

    rewritten = _rewrite_sql_table_identifiers(sql, table_map)

    assert "t_7c7782caaa9858b43e9969538c96f604_googleplaystore" in rewritten
    assert "FROM 7c7782caaa9858b43e9969538c96f604_googleplaystore" not in rewritten


def test_backtick_numeric_identifier_rewrite():
    sql = "SELECT app FROM `7table-name`"
    table_map = {"7table-name": "dummy.csv"}

    rewritten = _rewrite_sql_table_identifiers(sql, table_map)

    assert "t_7table_name" in rewritten


def test_double_quoted_numeric_identifier_rewrite():
    sql = 'SELECT app FROM "9abc"'
    table_map = {"9abc": "dummy.csv"}

    rewritten = _rewrite_sql_table_identifiers(sql, table_map)

    assert "t_9abc" in rewritten


def test_no_rewrite_when_table_not_present():
    sql = "SELECT * FROM users"
    rewritten = _rewrite_sql_table_identifiers(sql, {})
    assert rewritten == sql
