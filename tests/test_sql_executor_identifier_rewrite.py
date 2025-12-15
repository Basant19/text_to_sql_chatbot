# tests/test_sql_executor_identifier_rewrite.py
import os
import csv
import tempfile

import pytest

# Project imports
from app import config
from app import database
from app.sql_executor import execute_sql


SAMPLE_ROWS = [
    ["app", "rating"],
    ["Alpha", "4.5"],
    ["Beta", "3.8"],
]


def _write_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(SAMPLE_ROWS)


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    """
    Ensure each test uses its own DuckDB file and that connection is closed/cleaned up.
    """
    db_file = str(tmp_path / "text_to_sql.db")
    monkeypatch.setattr(config, "DATABASE_PATH", db_file, raising=False)
    # Ensure any existing module-level connection closed before test
    try:
        database.close_connection()
    except Exception:
        pass
    yield
    # Close connection after test to allow file cleanup on Windows
    try:
        database.close_connection()
    except Exception:
        pass


def test_unquoted_identifier_rewrite_for_digit_start(tmp_path):
    """
    Table key starts with a digit (e.g. "7apps") — executor should sanitize,
    load CSV and rewrite SQL referencing the original canonical name.
    """
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    csv_path = str(uploads / "apps.csv")
    _write_csv(csv_path)

    # Use a canonical that starts with a digit -> requires sanitized name in DuckDB
    canonical = "7apps"  # invalid bare identifier for SQL parser
    table_map = {canonical: csv_path}

    # Run a simple COUNT(*) referencing the canonical name unquoted
    res = execute_sql(f"SELECT COUNT(*) FROM {canonical}", table_map=table_map, read_only=True)
    assert isinstance(res, dict)
    # COUNT returns one row
    assert res["meta"]["rowcount"] == 1
    # The count value should be 2 (two data rows)
    cnt_val = list(res["rows"][0].values())[0]
    assert int(cnt_val) == 2


def test_double_quoted_identifier_rewrite(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    csv_path = str(uploads / "apps2.csv")
    _write_csv(csv_path)

    canonical = "7apps"  # same problematic canonical
    table_map = {canonical: csv_path}

    # Use double-quoted identifier in SQL (as LLM might)
    res = execute_sql(f'SELECT COUNT(*) FROM "{canonical}"', table_map=table_map, read_only=True)
    assert res["meta"]["rowcount"] == 1
    cnt_val = list(res["rows"][0].values())[0]
    assert int(cnt_val) == 2


def test_backtick_quoted_identifier_rewrite(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    csv_path = str(uploads / "apps3.csv")
    _write_csv(csv_path)

    canonical = "7apps"
    table_map = {canonical: csv_path}

    # Use backtick quoting (some generators output backticks)
    res = execute_sql(f"SELECT COUNT(*) FROM `{canonical}`", table_map=table_map, read_only=True)
    assert res["meta"]["rowcount"] == 1
    cnt_val = list(res["rows"][0].values())[0]
    assert int(cnt_val) == 2


def test_hyphenated_canonical_name_rewrite(tmp_path):
    """
    Canonical contains a hyphen (e.g. 'my-apps') — executor should sanitize to my_apps
    and rewrite SQL referencing original name (quoted/unquoted).
    """
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    csv_path = str(uploads / "myapps.csv")
    _write_csv(csv_path)

    canonical = "my-apps"
    table_map = {canonical: csv_path}

    # unquoted
    res1 = execute_sql(f"SELECT COUNT(*) FROM {canonical}", table_map=table_map, read_only=True)
    assert res1["meta"]["rowcount"] == 1
    assert int(list(res1["rows"][0].values())[0]) == 2

    # double-quoted
    res2 = execute_sql(f'SELECT COUNT(*) FROM "{canonical}"', table_map=table_map, read_only=True)
    assert res2["meta"]["rowcount"] == 1
    assert int(list(res2["rows"][0].values())[0]) == 2

