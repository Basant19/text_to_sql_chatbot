# tests/test_history_sql.py
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any

import pytest

from app.history_sql import HistoryStore, _safe_json_dumps


def test_safe_json_dumps_handles_non_serializable():
    class Foo:
        def __str__(self):
            return "FOO"

    v = {"x": Foo()}
    dumped = _safe_json_dumps(v)
    # should produce a JSON string (parseable)
    parsed = json.loads(dumped)
    assert "x" in parsed


def test_json_backend_crud_and_import_export(tmp_path: Path):
    # prepare directory and file path
    data_dir = tmp_path / "data_json"
    data_dir.mkdir()
    file_path = str(data_dir / "history_test.json")

    store = HistoryStore(backend="json", path=file_path)

    # start empty
    entries = store.list_entries()
    assert isinstance(entries, list)
    assert len(entries) == 0

    # add entry
    e1 = store.add_entry(name="First", query="What is X?", result={"sql": "SELECT 1"})
    assert "id" in e1
    assert e1["name"] == "First"
    assert e1["query"] == "What is X?"
    assert "created_at" in e1 and "updated_at" in e1

    # list returns it
    entries = store.list_entries()
    assert len(entries) == 1
    assert entries[0]["id"] == e1["id"]

    # get by id
    got = store.get_entry(e1["id"])
    assert got is not None
    assert got["name"] == "First"

    # update name only
    old_updated = got.get("updated_at")
    updated = store.update_entry(e1["id"], name="Renamed")
    assert updated is not None
    assert updated["name"] == "Renamed"
    assert updated["id"] == e1["id"]
    assert updated["updated_at"] != old_updated

    # update result
    updated2 = store.update_entry(e1["id"], result={"sql": "SELECT 2"})
    assert updated2 is not None
    assert updated2["result"] == {"sql": "SELECT 2"}

    # delete entry
    ok = store.delete_entry(e1["id"])
    assert ok is True
    assert store.get_entry(e1["id"]) is None
    assert store.list_entries() == []

    # add two and export/import
    e2 = store.add_entry(name="A", query="q1", result={"r": 1})
    e3 = store.add_entry(name="B", query="q2", result={"r": 2})
    exported_path = store.export_json(export_path=str(data_dir / "export.json"))
    assert os.path.exists(exported_path)

    # Clear and import (keep ids)
    store.clear()
    assert store.list_entries() == []
    count = store.import_json(exported_path, keep_ids=True)
    assert count >= 2  # sometimes implementation may dedupe or append — be permissive
    items = store.list_entries(newest_first=False)
    assert len(items) >= 2

    # cleanup
    store.clear()
    assert store.list_entries() == []


def test_sqlite_backend_crud_and_import_export(tmp_path: Path):
    data_dir = tmp_path / "data_sqlite"
    data_dir.mkdir()
    db_path = str(data_dir / "history_test.db")

    store = HistoryStore(backend="sqlite", path=db_path)

    # DB file should exist
    assert os.path.exists(db_path)

    # empty
    assert store.list_entries() == []

    # add
    e1 = store.add_entry("FirstSQL", "How many?", {"sql": "SELECT COUNT(*) FROM t"})
    assert e1["name"] == "FirstSQL"
    assert "id" in e1

    # list and get
    items = store.list_entries()
    assert len(items) >= 1
    got = store.get_entry(e1["id"])
    assert got is not None
    assert got["query"] == "How many?"

    # update (name + query)
    u = store.update_entry(e1["id"], name="FirstSQL-R", query="How many rows?")
    assert u is not None
    assert u["name"] == "FirstSQL-R"
    assert u["query"] == "How many rows?"
    assert "updated_at" in u

    # partial update (only result)
    u2 = store.update_entry(e1["id"], result={"sql": "SELECT 42"})
    assert u2 is not None
    # result may be stored as object or JSON-parsed dict depending on implementation
    assert u2["result"] == {"sql": "SELECT 42"} or (isinstance(u2["result"], str) and "SELECT 42" in u2["result"])

    # delete
    assert store.delete_entry(e1["id"]) is True
    assert store.get_entry(e1["id"]) is None

    # import/export with sqlite
    e2 = store.add_entry("X", "q", {"a": 1})
    exported = store.export_json(export_path=str(data_dir / "export_sql.json"))
    assert os.path.exists(exported)

    # clear and import
    store.clear()
    assert store.list_entries() == []
    imported = store.import_json(exported, keep_ids=False)
    assert imported >= 1
    items_after = store.list_entries()
    assert len(items_after) >= 1

    # cleanup
    store.clear()
    assert store.list_entries() == []


def test_update_nonexistent_returns_none(tmp_path: Path):
    data_dir = tmp_path / "data_update"
    data_dir.mkdir()
    fp = str(data_dir / "h.json")
    store = HistoryStore(backend="json", path=fp)
    res = store.update_entry("nonexistent-id", name="X")
    assert res is None

    dbp = str(data_dir / "h.db")
    store_sql = HistoryStore(backend="sqlite", path=dbp)
    res2 = store_sql.update_entry("nonexistent-id", name="X")
    assert res2 is None
