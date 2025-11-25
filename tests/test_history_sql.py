# tests/test_history_sql.py
import os
import json
from pathlib import Path
from typing import Any

import pytest

from app.history_sql import HistoryStore, _safe_json_dumps


def _normalize_result_field(val: Any):
    """
    Helper to normalize 'result' field returned by HistoryStore implementations.
    Accept either dict or JSON string form; return parsed dict if possible.
    """
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return val
    return val


def test_safe_json_dumps_handles_non_serializable():
    class Foo:
        def __str__(self):
            return "FOO"

    v = {"x": Foo()}
    dumped = _safe_json_dumps(v)
    parsed = json.loads(dumped)
    assert "x" in parsed


def test_json_backend_crud_and_import_export(tmp_path: Path):
    data_dir = tmp_path / "data_json"
    data_dir.mkdir()
    file_path = str(data_dir / "history_test.json")

    store = HistoryStore(backend="json", path=file_path)

    # start empty
    assert isinstance(store.list_entries(), list)
    assert len(store.list_entries()) == 0

    # add entry
    e1 = store.add_entry(name="First", query="What is X?", result={"sql": "SELECT 1"})
    assert "id" in e1
    assert e1["name"] == "First"
    assert "created_at" in e1 and "updated_at" in e1

    # list/get
    entries = store.list_entries()
    assert len(entries) == 1
    got = store.get_entry(e1["id"])
    assert got is not None
    assert _normalize_result_field(got.get("result")) == {"sql": "SELECT 1"}

    # update name only
    old_updated = got.get("updated_at")
    updated = store.update_entry(e1["id"], name="Renamed")
    assert updated is not None
    assert updated["name"] == "Renamed"
    assert updated["updated_at"] != old_updated

    # update result
    updated2 = store.update_entry(e1["id"], result={"sql": "SELECT 2"})
    assert updated2 is not None
    assert _normalize_result_field(updated2.get("result")) == {"sql": "SELECT 2"}

    # delete
    assert store.delete_entry(e1["id"]) is True
    assert store.get_entry(e1["id"]) is None
    assert store.list_entries() == []

    # add two and export/import
    e2 = store.add_entry(name="A", query="q1", result={"r": 1})
    e3 = store.add_entry(name="B", query="q2", result={"r": 2})
    exported = store.export_json(export_path=str(data_dir / "export.json"))
    assert os.path.exists(exported)

    store.clear()
    assert store.list_entries() == []
    count = store.import_json(exported, keep_ids=True)
    assert count >= 2
    items = store.list_entries(newest_first=False)
    assert len(items) >= 2
    store.clear()


def test_sqlite_backend_crud_and_import_export(tmp_path: Path):
    data_dir = tmp_path / "data_sqlite"
    data_dir.mkdir()
    db_path = str(data_dir / "history_test.db")

    store = HistoryStore(backend="sqlite", path=db_path)
    assert os.path.exists(db_path)

    assert store.list_entries() == []

    e1 = store.add_entry("FirstSQL", "How many?", {"sql": "SELECT COUNT(*) FROM t"})
    assert "id" in e1

    items = store.list_entries()
    assert isinstance(items, list)
    assert len(items) >= 1

    got = store.get_entry(e1["id"])
    assert got is not None
    assert _normalize_result_field(got.get("result")) == {"sql": "SELECT COUNT(*) FROM t"}

    u = store.update_entry(e1["id"], name="FirstSQL-R", query="How many rows?")
    assert u is not None
    assert u["name"] == "FirstSQL-R"
    assert u["query"] == "How many rows?"
    assert "updated_at" in u

    u2 = store.update_entry(e1["id"], result={"sql": "SELECT 42"})
    assert u2 is not None
    assert _normalize_result_field(u2.get("result")) == {"sql": "SELECT 42"}

    assert store.delete_entry(e1["id"]) is True
    assert store.get_entry(e1["id"]) is None

    e2 = store.add_entry("X", "q", {"a": 1})
    exported = store.export_json(export_path=str(data_dir / "export_sql.json"))
    assert os.path.exists(exported)

    store.clear()
    assert store.list_entries() == []
    imported = store.import_json(exported, keep_ids=False)
    assert imported >= 1
    items_after = store.list_entries()
    assert len(items_after) >= 1
    store.clear()


def test_update_nonexistent_returns_none(tmp_path: Path):
    d = tmp_path / "data_update"
    d.mkdir()
    fp = str(d / "h.json")
    store = HistoryStore(backend="json", path=fp)
    assert store.update_entry("nonexistent-id", name="X") is None

    dbp = str(d / "h.db")
    store_sql = HistoryStore(backend="sqlite", path=dbp)
    assert store_sql.update_entry("nonexistent-id", name="X") is None
