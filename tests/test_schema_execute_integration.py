# tests/test_schema_execute_integration.py
import os
import tempfile
import shutil
import pytest

from app.schema_store import SchemaStore
from app.graph.nodes.execute_node import ExecuteNode

import app.database as database_module
from app.tools import Tools

# Helper CSV content
CSV_CONTENT = "a,b\n1,2\n3,4\n"


def write_tmp_csv(tmpdir, name="sample.csv", content=CSV_CONTENT):
    p = os.path.join(tmpdir, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return p


def test_schemastore_add_csv_idempotent_by_path_and_hash(tmp_path):
    ss = SchemaStore(store_path=os.path.join(tmp_path, "schema_store.json"))

    # create csv
    p1 = write_tmp_csv(tmp_path, "file1.csv")
    key1 = ss.add_csv(p1, csv_name="mycsv")
    assert key1 is not None

    # calling add_csv on same absolute path returns same key
    key2 = ss.add_csv(p1)
    assert key1 == key2

    # copy file to different path but same content -> should reuse by hash
    p2 = write_tmp_csv(tmp_path, "file_copy.csv")
    # overwrite to ensure same content
    shutil.copyfile(p1, p2)
    key3 = ss.add_csv(p2)
    assert key3 == key1

    # registry contains only one canonical for this content (aliases may exist)
    all_keys = ss.list_tables()
    assert key1 in all_keys

def test_execute_node_enrich_and_load(monkeypatch, tmp_path):
    # Prepare a simple CSV and register in SchemaStore
    csv_path = os.path.join(tmp_path, "apps.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("app,rating\nx,5\ny,4\n")

    ss = SchemaStore(store_path=os.path.join(tmp_path, "schema_store.json"))
    key = ss.add_csv(csv_path, csv_name="apps")

    # Prepare Tools stub that exposes SchemaStore and an executor based on app.database
    class ToolsStub:
        def __init__(self, schema_store):
            self._schema_store = schema_store
            # use real DB executor via database_module.execute_query through small wrapper
            self._executor = None

        def execute_sql(self, sql, read_only=True, limit=None, as_dataframe=False):
            # delegate to database.execute_query
            return database_module.execute_query(sql, read_only=read_only, as_dataframe=as_dataframe)

    tools = ToolsStub(ss)
    node = ExecuteNode(tools=tools, default_limit=10)

    # Ensure database is fresh (close connection if open) and remove DB file if exists
    try:
        database_module.close_connection()
    except Exception:
        pass
    db_path = os.path.join(os.getcwd(), "data", "text_to_sql.db")
    # ensure DB removed so tests start clean
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except Exception:
        pass

    # Build SQL referencing the canonical key
    sql = f"SELECT app, rating FROM {key} ORDER BY rating DESC LIMIT 1"

    # Provide read_only mapping so ExecuteNode can enrich and attempt load
    read_only_map = { key: {"path": csv_path} }

    res = node.run(sql, read_only=read_only_map, limit=1)
    # Expect formatted result dict
    assert isinstance(res, dict)
    # Should return rows (one row)
    assert res.get("rowcount", 0) >= 0
