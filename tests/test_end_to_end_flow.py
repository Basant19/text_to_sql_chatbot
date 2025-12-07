# tests/test_end_to_end_flow.py
import os
import csv
import sqlite3
import tempfile
from typing import Any, Dict, List, Optional
import uuid

import pytest

# project imports
from app.csv_loader import CSVLoader, load_csv_metadata
from app.schema_store import SchemaStore
from app.vector_search import get_vector_search, VectorSearch
from app.tools import Tools
from app.graph.nodes.validate_node import ValidateNode
from app.graph.nodes.format_node import FormatNode
from app.graph.nodes.context_node import ContextNode
from app.graph.nodes.error_node import ErrorNode

import app.graph.nodes.validate_node as validate_node_module



def test_full_end_to_end_flow(tmp_path, monkeypatch):
    # Prepare temporary directories & files
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    faiss_dir = tmp_path / "faiss"
    faiss_dir.mkdir()
    schema_store_path = tmp_path / "schema_store.json"
    index_path = str(faiss_dir / "index.faiss")

    # create CSV
    csv_path = uploads_dir / "apps_sample.csv"
    rows = [
        ["App", "Rating", "Desc"],
        ["Alpha", "4.5", "Alpha is a testing app"],
        ["Beta", "3.8", "Beta does many things"],
        ["Gamma", "4.9", "Gamma is excellent"],
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # register CSV in SchemaStore
    schema_store = SchemaStore(store_path=str(schema_store_path))
    canonical = schema_store.register_from_csv(str(csv_path))
    assert canonical

    # get the inner schemas map (store_key -> meta)
    dump = schema_store.dump()
    schemas_map = dump.get("schemas") if isinstance(dump, dict) else dump
    assert isinstance(schemas_map, dict) and len(schemas_map) > 0
    store_key = next(iter(schemas_map.keys()))

    # VectorSearch (deterministic small dim)
    vs = get_vector_search(index_path=index_path, embedding_fn=None, dim=16)
    assert isinstance(vs, VectorSearch)
    assert vs.list_ids() == []

    # CSVLoader chunk & index
    loader = CSVLoader(upload_dir=str(uploads_dir), chunk_size=50, chunk_overlap=5)
    doc_ids = loader.chunk_and_index(str(csv_path), vector_search_client=vs, id_prefix=canonical)
    assert isinstance(doc_ids, list) and len(doc_ids) > 0
    assert len(vs.list_ids()) >= len(doc_ids)

    # semantic retrieval
    results = vs.search("excellent app", top_k=3)
    assert isinstance(results, list) and len(results) > 0
    for r in results[:1]:
        assert "id" in r and "score" in r and "meta" in r

    # Tools + SQLiteExecutor
    executor = SQLiteExecutor()
    tools = Tools(db=None, schema_store=schema_store, vector_search=vs, executor=executor)

    # load CSV into sqlite using the authoritative store_key
    executor.load_csv_table(str(csv_path), store_key, force_reload=True)

    # ValidateNode — pass the inner schemas_map (not the whole dump)
    monkeypatch.setattr(validate_node_module.utils, "_canonicalize_name", lambda s: (s or "").strip().lower().replace(" ", "_"))
    monkeypatch.setattr(validate_node_module.utils, "is_select_query", lambda sql: (sql or "").strip().lower().startswith("select"))
    node = ValidateNode(safety_rules={"max_row_limit": 100})
    # use exact store_key (quote it for SQLite compatibility)
    sql = f'SELECT App, Rating FROM "{store_key}"'
    # <-- IMPORTANT: pass schemas_map, not schema_store.dump()
    validation = node.run(sql, schemas_map)
    assert validation["valid"] is True
    assert validation["errors"] == []

    # Execute & Format (also pass schemas_map to FormatNode)
    exec_res = executor.execute_sql(sql)
    assert isinstance(exec_res, dict)
    assert exec_res.get("rowcount", 0) == 3
    assert isinstance(exec_res.get("rows"), list)
    assert exec_res["rows"][0].get("App") in {"Alpha", "Beta", "Gamma"}

    fmt = FormatNode(pretty=True)
    formatted = fmt.run(sql, schemas=schemas_map, retrieved=results, execution=exec_res, raw={"validation": validation})
    assert isinstance(formatted, dict)
    assert formatted.get("sql") == sql
    assert "output" in formatted and isinstance(formatted["output"], str)
    assert formatted.get("meta", {}).get("validation", {}).get("valid") in (True, None)

    # ContextNode — pass Tools (it uses tools._schema_store internally)
    ctx = ContextNode(tools=tools, sample_limit=2)
    ctx_out = ctx.run([str(csv_path), store_key, "nonexistent"])
    assert any(k for k in ctx_out.keys() if store_key in k or store_key == k)
    assert "nonexistent" in ctx_out or any(not v["columns"] for k, v in ctx_out.items() if "nonexistent" in k)

    # ErrorNode check
    err_node = ErrorNode()
    try:
        raise ValueError("simulated")
    except Exception as exc:
        err_payload = err_node.run(exc, step="execute", context={"sql": sql})
        assert err_payload["valid"] is False
        assert "formatted" in err_payload and "output" in err_payload["formatted"]

    # cleanup
    vs.clear()
    schema_store.clear()
