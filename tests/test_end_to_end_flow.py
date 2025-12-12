# tests/test_end_to_end_flow.py
import os
import csv
import sqlite3
from typing import Any, Dict, List, Optional

import pytest

# project imports
from app.csv_loader import CSVLoader
from app.schema_store import SchemaStore
from app.vector_search import get_vector_search, VectorSearch
from app.tools import Tools
from app.graph.nodes.validate_node import ValidateNode
from app.graph.nodes.format_node import FormatNode
from app.graph.nodes.context_node import ContextNode
from app.graph.nodes.error_node import ErrorNode

import app.graph.nodes.validate_node as validate_node_module


# -------------------------
# SQLiteExecutor for tests
# -------------------------
class SQLiteExecutor:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or ":memory:"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def load_csv_table(self, csv_path: str, table_name: str, force_reload: bool = False) -> None:
        if force_reload:
            with self.conn:
                self.conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []

            cols = [c.strip().replace('"', '""') or f"col{i}" for i, c in enumerate(header, start=1)]
            col_defs = ", ".join(f'"{c}" TEXT' for c in cols)

            with self.conn:
                self.conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})')

                col_names = ", ".join(f'"{c}"' for c in cols)
                placeholders = ", ".join("?" for _ in cols)
                insert_sql = f'INSERT INTO "{table_name}" ({col_names}) VALUES ({placeholders})'

                batch = []
                for row in reader:
                    row_norm = list(row) + [None] * (len(cols) - len(row))
                    row_norm = row_norm[: len(cols)]
                    batch.append(tuple(str(x) if x is not None else None for x in row_norm))
                if batch:
                    self.conn.executemany(insert_sql, batch)

    def execute_sql(self, sql: str, read_only: bool = True, limit: Optional[int] = None, as_dataframe: bool = False) -> Dict[str, Any]:
        q = sql.strip()
        if limit and "limit" not in q.lower():
            q = q.rstrip(";") + f" LIMIT {limit}"
        cur = self.conn.cursor()
        try:
            cur.execute(q)
            rows = [dict(r) for r in cur.fetchall()]
            cols = [c[0] for c in cur.description] if cur.description else []
            return {"rows": rows, "columns": cols, "rowcount": len(rows)}
        finally:
            cur.close()


# -------------------------
# Extended End-to-End Test
# -------------------------
def test_extended_end_to_end_flow(tmp_path, monkeypatch):
    """Extended E2E test covering CSV upload, schema, vector search, SQL, chunking, and error handling."""

    # Temp directories and files
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    faiss_dir = tmp_path / "faiss"
    faiss_dir.mkdir()
    schema_store_path = tmp_path / "schema_store.json"
    index_path = str(faiss_dir / "index.faiss")

    # --- 1) Create multiple CSVs ---
    csv1 = uploads_dir / "apps1.csv"
    csv2 = uploads_dir / "apps2.csv"

    rows1 = [["App", "Rating", "Desc"],
             ["Alpha", "4.5", "Alpha app"],
             ["Beta", "3.8", "Beta app"]]
    rows2 = [["App", "Rating", "Desc"],
             ["Gamma", "4.9", "Gamma app"],
             ["Delta", "4.2", "Delta app"]]

    for path, rows in [(csv1, rows1), (csv2, rows2)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    # --- 2) SchemaStore ---
    schema_store = SchemaStore(store_path=str(schema_store_path))
    canonical1 = schema_store.register_from_csv(str(csv1))
    canonical2 = schema_store.register_from_csv(str(csv2))

    dump = schema_store.dump()
    schemas_map = dump.get("schemas") if isinstance(dump, dict) else dump
    store_keys = list(schemas_map.keys())
    assert len(store_keys) == 2

    # --- 3) VectorSearch Singleton / Dim check ---
    vs = get_vector_search(index_path=index_path, embedding_fn=None, dim=16)
    vs2 = get_vector_search(index_path=index_path, embedding_fn=None, dim=16)
    # Ensure singleton: both instances point to same object
    assert vs is vs2

    # Chunk & index CSVs
    loader = CSVLoader(upload_dir=str(uploads_dir), chunk_size=50, chunk_overlap=5)
    doc_ids1 = loader.chunk_and_index(str(csv1), vector_search_client=vs, id_prefix=canonical1)
    doc_ids2 = loader.chunk_and_index(str(csv2), vector_search_client=vs, id_prefix=canonical2)
    assert len(vs.list_ids()) >= len(doc_ids1) + len(doc_ids2)

    # Semantic search returns all documents
    results = vs.search("Alpha app", top_k=5)
    assert any("Alpha" in r["meta"].get("text", "") for r in results)

    # --- 4) SQLiteExecutor ---
    executor = SQLiteExecutor()
    for csv_path, key in [(csv1, store_keys[0]), (csv2, store_keys[1])]:
        executor.load_csv_table(str(csv_path), key, force_reload=True)

    sql = f'SELECT App, Rating FROM "{store_keys[0]}"'
    res = executor.execute_sql(sql)
    assert res["rowcount"] == 2
    assert all(r["App"] in ["Alpha", "Beta"] for r in res["rows"])

    # --- 5) ValidateNode & monkeypatch ---
    monkeypatch.setattr(validate_node_module.utils, "_canonicalize_name",
                        lambda s: (s or "").strip().lower().replace(" ", "_"))
    monkeypatch.setattr(validate_node_module.utils, "is_select_query",
                        lambda sql: (sql or "").strip().lower().startswith("select"))

    node = ValidateNode(safety_rules={"max_row_limit": 10})
    validation = node.run(sql, schemas_map)
    assert validation["valid"] is True

    # --- 6) FormatNode ---
    fmt = FormatNode(pretty=True)
    formatted = fmt.run(sql, schemas=schemas_map, retrieved=results, execution=res, raw={"validation": validation})
    assert "output" in formatted

    # --- 7) ContextNode ---
    tools = Tools(db=None, schema_store=schema_store, vector_search=vs, executor=executor)
    ctx = ContextNode(tools=tools, sample_limit=2)
    ctx_out = ctx.run([str(csv1), store_keys[0], "nonexistent"])
    assert store_keys[0] in ctx_out

    # --- 8) ErrorNode ---
    err_node = ErrorNode()
    try:
        raise ValueError("simulated error")
    except Exception as e:
        payload = err_node.run(e, step="execute", context={"sql": sql})
        assert payload["valid"] is False
        assert "formatted" in payload and "output" in payload["formatted"]

    # Cleanup
    vs.clear()
    schema_store.clear()
