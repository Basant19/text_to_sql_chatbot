# tests/test_tools.py
import os
import sys
import tempfile
import pytest

# Ensure project root on sys.path for direct runs
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.tools import Tools
from app.exception import CustomException


# ---------------- Dummy/test doubles ----------------
class DummyDB:
    def __init__(self):
        self.loaded = []

    def load_csv_table(self, path, table_name, force_reload=False):
        self.loaded.append((path, table_name, force_reload))

    def list_tables(self):
        return ["people", "orders"]


class DummySchemaStore:
    def __init__(self):
        self.data = {
            "people": {"columns": ["id", "name"], "sample_rows": [{"id": 1, "name": "Alice"}]}
        }

    def get_schema(self, name):
        return self.data.get(name, {}).get("columns")

    def get_sample_rows(self, name):
        return self.data.get(name, {}).get("sample_rows")

    def list_csvs(self):
        return list(self.data.keys())


class DummyVectorSearch:
    def search(self, query, top_k=5):
        return [{"id": "d1", "score": 0.9, "text": f"hit for {query}", "meta": {"path": "/data/people.csv"}}]


class DummyExecutor:
    def execute_sql(self, sql, read_only=True, limit=None, as_dataframe=False):
        return {"rows": [{"id": 1, "name": "Alice"}], "columns": ["id", "name"], "meta": {"rowcount": 1, "runtime": 0.001}}


# ---------------- Tests ----------------
def test_load_table_calls_db():
    db = DummyDB()
    t = Tools(db=db, schema_store=DummySchemaStore(), vector_search=DummyVectorSearch(), executor=DummyExecutor())
    t.load_table("/tmp/people.csv", "people", force_reload=True)
    assert db.loaded == [("/tmp/people.csv", "people", True)]


def test_list_tables_returns_expected():
    db = DummyDB()
    t = Tools(db=db, schema_store=DummySchemaStore(), vector_search=DummyVectorSearch(), executor=DummyExecutor())
    assert t.list_tables() == ["people", "orders"]


def test_get_schema_and_samples():
    ss = DummySchemaStore()
    t = Tools(db=DummyDB(), schema_store=ss, vector_search=DummyVectorSearch(), executor=DummyExecutor())
    assert t.get_schema("people") == ["id", "name"]
    assert t.get_sample_rows("people") == [{"id": 1, "name": "Alice"}]
    assert t.list_csvs() == ["people"]


def test_search_vectors_returns_docs():
    vs = DummyVectorSearch()
    t = Tools(db=DummyDB(), schema_store=DummySchemaStore(), vector_search=vs, executor=DummyExecutor())
    docs = t.search_vectors("apple", top_k=3)
    assert isinstance(docs, list)
    assert docs[0]["text"].startswith("hit for apple")


def test_execute_sql_wrapper():
    execu = DummyExecutor()
    t = Tools(db=DummyDB(), schema_store=DummySchemaStore(), vector_search=DummyVectorSearch(), executor=execu)
    res = t.execute_sql("SELECT * FROM people", read_only=True, limit=10)
    assert isinstance(res, dict)
    assert res["meta"]["rowcount"] == 1
    assert res["rows"][0]["name"] == "Alice"


def test_tools_raises_if_missing_methods():
    # Provide a DB without required methods
    class BadDB:
        pass

    with pytest.raises(CustomException):
        Tools(db=BadDB(), schema_store=DummySchemaStore(), vector_search=DummyVectorSearch(), executor=DummyExecutor()).list_tables()
