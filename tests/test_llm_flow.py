import os
import sys
import tempfile
import traceback

# Ensure project root on sys.path so imports work when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest  # noqa: E402

from app import config  # noqa: E402
from app.logger import get_logger  # noqa: E402
from app.exception import CustomException  # noqa: E402
from app import schema_store  # noqa: E402
from app import database  # noqa: E402
from app import vector_search  # noqa: E402
from app import llm_flow  # noqa: E402

logger = get_logger("test_llm_flow")

SAMPLE_CSV = """id,name,age
1,Alice,30
2,Bob,25
3,Charlie,40
4,Dana,28
"""


def _write_sample_csv(path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(SAMPLE_CSV)


class DummyClient:
    def __init__(self, response_text: str):
        self._resp = {"text": response_text, "raw": {"mocked": True}}

    def generate(self, prompt: str, model: str = "gpt", max_tokens: int = 512):
        # Return the mocked response object
        return self._resp


# ---------------- pytest tests ----------------
def test_generate_sql_without_execution(tmp_path, monkeypatch):
    # Setup a schema store with a CSV
    csv_path = str(tmp_path / "people.csv")
    _write_sample_csv(csv_path)

    store_path = str(tmp_path / "store.json")
    ss = schema_store.SchemaStore(store_path=store_path, sample_limit=2)
    ss.clear()
    ss.add_csv(csv_path, csv_name="people")

    # monkeypatch VectorSearch.search to return no docs (not needed)
    monkeypatch.setattr(vector_search, "VectorSearch", lambda *args, **kwargs: vector_search.VectorSearch(index_path=str(tmp_path / "idx"), embedding_fn=lambda t: [0.0]*8, dim=8))

    # monkeypatch LangSmithClient with our dummy client
    dummy_sql = "SELECT id, name FROM people ORDER BY id"
    dummy_client = DummyClient(dummy_sql)
    result = llm_flow.generate_sql_from_query("list names", ["people"], run_query=False, client=dummy_client)

    assert "sql" in result
    assert result["sql"].strip().upper().startswith("SELECT")
    assert result["valid"] is True
    assert result["execution"] is None


def test_generate_and_execute_returns_results(tmp_path, monkeypatch):
    # Setup DB & CSV
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr(config, "DATABASE_PATH", db_file, raising=False)

    csv_path = str(tmp_path / "people.csv")
    _write_sample_csv(csv_path)

    # load into duckdb
    database.load_csv_table(csv_path, "people", force_reload=True)

    # Create schema store to satisfy prompt building (not strictly required for execution)
    store_path = str(tmp_path / "store.json")
    ss = schema_store.SchemaStore(store_path=store_path, sample_limit=2)
    ss.clear()
    ss.add_csv(csv_path, csv_name="people")

    # Dummy client returns a safe SELECT
    dummy_client = DummyClient("SELECT id, name FROM people ORDER BY id")
    res = llm_flow.generate_sql_from_query("names please", ["people"], run_query=True, client=dummy_client)
    assert res["valid"] is True
    assert res["execution"] is not None
    assert isinstance(res["execution"].get("rows"), list)
    assert res["execution"]["rows"][0]["name"] == "Alice"


def test_unsafe_sql_is_blocked(tmp_path, monkeypatch):
    # Setup DB & CSV
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr(config, "DATABASE_PATH", db_file, raising=False)

    csv_path = str(tmp_path / "people.csv")
    _write_sample_csv(csv_path)
    database.load_csv_table(csv_path, "people", force_reload=True)

    # Dummy client returns a DROP statement
    dummy_client = DummyClient("DROP TABLE people")
    res = llm_flow.generate_sql_from_query("delete table", ["people"], run_query=True, client=dummy_client)

    # Should not execute; valid flag should be False and error present
    assert res["valid"] is False
    assert res["execution"] is None
    assert res["error"] is not None


# ---------------- Standalone runner ----------------
def _run_as_script():
    print("Running llm_flow tests in standalone mode (no pytest).")
    successes = 0
    failures = 0

    td = tempfile.TemporaryDirectory()
    try:
        tmp = td.name
        csv_path = os.path.join(tmp, "people.csv")
        _write_sample_csv(csv_path)

        # create schema store and DB
        store_path = os.path.join(tmp, "store.json")
        ss = schema_store.SchemaStore(store_path=store_path, sample_limit=2)
        ss.clear()
        ss.add_csv(csv_path, csv_name="people")

        db_file = os.path.join(tmp, "test.db")
        config.DATABASE_PATH = db_file
        database.load_csv_table(csv_path, "people", force_reload=True)

        # test generate only
        dummy_client = DummyClient("SELECT id, name FROM people ORDER BY id")
        r1 = llm_flow.generate_sql_from_query("names", ["people"], run_query=False, client=dummy_client)
        if r1 and r1.get("sql", "").upper().startswith("SELECT"):
            print("✔ generate only: OK")
            successes += 1
        else:
            print("✖ generate only: FAIL")
            failures += 1

        # test execute
        r2 = llm_flow.generate_sql_from_query("names", ["people"], run_query=True, client=dummy_client)
        if r2.get("execution") and isinstance(r2["execution"].get("rows"), list):
            print("✔ generate+execute: OK")
            successes += 1
        else:
            print("✖ generate+execute: FAIL")
            failures += 1

        # test unsafe
        dummy_client2 = DummyClient("DROP TABLE people")
        r3 = llm_flow.generate_sql_from_query("drop it", ["people"], run_query=True, client=dummy_client2)
        if r3.get("valid") is False and r3.get("error"):
            print("✔ unsafe SQL blocked: OK")
            successes += 1
        else:
            print("✖ unsafe SQL blocked: FAIL")
            failures += 1

    except Exception:
        print("✖ Exception during standalone tests")
        traceback.print_exc()
        failures += 1
    finally:
        # ensure DuckDB connection is closed so Windows can remove temp files
        try:
            database.close_connection()
        except Exception:
            # don't let a close error stop cleanup; log for visibility
            logger.warning("database.close_connection() raised while cleaning up standalone runner")

        # cleanup the temporary directory, but be tolerant of permission errors on Windows
        try:
            td.cleanup()
        except PermissionError as pe:
            # Best effort: try closing DB again, then retry
            logger.warning(f"PermissionError while cleaning temp dir: {pe}. Retrying close + cleanup.")
            try:
                database.close_connection()
            except Exception:
                pass
            try:
                td.cleanup()
            except Exception as e:
                logger.error(f"Final cleanup failed: {e}")



if __name__ == "__main__":
    ok = _run_as_script()
    if not ok:
        sys.exit(1)
