#D:\text_to_sql_bot\tests\test_database.py
import os
import sys
import tempfile
import traceback

# Ensure project root on sys.path so imports work when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest  # noqa: E402
from app.database import get_connection, load_csv_table, execute_query, table_exists, list_tables, close_connection  # noqa: E402
from app.exception import CustomException  # noqa: E402
from app import config  # noqa: E402
from app.logger import get_logger  # noqa: E402

logger = get_logger("test_database")

SAMPLE_CSV = """id,name,age
1,Alice,30
2,Bob,25
3,Charlie,40
4,Dana,28
"""


def _write_sample_csv(path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(SAMPLE_CSV)


# ---------------- pytest tests ----------------
def test_connection_and_load_and_query(tmp_path, monkeypatch):
    # point DATABASE_PATH to a temp file
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr(config, "DATABASE_PATH", db_file, raising=False)

    # write csv
    csv_path = str(tmp_path / "people.csv")
    _write_sample_csv(csv_path)

    try:
        # load into table
        load_csv_table(csv_path, "people", force_reload=True)
        assert table_exists("people") is True

        # run a simple query
        rows, cols, meta = execute_query("SELECT COUNT(*) FROM people", read_only=True)
        # normalize the returned row value to an int and assert
        assert isinstance(rows, list)
        assert rows and int(rows[0][0]) == 4
        assert "COUNT" in cols[0].upper() or cols[0].lower().startswith("count")

        # select all rows
        rows2, cols2, meta2 = execute_query("SELECT id, name, age FROM people ORDER BY id", read_only=True)
        assert len(rows2) == 4
        # ensure columns normalized to expected names
        assert [c.lower() for c in cols2] == ["id", "name", "age"]
    finally:
        # ensure DB connection closed so Windows can remove file
        try:
            close_connection()
        except Exception:
            pass


def test_readonly_protects_destructive_sql(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr(config, "DATABASE_PATH", db_file, raising=False)

    csv_path = str(tmp_path / "people.csv")
    _write_sample_csv(csv_path)

    try:
        load_csv_table(csv_path, "people", force_reload=True)

        with pytest.raises(CustomException):
            # Drop should be rejected in read-only mode
            execute_query("DROP TABLE people", read_only=True)
    finally:
        try:
            close_connection()
        except Exception:
            pass


# ---------------- Standalone script runner ----------------
# ---------------- Standalone script runner ----------------
def _run_as_script():
    print("Running database tests in standalone mode (no pytest).")
    successes = 0
    failures = 0

    # import close_connection locally to avoid circular import issues at module load
    from app.database import close_connection  # noqa: E402

    td = tempfile.TemporaryDirectory()
    original_db = getattr(config, "DATABASE_PATH", None)
    try:
        db_file = os.path.join(td.name, "test.db")
        config.DATABASE_PATH = db_file

        csv_path = os.path.join(td.name, "people.csv")
        _write_sample_csv(csv_path)

        # load into table
        load_csv_table(csv_path, "people", force_reload=True)
        if table_exists("people"):
            print("✔ load_csv_table + table_exists: OK")
            successes += 1
        else:
            print("✘ load_csv_table + table_exists: FAIL")
            failures += 1

        # run a simple query
        rows, cols, meta = execute_query("SELECT COUNT(*) FROM people", read_only=True)
        if isinstance(rows, list) and rows and int(rows[0][0]) == 4:
            print("✔ execute_query (COUNT): OK")
            successes += 1
        else:
            print("✘ execute_query (COUNT): FAIL")
            failures += 1

        # test destructive SQL in read-only mode
        try:
            execute_query("DROP TABLE people", read_only=True)
            print("✘ execute_query (DROP in read-only): FAIL")
            failures += 1
        except CustomException:
            print("✔ execute_query (DROP in read-only): OK")
            successes += 1

    except Exception as e:
        print(f"Unexpected error during tests: {e}")
        traceback.print_exc()
        failures += 1
    finally:
        # 1) close DB connection so Windows can remove the file
        try:
            close_connection()
        except Exception:
            pass

        # 2) restore original DATABASE_PATH config
        if original_db is None:
            try:
                delattr(config, "DATABASE_PATH")
            except Exception:
                pass
        else:
            config.DATABASE_PATH = original_db

        # 3) cleanup tempdir
        try:
            td.cleanup()
        except Exception:
            pass

    print(f"\nStandalone run complete. successes={successes}, failures={failures}")
    return failures == 0


if __name__ == "__main__":
    ok = _run_as_script()
    if not ok:
        sys.exit(1)
