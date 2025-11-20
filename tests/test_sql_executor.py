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
from app import database  # noqa: E402
from app.sql_executor import execute_sql  # noqa: E402

logger = get_logger("test_sql_executor")

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
def test_execute_select_and_format(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr(config, "DATABASE_PATH", db_file, raising=False)

    csv_path = str(tmp_path / "people.csv")
    _write_sample_csv(csv_path)

    # load csv into DuckDB table using database helper
    database.load_csv_table(csv_path, "people", force_reload=True)

    res = execute_sql("SELECT id, name, age FROM people ORDER BY id", read_only=True)
    assert "rows" in res and "columns" in res and "meta" in res
    assert res["columns"] == ["id", "name", "age"]
    assert len(res["rows"]) == 4
    assert res["rows"][0]["name"] == "Alice"

    # test limit wrapper
    res2 = execute_sql("SELECT id, name, age FROM people ORDER BY id", read_only=True, limit=2)
    assert len(res2["rows"]) == 2


def test_disallow_destructive_sql(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr(config, "DATABASE_PATH", db_file, raising=False)

    csv_path = str(tmp_path / "people.csv")
    _write_sample_csv(csv_path)
    database.load_csv_table(csv_path, "people", force_reload=True)

    with pytest.raises(CustomException):
        execute_sql("DROP TABLE people", read_only=True)


# ---------------- Standalone runner ----------------
def _run_as_script():
    print("Running sql_executor tests in standalone mode (no pytest).")
    successes = 0
    failures = 0

    # import close_connection helper for cleanup
    from app.database import close_connection  # noqa: E402

    td = tempfile.TemporaryDirectory()
    original_db = getattr(config, "DATABASE_PATH", None)
    try:
        db_file = os.path.join(td.name, "test.db")
        config.DATABASE_PATH = db_file

        csv_path = os.path.join(td.name, "people.csv")
        _write_sample_csv(csv_path)
        database.load_csv_table(csv_path, "people", force_reload=True)

        res = execute_sql("SELECT id, name FROM people ORDER BY id", read_only=True)
        if res and res.get("rows") and res["rows"][0]["name"] == "Alice":
            print("✔ select & format: OK")
            successes += 1
        else:
            print("✖ select & format: FAIL")
            failures += 1

        # limit
        res2 = execute_sql("SELECT id, name FROM people ORDER BY id", read_only=True, limit=2)
        if len(res2["rows"]) == 2:
            print("✔ limit wrapping: OK")
            successes += 1
        else:
            print("✖ limit wrapping: FAIL")
            failures += 1

        # destructive SQL should be blocked
        try:
            execute_sql("DROP TABLE people", read_only=True)
            print("✖ destructive SQL block: FAIL")
            failures += 1
        except CustomException:
            print("✔ destructive SQL block: OK")
            successes += 1

    except Exception:
        print("✖ Exception during standalone tests")
        traceback.print_exc()
        failures += 1
    finally:
        # close DB connection before cleaning up (windows file lock)
        try:
            close_connection()
        except Exception:
            pass

        # restore original DATABASE_PATH
        if original_db is None:
            try:
                delattr(config, "DATABASE_PATH")
            except Exception:
                pass
        else:
            config.DATABASE_PATH = original_db

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
