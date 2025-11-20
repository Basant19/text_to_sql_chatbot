import os
import sys
import tempfile
import traceback
import pytest

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.schema_store import SchemaStore
from app.exception import CustomException

SAMPLE_CSV = """id,name,age
1,Alice,30
2,Bob,25
3,Charlie,40
4,Dana,28
"""

def _write_sample_csv(path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(SAMPLE_CSV)


def test_add_csv_and_retrieve(tmp_path):
    csv_path = str(tmp_path / "people.csv")
    _write_sample_csv(csv_path)

    store_path = str(tmp_path / "store.json")
    ss = SchemaStore(store_path=store_path, sample_limit=3)
    ss.clear()

    # Add CSV
    ss.add_csv(csv_path, csv_name="people")
    assert "people" in ss.list_csvs()

    cols = ss.get_schema("people")
    assert cols == ["id", "name", "age"]

    samples = ss.get_sample_rows("people")
    assert len(samples) == 3
    assert samples[0]["name"] == "Alice"

    # Clear store
    ss.clear()
    assert ss.list_csvs() == []


# ---------------- Standalone runner ----------------
def _run_as_script():
    print("Running schema_store tests in standalone mode (no pytest).")
    successes, failures = 0, 0

    td = tempfile.TemporaryDirectory()
    try:
        csv_path = os.path.join(td.name, "people.csv")
        _write_sample_csv(csv_path)

        store_path = os.path.join(td.name, "store.json")
        ss = SchemaStore(store_path=store_path, sample_limit=2)
        ss.clear()

        ss.add_csv(csv_path, csv_name="people")
        if "people" in ss.list_csvs():
            print("✔ add_csv + list_csvs: OK")
            successes += 1
        else:
            print("✖ add_csv + list_csvs: FAIL")
            failures += 1

        schema = ss.get_schema("people")
        if schema == ["id", "name", "age"]:
            print("✔ get_schema: OK")
            successes += 1
        else:
            print("✖ get_schema: FAIL")
            failures += 1

        samples = ss.get_sample_rows("people")
        if len(samples) == 2:
            print("✔ get_sample_rows: OK")
            successes += 1
        else:
            print("✖ get_sample_rows: FAIL")
            failures += 1

        ss.clear()
        if ss.list_csvs() == []:
            print("✔ clear store: OK")
            successes += 1
        else:
            print("✖ clear store: FAIL")
            failures += 1

    except Exception:
        print("✖ Exception during standalone tests")
        traceback.print_exc()
        failures += 1
    finally:
        td.cleanup()

    print(f"\nStandalone run complete. successes={successes}, failures={failures}")
    return failures == 0


if __name__ == "__main__":
    ok = _run_as_script()
    if not ok:
        sys.exit(1)
