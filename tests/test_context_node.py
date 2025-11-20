# tests/test_context_node.py
import os
import sys
import tempfile
import traceback

# Ensure project root on sys.path so imports work when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest  # noqa: E402

from app.graph.nodes.context_node import ContextNode  # noqa: E402
from app import schema_store  # noqa: E402
from app.logger import get_logger  # noqa: E402

logger = get_logger("test_context_node")

SAMPLE_CSV = """id,name,age
1,Alice,30
2,Bob,25
3,Charlie,40
4,Dana,28
"""


def _write_sample_csv(path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(SAMPLE_CSV)


def test_collect_schemas_returns_expected(tmp_path, monkeypatch):
    # Prepare a temp csv and a SchemaStore pointed at a temp store file
    csv_path = str(tmp_path / "people.csv")
    _write_sample_csv(csv_path)

    store_path = str(tmp_path / "store.json")
    ss = schema_store.SchemaStore(store_path=store_path, sample_limit=2)
    ss.clear()
    ss.add_csv(csv_path, csv_name="people")

    # Inject the SchemaStore into the ContextNode
    node = ContextNode(store=ss)
    result = node.run(["people"])

    assert isinstance(result, dict)
    assert "people" in result
    assert result["people"]["columns"] == ["id", "name", "age"]
    assert isinstance(result["people"]["sample_rows"], list)
    assert len(result["people"]["sample_rows"]) == 2


def test_missing_schema_returns_empty_entry(tmp_path):
    # Use an empty SchemaStore
    store_path = str(tmp_path / "store.json")
    ss = schema_store.SchemaStore(store_path=store_path, sample_limit=2)
    ss.clear()

    node = ContextNode(store=ss)
    result = node.run(["no_such_csv"])

    assert isinstance(result, dict)
    assert "no_such_csv" in result
    assert result["no_such_csv"]["columns"] == []
    assert result["no_such_csv"]["sample_rows"] == []


# ---------------- Standalone runner ----------------
def _run_as_script():
    print("Running context_node tests in standalone mode (no pytest).")
    successes = 0
    failures = 0

    with tempfile.TemporaryDirectory() as td:
        try:
            csv_path = os.path.join(td, "people.csv")
            _write_sample_csv(csv_path)

            store_path = os.path.join(td, "store.json")
            ss = schema_store.SchemaStore(store_path=store_path, sample_limit=2)
            ss.clear()
            ss.add_csv(csv_path, csv_name="people")

            node = ContextNode(store=ss)
            res = node.run(["people"])
            if res and res.get("people") and res["people"]["columns"] == ["id", "name", "age"]:
                print("✔ collect schema: OK")
                successes += 1
            else:
                print("✖ collect schema: FAIL")
                failures += 1

            # missing schema case
            res2 = node.run(["nope"])
            if "nope" in res2 and res2["nope"]["columns"] == []:
                print("✔ missing schema returns empty: OK")
                successes += 1
            else:
                print("✖ missing schema returns empty: FAIL")
                failures += 1

        except Exception:
            print("✖ Exception during standalone tests")
            traceback.print_exc()
            failures += 1

    print(f"\nStandalone run complete. successes={successes}, failures={failures}")
    return failures == 0


if __name__ == "__main__":
    ok = _run_as_script()
    if not ok:
        sys.exit(1)
