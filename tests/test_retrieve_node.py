
import os
import sys
import tempfile
import traceback

# Ensure project root on sys.path so imports work when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest  # noqa: E402

from app.graph.nodes.retrieve_node import RetrieveNode  # noqa: E402
from app.logger import get_logger  # noqa: E402

logger = get_logger("test_retrieve_node")


class FakeVectorSearch:
    """A tiny fake VectorSearch used for deterministic tests."""
    def __init__(self, docs=None):
        # docs should be list of dicts with keys: id, score, text, meta
        self._docs = docs or []

    def search(self, query: str, top_k: int = 5):
        # ignore query/top_k for deterministic behavior
        return self._docs[:top_k]


def test_retrieve_filters_by_table_name():
    docs = [
        {"id": "d1", "score": 0.9, "text": "People doc", "meta": {"path": "/data/people.csv", "table_name": "people"}},
        {"id": "d2", "score": 0.8, "text": "Orders doc", "meta": {"path": "/data/orders.csv", "table_name": "orders"}},
        {"id": "d3", "score": 0.7, "text": "Generic doc", "meta": {"source": "misc"}}
    ]
    fake_vs = FakeVectorSearch(docs=docs)
    node = RetrieveNode(vs_instance=fake_vs)

    # filter by 'people' should return only the people doc
    res = node.run("some query", csv_names=["people"], top_k=5)
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0]["id"] == "d1"

    # filter by name that doesn't match metadata should fallback to original results
    res2 = node.run("some query", csv_names=["nonexistent_table"], top_k=5)
    # fallback expected: returns original results since none matched
    assert len(res2) == 3


def test_retrieve_empty_results_return_empty():
    fake_vs = FakeVectorSearch(docs=[])
    node = RetrieveNode(vs_instance=fake_vs)
    res = node.run("anything", csv_names=["people"], top_k=3)
    assert res == []


def test_retrieve_no_filter_returns_all():
    docs = [
        {"id": "d1", "score": 0.9, "text": "X", "meta": {"path": "x"}},
        {"id": "d2", "score": 0.8, "text": "Y", "meta": {"path": "y"}},
    ]
    fake_vs = FakeVectorSearch(docs=docs)
    node = RetrieveNode(vs_instance=fake_vs)
    res = node.run("q", csv_names=None, top_k=5)
    assert len(res) == 2
    assert res[0]["id"] == "d1"


# ---------------- Standalone runner ----------------
def _run_as_script():
    print("Running retrieve_node tests in standalone mode (no pytest).")
    successes = 0
    failures = 0

    with tempfile.TemporaryDirectory() as td:
        try:
            docs = [
                {"id": "d1", "score": 0.9, "text": "People doc", "meta": {"path": "/data/people.csv", "table_name": "people"}},
                {"id": "d2", "score": 0.8, "text": "Orders doc", "meta": {"path": "/data/orders.csv", "table_name": "orders"}},
            ]
            fake_vs = FakeVectorSearch(docs=docs)
            node = RetrieveNode(vs_instance=fake_vs)

            r1 = node.run("q", csv_names=["people"], top_k=5)
            if len(r1) == 1 and r1[0]["id"] == "d1":
                print("✔ filter by table: OK")
                successes += 1
            else:
                print("✖ filter by table: FAIL")
                failures += 1

            r2 = node.run("q", csv_names=None, top_k=5)
            if len(r2) == 2:
                print("✔ no filter returns all: OK")
                successes += 1
            else:
                print("✖ no filter returns all: FAIL")
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
