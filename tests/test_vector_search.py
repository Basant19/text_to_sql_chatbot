import os
import sys
import tempfile
import traceback

# Ensure project root on sys.path so imports work when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest  # noqa: E402

from app.vector_search import VectorSearch  # noqa: E402
from app import config  # noqa: E402
from app.logger import get_logger  # noqa: E402
from app.exception import CustomException  # noqa: E402

logger = get_logger("test_vector_search")


# Deterministic embedding function for tests:
def deterministic_embedding(text: str) -> list:
    """
    Very small deterministic embedding:
    - length 8 vector
    - value i is count of occurrences of chr(97+i) in text
    """
    dim = 8
    vec = [0.0] * dim
    for ch in (text or "").lower():
        idx = ord(ch) - 97
        if 0 <= idx < dim:
            vec[idx] += 1.0
    # normalize
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


# ---------------- pytest tests ----------------
def test_upsert_and_search(tmp_path, monkeypatch):
    # Use a temporary index path
    idx_path = str(tmp_path / "test_index.faiss")
    monkeypatch.setattr(config, "VECTOR_INDEX_PATH", idx_path, raising=False)

    vs = VectorSearch(index_path=idx_path, embedding_fn=deterministic_embedding, dim=8)
    vs.clear()

    docs = [
        {"id": "d1", "text": "apple banana", "meta": {"source": "A"}},
        {"id": "d2", "text": "banana orange", "meta": {"source": "B"}},
        {"id": "d3", "text": "grape apple", "meta": {"source": "C"}},
    ]
    vs.upsert_documents(docs)

    # search for "apple"
    res = vs.search("apple", top_k=2)
    assert isinstance(res, list)
    assert len(res) == 2
    # top result should be one containing 'apple' (d1 or d3), and score positive
    assert any("apple" in (r.get("text") or "") for r in res)
    assert all(isinstance(r.get("score"), float) for r in res)

    # cleanup
    vs.clear()


def test_persistence_and_reload(tmp_path, monkeypatch):
    idx_path = str(tmp_path / "persist_index.faiss")
    monkeypatch.setattr(config, "VECTOR_INDEX_PATH", idx_path, raising=False)

    vs1 = VectorSearch(index_path=idx_path, embedding_fn=deterministic_embedding, dim=8)
    vs1.clear()

    docs = [
        {"id": "a", "text": "aaa b", "meta": {"tag": 1}},
        {"id": "b", "text": "c d e", "meta": {"tag": 2}},
    ]
    vs1.upsert_documents(docs)
    # make sure files written
    assert os.path.exists(idx_path) or os.path.exists(f"{idx_path}.npy")
    assert os.path.exists(f"{idx_path}.meta.json")

    # Create a new instance pointing to same path, it should load metadata and vectors
    vs2 = VectorSearch(index_path=idx_path, embedding_fn=deterministic_embedding, dim=8)
    res = vs2.search("aaa", top_k=1)
    assert len(res) >= 1
    assert res[0]["id"] in ("a", "b")

    vs1.clear()


def test_search_empty_index_returns_empty(tmp_path, monkeypatch):
    idx_path = str(tmp_path / "empty_index.faiss")
    monkeypatch.setattr(config, "VECTOR_INDEX_PATH", idx_path, raising=False)

    vs = VectorSearch(index_path=idx_path, embedding_fn=deterministic_embedding, dim=8)
    vs.clear()
    res = vs.search("anything", top_k=3)
    assert res == []


# ---------------- Standalone runner ----------------
def _run_as_script():
    print("Running vector_search tests in standalone mode (no pytest).")
    successes = 0
    failures = 0

    with tempfile.TemporaryDirectory() as td:
        try:
            idx_path = os.path.join(td, "index.faiss")
            original = getattr(config, "VECTOR_INDEX_PATH", None)
            config.VECTOR_INDEX_PATH = idx_path

            vs = VectorSearch(index_path=idx_path, embedding_fn=deterministic_embedding, dim=8)
            vs.clear()

            docs = [
                {"id": "d1", "text": "apple banana", "meta": {"source": "A"}},
                {"id": "d2", "text": "banana orange", "meta": {"source": "B"}},
            ]
            vs.upsert_documents(docs)
            print("✔ upsert_documents: OK")
            successes += 1

            res = vs.search("apple", top_k=2)
            if res and any("apple" in (r.get("text") or "") for r in res):
                print("✔ search returned relevant results: OK")
                successes += 1
            else:
                print("✖ search returned relevant results: FAIL")
                failures += 1

            # persistence
            vs2 = VectorSearch(index_path=idx_path, embedding_fn=deterministic_embedding, dim=8)
            res2 = vs2.search("banana", top_k=1)
            if res2:
                print("✔ persistence & reload: OK")
                successes += 1
            else:
                print("✖ persistence & reload: FAIL")
                failures += 1

            # cleanup
            vs.clear()
            if original is None:
                delattr(config, "VECTOR_INDEX_PATH")
            else:
                config.VECTOR_INDEX_PATH = original

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
