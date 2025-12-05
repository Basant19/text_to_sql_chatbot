# tests/test_vector_search.py
import os
import shutil
import pytest
import numpy as np
from app.vector_search import get_vector_search, VectorSearch

TEST_DIR = "./tests"
os.makedirs(TEST_DIR, exist_ok=True)
INDEX_PATH = os.path.join(TEST_DIR, "faiss_test_index.faiss")
INDEX_PATH_DIM = os.path.join(TEST_DIR, "faiss_dim.faiss")


def cleanup(paths):
    for p in paths:
        try:
            if os.path.exists(p):
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
        except Exception:
            pass


def test_singleton_and_upsert_search():
    # cleanup any previous artifacts
    cleanup([INDEX_PATH, f"{INDEX_PATH}.meta.json"])

    # request singleton for the test index
    vs1 = get_vector_search(index_path=INDEX_PATH, dim=16)
    vs2 = get_vector_search(index_path=INDEX_PATH, dim=16)
    assert vs1 is vs2, "Singleton failed"

    # upsert documents
    docs = [
        {"text": "Hello world", "meta": {"category": "greeting"}},
        {"text": "Goodbye world", "meta": {"category": "farewell"}},
    ]
    ids = vs1.upsert_documents(docs)
    assert len(ids) == 2

    # search should find greeting for "Hello"
    results = vs1.search("Hello")
    assert len(results) >= 1
    # find best match with greeting category
    assert any(r.get("meta", {}).get("category") == "greeting" for r in results)

    info = vs1.info()
    assert info["entries"] == 2

    # clear index and assert entries reset
    vs1.clear()
    assert vs1.info()["entries"] == 0

    # create a new singleton using same index path; since cleared, entries should still be 0
    vs3 = get_vector_search(index_path=INDEX_PATH, dim=16)
    assert vs3.info()["entries"] == 0

    cleanup([INDEX_PATH, f"{INDEX_PATH}.meta.json"])


def test_dimension_adjustment():
    cleanup([INDEX_PATH_DIM, f"{INDEX_PATH_DIM}.meta.json"])
    vs = VectorSearch(index_path=INDEX_PATH_DIM, dim=8)
    # create an example longer vector
    arr = np.arange(12, dtype=float)
    vec = vs._adjust_and_normalize(np.asarray(arr, dtype=np.float32))
    assert len(vec) == 8
    # normalized
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)

    cleanup([INDEX_PATH_DIM, f"{INDEX_PATH_DIM}.meta.json"])


if __name__ == "__main__":
    pytest.main(["-q", __file__])
