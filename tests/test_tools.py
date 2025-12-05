# tests/test_tools.py
import os
import shutil
import pytest
import numpy as np
from app.tools import Tools
from app.vector_search import get_vector_search

TEST_DIR = "./tests"
os.makedirs(TEST_DIR, exist_ok=True)
INDEX_PATH = os.path.join(TEST_DIR, "tools_faiss_test_index.faiss")
META_PATH = INDEX_PATH + ".meta.json"


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


def test_build_embedding_wrapper_fallback_and_shape():
    # Use Tools._build_embedding_wrapper directly via instance; expect deterministic fallback
    t = Tools(auto_init_vector=False)
    emb = t._build_embedding_wrapper(dim=12, config_module=None)

    # Single string -> list of floats of length 12
    v = emb("hello world")
    assert isinstance(v, list)
    assert len(v) == 12

    # List of strings -> list of vectors
    vs = emb(["one", "two", "three"])
    assert isinstance(vs, list)
    assert len(vs) == 3
    assert all(isinstance(vec, list) and len(vec) == 12 for vec in vs)


def test_auto_init_vector_and_upsert_search_integration():
    # Cleanup indices (ensure fresh)
    cleanup([INDEX_PATH, META_PATH])

    # Tools with auto_init_vector True and custom index path (use deterministic embedding wrapper)
    t = Tools(auto_init_vector=True, vector_index_path=INDEX_PATH)

    # Ensure vector backend was created
    assert t._vector_search is not None

    # Create docs and upsert
    docs = [
        {"text": "apple banana", "meta": {"category": "fruit"}},
        {"text": "python java", "meta": {"category": "programming"}},
    ]
    ids = t.upsert_vectors(docs)
    assert len(ids) == 2

    # Search for 'apple' should return the fruit doc
    results = t.search_vectors("apple", top_k=2)
    assert isinstance(results, list)
    assert any(r.get("meta", {}).get("category") == "fruit" for r in results)

    # Get metadata for inserted id
    meta0 = t.get_vector_meta(ids[0])
    assert isinstance(meta0, dict)
    assert "category" in meta0 or "text" in meta0

    # Clear and ensure empty
    t.clear_vectors()
    assert t._vector_search.info()["entries"] == 0

    cleanup([INDEX_PATH, META_PATH])


if __name__ == "__main__":
    pytest.main(["-q", __file__])
