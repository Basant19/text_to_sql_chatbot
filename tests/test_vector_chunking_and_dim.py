# tests/test_vector_chunking_and_dim.py
import pytest
from pathlib import Path
import csv
import numpy as np

from app.csv_loader import CSVLoader
from app.vector_search import get_vector_search, VectorSearch

CSV_CONTENT = [
    ["App", "Description"],
    ["Alpha", "This is a long description for Alpha app that should be chunked properly."],
    ["Beta", "Beta app has another description to test chunking behavior."],
]

@pytest.fixture
def tmp_csv(tmp_path):
    csv_file = tmp_path / "apps.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(CSV_CONTENT)
    return csv_file

@pytest.fixture
def faiss_index_path(tmp_path):
    return str(tmp_path / "index.faiss")


# -------------------------
# Test 1: Chunking behavior
# -------------------------
def test_chunking_and_overlap(tmp_csv, faiss_index_path):
    loader = CSVLoader(upload_dir=str(tmp_csv.parent), chunk_size=10, chunk_overlap=3)
    vs = get_vector_search(index_path=faiss_index_path, embedding_fn=lambda x: [0.0] * 16, dim=16)

    doc_ids = loader.chunk_and_index(str(tmp_csv), vector_search_client=vs, id_prefix="apps")
    assert len(doc_ids) > 0, "Chunks should be generated from CSV"

    # Validate metadata length exists for each chunk (use exposed metadata dict)
    for doc_id in doc_ids:
        meta = vs._metadata.get(doc_id)
        assert meta is not None, "Metadata should exist for each chunk"
        text = meta.get("text", "")
        # chunk_size in CSVLoader is measured in characters in this test setup.
        # Ensure chunk text is not wildly longer than chunk_size + overlap
        assert len(text) <= 10 + 3 + 50, "Chunk text unexpectedly long (allow small slack)"


# -------------------------
# Test 2: Embedding dimension handling
# -------------------------
def test_embedding_dimension_validation(faiss_index_path):
    # Use deterministic fallback embedding (None => internal default wrapper used)
    vs = get_vector_search(index_path=faiss_index_path, embedding_fn=None, dim=16)

    # Insert a document via upsert_documents (which handles embeddings internally)
    ids = vs.upsert_documents([{"id": "vec1", "text": "dummy"}])
    assert "vec1" in vs.list_ids()

    # Verify that _adjust_and_normalize pads/truncates vectors to the configured dim
    short_vec = np.zeros(8, dtype=float)
    normalized = vs._adjust_and_normalize(short_vec)
    assert normalized.shape[0] == 16

    long_vec = np.arange(32, dtype=float)
    normalized2 = vs._adjust_and_normalize(long_vec)
    assert normalized2.shape[0] == 16


# -------------------------
# Test 3: FAISS persistence across a "new session"
# -------------------------
def test_faiss_persistence(tmp_csv, faiss_index_path):
    # Create an instance and index chunks (upsert_documents persists index + meta)
    vs1 = get_vector_search(index_path=faiss_index_path, embedding_fn=lambda x: [0.0] * 16, dim=16)
    loader = CSVLoader(upload_dir=str(tmp_csv.parent), chunk_size=5, chunk_overlap=2)
    doc_ids = loader.chunk_and_index(str(tmp_csv), vector_search_client=vs1, id_prefix="persist")

    # Ensure the index was persisted by creating a fresh VectorSearch instance directly
    # (bypass singleton accessor so we truly re-load from disk)
    vs2 = VectorSearch(index_path=faiss_index_path, embedding_fn=None, dim=16)
    ids_after = set(vs2.list_ids())
    for d in doc_ids:
        assert d in ids_after, f"{d} should persist in the reloaded index"
