import os
import tempfile
import unittest
from typing import List, Dict

from app.csv_loader import load_csv_metadata
from app.schema_store import SchemaStore
from app.vector_search import VectorSearch


def simple_deterministic_emb(text: str) -> List[float]:
    """Simple deterministic embedding for tests."""
    dim = 16
    vec = [0.0] * dim
    s = (text or "")[:256]
    for i, ch in enumerate(s):
        vec[i % dim] += (ord(ch) % 97) * 0.01
    # normalize
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class TestCSVSchemaVectorIntegration(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.tmpdir = self.td.name

        # create small CSV
        self.csv_path = os.path.join(self.tmpdir, "sample.csv")
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            f.write("id,name,score\n1,apple,9.5\n2,banana,7.2\n3,cherry,8.1\n")

        # SchemaStore pointing to temp file
        self.schema_store_path = os.path.join(self.tmpdir, "schema_store.json")
        self.store = SchemaStore(store_path=self.schema_store_path, sample_limit=3)

    def tearDown(self):
        self.td.cleanup()

    def test_csv_metadata_and_schema_store_and_vector_search(self):
        # 1) Load metadata
        metadata = load_csv_metadata(self.csv_path, sample_rows=2)
        self.assertIsInstance(metadata, dict)
        self.assertIn("columns", metadata)
        self.assertEqual(metadata["columns"], ["id", "name", "score"])

        # 2) Add to SchemaStore
        key = self.store.add_csv(self.csv_path, "sample")
        self.assertIsInstance(key, str)

        # list and check
        csvs = self.store.list_csvs()
        self.assertIn(key, csvs)
        self.assertEqual(self.store.get_schema("sample"), ["id", "name", "score"])
        samples = self.store.get_sample_rows("sample")
        self.assertIsInstance(samples, list)
        self.assertGreaterEqual(len(samples), 1)

        # 3) VectorSearch test
        idx_path = os.path.join(self.tmpdir, "vector.index")
        vs = VectorSearch(index_path=idx_path, embedding_fn=simple_deterministic_emb, dim=16)
        vs.clear()
        self.assertEqual(vs.info()["entries"], 0)

        ids = vs.upsert_documents([{"id": "doc1", "text": "apple banana", "meta": {"source": "test"}}])
        self.assertEqual(len(ids), 1)

        results = vs.search("apple", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        first = results[0]
        self.assertIn("id", first)
        self.assertIn(first["id"], ids)
        self.assertIn("text", first)
        self.assertEqual(first["text"], "apple banana")

        vs.clear()


if __name__ == "__main__":
    unittest.main()
