# tests/test_schema_vector_integration.py
import os
import shutil
import tempfile
import unittest
from app.schema_store import SchemaStore
from app.vector_search import VectorSearch

# Simple deterministic embedding for testing
def simple_embedding(text: str):
    vec = [0.0] * 16
    for i, ch in enumerate(text[:16]):
        vec[i] = (ord(ch) % 97) * 0.01
    return vec

class TestSchemaVectorIntegration(unittest.TestCase):
    def setUp(self):
        # temporary workspace
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name
        os.makedirs(self.tmpdir, exist_ok=True)

        # create a sample CSV
        self.csv_path = os.path.join(self.tmpdir, "sample.csv")
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("id,name,score\n")
            f.write("1,apple,9.5\n")
            f.write("2,banana,7.2\n")
            f.write("3,cherry,8.1\n")

        # SchemaStore
        self.schema_store_path = os.path.join(self.tmpdir, "schema_store.json")
        self.store = SchemaStore(store_path=self.schema_store_path, sample_limit=3)

        # VectorSearch
        self.index_path = os.path.join(self.tmpdir, "vector.index")
        self.vs = VectorSearch(index_path=self.index_path, embedding_fn=simple_embedding, dim=16)

    def tearDown(self):
        # remove all files and directory
        self.tmpdir_obj.cleanup()
        # alternatively: shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_schema_and_vector_search(self):
        # --- SchemaStore ---
        key = self.store.add_csv(self.csv_path, "sample")
        self.assertIsInstance(key, str)

        schema = self.store.get_schema("sample")
        self.assertEqual(schema, ["id", "name", "score"])

        samples = self.store.get_sample_rows("sample")
        self.assertGreaterEqual(len(samples), 1)

        # --- VectorSearch ---
        docs = [{"id": s["id"], "text": s["name"], "meta": {"score": s["score"]}} for s in samples]
        ids = self.vs.upsert_documents(docs)
        self.assertEqual(len(ids), len(samples))

        # Search for 'apple'
        results = self.vs.search("apple", top_k=3)
        self.assertTrue(any(r["text"] == "apple" for r in results))

        # Check metadata preserved
        apple_meta = next((r["meta"] for r in results if r["text"] == "apple"), None)
        self.assertIsNotNone(apple_meta)
        self.assertEqual(apple_meta["score"], "9.5")


if __name__ == "__main__":
    unittest.main()
