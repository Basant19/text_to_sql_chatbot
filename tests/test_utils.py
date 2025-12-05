# tests/test_utils.py
import unittest
from app import utils


class TestUtils(unittest.TestCase):

    def test_extract_sql_from_text_codeblock(self):
        text = "Here is SQL:\n```sql\nSELECT * FROM people;\n```"
        sql = utils.extract_sql_from_text(text)
        self.assertEqual(sql, "SELECT * FROM people;")

    def test_extract_sql_from_text_select(self):
        text = "Please run this query: SELECT id, name FROM people;"
        sql = utils.extract_sql_from_text(text)
        # inline SELECT extraction strips trailing semicolon
        self.assertEqual(sql, "SELECT id, name FROM people")

    def test_is_safe_sql(self):
        safe_sql = "SELECT * FROM people"
        unsafe_sql = "DROP TABLE people"
        self.assertTrue(utils.is_safe_sql(safe_sql))
        self.assertFalse(utils.is_safe_sql(unsafe_sql))

    def test_limit_sql_rows(self):
        sql = "SELECT * FROM people"
        limited_sql = utils.limit_sql_rows(sql, limit=10)
        self.assertIn("LIMIT 10", limited_sql)

    def test_format_few_shot_examples(self):
        examples = [{"query": "Get all people", "sql": "SELECT * FROM people"}]
        formatted = utils.format_few_shot_examples(examples)
        self.assertIn("Q: Get all people", formatted)
        self.assertIn("A: SELECT * FROM people", formatted)

    def test_format_retrieved_docs(self):
        # use shape supported by utils.format_retrieved_docs (id/text/meta)
        docs = [{"id": "doc1", "text": "Content of doc1", "meta": {"table_name": "Doc1"}}]
        formatted = utils.format_retrieved_docs(docs)
        self.assertIn("Doc1:", formatted)
        self.assertIn("Content of doc1", formatted)

    def test_build_prompt(self):
        # schema can be simple dict mapping table->list of cols
        schemas = {"people": ["id", "name"]}
        # retrieved docs use meta/title/text shape
        docs = [{"id": "doc1", "text": "Info", "meta": {"table_name": "Doc1"}}]
        few_shot = [{"query": "Get names", "sql": "SELECT name FROM people"}]
        prompt = utils.build_prompt("List all people", schemas, docs, few_shot)
        self.assertIn("User Query: List all people", prompt)
        self.assertIn("Table: people | Columns: id, name", prompt)
        self.assertIn("Doc1:", prompt)
        self.assertIn("Q: Get names", prompt)

    def test_flatten_schema(self):
        schema = {"people": ["id", "name"], "orders": ["id", "amount"]}
        text = utils.flatten_schema(schema)
        self.assertIn("Table: people | Columns: id, name", text)
        self.assertIn("Table: orders | Columns: id, amount", text)

    def test_preview_sample_rows(self):
        rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        preview = utils.preview_sample_rows(rows)
        self.assertIn("id=1, name=Alice", preview)
        self.assertIn("id=2, name=Bob", preview)


if __name__ == "__main__":
    unittest.main()
