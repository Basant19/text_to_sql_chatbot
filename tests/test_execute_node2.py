import os
import tempfile
from unittest.mock import MagicMock

from app.graph.nodes.execute_node import ExecuteNode


def test_execute_node_passes_table_map_for_numeric_table():
    tools = MagicMock()
    captured = {}

    def fake_execute_sql(sql, *, table_map=None, **kwargs):
        captured["sql"] = sql
        captured["table_map"] = table_map
        return {
            "rows": [{"app": "TestApp"}],
            "columns": ["app"],
            "meta": {"rowcount": 1},
        }

    tools.execute_sql = fake_execute_sql

    node = ExecuteNode(tools=tools)

    # -------------------------------------------------------
    # Create a REAL temporary CSV file (important!)
    # -------------------------------------------------------
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp.write("app,category\nTestApp,GAME\n")
        csv_path = tmp.name

    try:
        sql = "SELECT app FROM 7c7782_googleplaystore"

        schemas = {
            "7c7782_googleplaystore": {
                "path": csv_path
            }
        }

        node.run(
            sql,
            table_schemas=schemas,
            read_only=True
        )

        # ---------------------------------------------------
        # Assertions
        # ---------------------------------------------------
        assert captured["table_map"] is not None
        assert "7c7782_googleplaystore" in captured["table_map"]
        assert captured["table_map"]["7c7782_googleplaystore"] == os.path.abspath(csv_path)

    finally:
        # Cleanup temp file
        os.remove(csv_path)
