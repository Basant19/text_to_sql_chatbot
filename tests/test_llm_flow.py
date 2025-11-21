# tests/test_llm_flow.py

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from app.llm_flow import generate_sql_from_query
from app.exception import CustomException


@pytest.fixture
def dummy_schema_store():
    with patch("app.llm_flow.schema_store.SchemaStore") as mock_ss:
        instance = mock_ss.return_value
        instance.get_schema.return_value = ["col1", "col2"]
        instance.get_sample_rows.return_value = [{"col1": 1, "col2": "a"}]
        yield instance


@pytest.fixture
def dummy_vector_search():
    with patch("app.llm_flow.vector_search.VectorSearch") as mock_vs:
        instance = mock_vs.return_value
        instance.search.return_value = [
            {"meta": {"table_name": "test_csv"}, "content": "sample doc"}
        ]
        yield instance


@pytest.fixture
def dummy_langsmith_client():
    mock_client = MagicMock()
    mock_client.generate.return_value = {"text": "SELECT * FROM test_csv;"}
    yield mock_client


@pytest.fixture
def dummy_langgraph_agent():
    with patch("app.llm_flow.LangGraphAgent") as mock_agent:
        instance = mock_agent.return_value
        instance.run.return_value = (
            "SELECT * FROM test_csv;",
            "PROMPT TEXT",
            {"llm": "response"}
        )
        yield instance


@pytest.fixture
def dummy_sql_executor():
    with patch("app.llm_flow.sql_executor.execute_sql") as mock_exec:
        mock_exec.return_value = [{"col1": 1, "col2": "a"}]
        yield mock_exec


def test_generate_sql_langgraph(
    dummy_schema_store,
    dummy_vector_search,
    dummy_langgraph_agent,
    dummy_sql_executor,
    dummy_langsmith_client
):
    result = generate_sql_from_query(
        user_query="Get all rows",
        csv_names=["test_csv"],
        run_query=True,
        top_k=1,
        client=dummy_langsmith_client,
        use_langgraph=True
    )
    assert result["sql"] == "SELECT * FROM test_csv;"
    assert result["prompt"] == "PROMPT TEXT"
    assert result["valid"] is True
    assert result["execution"] == [{"col1": 1, "col2": "a"}]
    assert result["raw"] == {"llm": "response"}
    assert result["error"] is None
