# tests/test_generate_node.py
import sys
import pytest
from unittest.mock import MagicMock, patch

from app.graph.nodes.generate_node import GenerateNode
from app.exception import CustomException


# ---------------- Mock LangSmith client ----------------
class DummyLangSmithClient:
    def generate(self, prompt, model=None, max_tokens=None):
        return {"text": "SELECT * FROM users;"}


# ---------------- Mock LangGraph agent ----------------
class DummyLangGraphAgentTuple:
    def run(self, prompt):
        # returns tuple (sql, prompt_text, raw)
        return "SELECT id FROM orders;", prompt, {"mocked": True}


class DummyLangGraphAgentDict:
    def run(self, prompt):
        # returns dict-like
        return {"sql": "SELECT name FROM people;", "prompt": prompt, "raw": {"mocked": True}}


class DummyLangGraphAgentString:
    def run(self, prompt):
        # returns string
        return "SELECT id FROM invoices;"


# ---------------- Tests ----------------
def test_generate_with_langsmith():
    client = DummyLangSmithClient()
    node = GenerateNode(client=client)
    res = node.run("Get all users")
    assert res["sql"].upper().startswith("SELECT")
    assert res["prompt"] == "Get all users"
    assert isinstance(res["raw"], dict)
    assert "text" in res["raw"]


def test_generate_with_langgraph_tuple():
    client = DummyLangGraphAgentTuple()
    node = GenerateNode(client=client)
    res = node.run("List orders")
    assert res["sql"].upper().startswith("SELECT")
    assert res["prompt"] == "List orders"
    assert isinstance(res["raw"], dict)
    assert res["raw"]["mocked"] is True


def test_generate_with_langgraph_dict():
    client = DummyLangGraphAgentDict()
    node = GenerateNode(client=client)
    res = node.run("List people")
    assert res["sql"].upper().startswith("SELECT")
    assert res["prompt"] == "List people"
    assert isinstance(res["raw"], dict)
    assert res["raw"]["mocked"] is True


def test_generate_with_langgraph_string():
    client = DummyLangGraphAgentString()
    node = GenerateNode(client=client)
    res = node.run("Get invoices")
    assert res["sql"].upper().startswith("SELECT")
    assert res["prompt"] == "Get invoices"
    assert isinstance(res["raw"], str)


def test_generate_raises_custom_exception():
    class BadClient:
        pass  # has neither generate nor run

    node = GenerateNode(client=BadClient())
    with pytest.raises(CustomException):
        node.run("Test query")
# tests/test_generate_node.py
import sys
import pytest
from unittest.mock import MagicMock, patch

from app.graph.nodes.generate_node import GenerateNode
from app.exception import CustomException


# ---------------- Mock LangSmith client ----------------
class DummyLangSmithClient:
    def generate(self, prompt, model=None, max_tokens=None):
        return {"text": "SELECT * FROM users;"}


# ---------------- Mock LangGraph agent ----------------
class DummyLangGraphAgentTuple:
    def run(self, prompt):
        # returns tuple (sql, prompt_text, raw)
        return "SELECT id FROM orders;", prompt, {"mocked": True}


class DummyLangGraphAgentDict:
    def run(self, prompt):
        # returns dict-like with nested 'raw'
        return {"sql": "SELECT name FROM people;", "prompt": prompt, "raw": {"mocked": True}}


class DummyLangGraphAgentString:
    def run(self, prompt):
        # returns string
        return "SELECT id FROM invoices;"


# ---------------- Tests ----------------
def test_generate_with_langsmith():
    client = DummyLangSmithClient()
    node = GenerateNode(client=client)
    res = node.run("Get all users")
    assert res["sql"].upper().startswith("SELECT")
    assert res["prompt"] == "Get all users"
    assert isinstance(res["raw"], dict)
    assert "text" in res["raw"]


def test_generate_with_langgraph_tuple():
    client = DummyLangGraphAgentTuple()
    node = GenerateNode(client=client)
    res = node.run("List orders")
    assert res["sql"].upper().startswith("SELECT")
    assert res["prompt"] == "List orders"
    assert isinstance(res["raw"], dict)
    assert res["raw"]["mocked"] is True


def test_generate_with_langgraph_dict():
    client = DummyLangGraphAgentDict()
    node = GenerateNode(client=client)
    res = node.run("List people")
    assert res["sql"].upper().startswith("SELECT")
    assert res["prompt"] == "List people"
    assert isinstance(res["raw"], dict)
    # nested 'raw' key now checked
    assert res["raw"]["raw"]["mocked"] is True


def test_generate_with_langgraph_string():
    client = DummyLangGraphAgentString()
    node = GenerateNode(client=client)
    res = node.run("Get invoices")
    assert res["sql"].upper().startswith("SELECT")
    assert res["prompt"] == "Get invoices"
    assert isinstance(res["raw"], str)


def test_generate_raises_custom_exception():
    class BadClient:
        pass  # has neither generate nor run

    node = GenerateNode(client=BadClient())
    with pytest.raises(CustomException):
        node.run("Test query")
