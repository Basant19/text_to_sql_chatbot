# tests/test_agent_run.py

import os
import sys
import pytest

# Make imports work when running file directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from app.graph.agent import Agent


class DummyNode:
    """Simple dummy node for testing."""
    def __init__(self, tag):
        self.tag = tag

    def run(self, data):
        return {"result": f"{self.tag}:{data.get('value', '')}"}


class ErrorNode:
    """Node that raises exception."""
    def run(self, data):
        raise RuntimeError("Node failed intentionally")


def test_agent_runs_all_nodes():
    nodes = {
        "step1": DummyNode("A"),
        "step2": DummyNode("B"),
    }

    agent = Agent(nodes)
    result = agent.run({"value": "X"})

    assert "final_output" in result
    assert result["final_output"]["result"] == "B:A:X"

    assert len(result["steps"]) == 2
    assert result["steps"][0]["node"] == "step1"
    assert result["steps"][1]["node"] == "step2"


def test_agent_catches_node_error():
    nodes = {
        "ok": DummyNode("OK"),
        "fail": ErrorNode(),
        "after_fail": DummyNode("AF")  # will receive {"error": "..."}
    }

    agent = Agent(nodes)
    result = agent.run({"value": "DATA"})

    # Error captured, not thrown
    assert "error" in result["steps"][1]["output"]
    assert "Node failed intentionally" in result["steps"][1]["output"]["error"]

    # Subsequent node gets the error dict
    assert result["steps"][2]["output"]["result"].startswith("AF")


def test_agent_validates_nodes_are_dict():
    with pytest.raises(ValueError):
        Agent(["not", "a", "dict"])
