# tests/test_builder.py
import os
import sys

# Allow direct execution of tests (same pattern used in other tests)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest

from app.graph.builder import GraphBuilder


class DummyContext:
    def __init__(self, schemas=None):
        self._schemas = schemas or {"people": {"columns": ["id", "name"], "sample_rows": []}}

    def run(self, csv_names):
        # ignore csv_names for test simplicity
        return self._schemas


class DummyRetrieve:
    def run(self, user_query, schemas):
        return [{"id": "d1", "score": 0.9, "text": "doc", "meta": {"path": "/data/people.csv", "table_name": "people"}}]


class DummyPrompt:
    def run(self, user_query, schemas, retrieved_docs=None):
        return f"PROMPT for: {user_query}"


class DummyGenerate:
    def run(self, prompt):
        # return dict-like as generate_node would
        return {"prompt": prompt, "raw": {"mocked": True}, "sql": "SELECT id, name FROM people;"}


class DummyValidate:
    def run(self, sql, schemas):
        return {"sql": sql, "valid": True, "errors": []}


class DummyExecute:
    def run(self, sql, schemas):
        return {"rows": [{"id": 1, "name": "Alice"}], "columns": ["id", "name"], "meta": {"rowcount": 1}}


class DummyFormat:
    def run(self, sql, schemas, retrieved, execution, raw):
        return {"preview": "Alice", "rows": execution.get("rows") if execution else []}


class DummyError:
    def handle(self, exc, ctx):
        return {"prompt": None, "sql": None, "valid": False, "execution": None, "formatted": None, "raw": None, "error": f"handled: {exc}", "timings": {}}


def test_builder_happy_path():
    gb = GraphBuilder(
        context_node=DummyContext(),
        retrieve_node=DummyRetrieve(),
        prompt_node=DummyPrompt(),
        generate_node=DummyGenerate(),
        validate_node=DummyValidate(),
        execute_node=DummyExecute(),
        format_node=DummyFormat(),
        error_node=DummyError(),
    )

    out = gb.run("Who are the people?", ["people"], run_query=True)

    assert out["prompt"].startswith("PROMPT")
    assert out["sql"].upper().startswith("SELECT")
    assert out["valid"] is True
    assert isinstance(out["execution"], dict)
    assert out["formatted"]["preview"] == "Alice"
    assert out["raw"]["mocked"] is True
    assert out["error"] is None


def test_builder_error_path():
    class BrokenGenerate:
        def run(self, prompt):
            raise RuntimeError("boom")

    gb = GraphBuilder(
        context_node=DummyContext(),
        retrieve_node=DummyRetrieve(),
        prompt_node=DummyPrompt(),
        generate_node=BrokenGenerate(),
        validate_node=DummyValidate(),
        execute_node=DummyExecute(),
        format_node=DummyFormat(),
        error_node=DummyError(),
    )

    out = gb.run("trigger error", ["people"], run_query=False)

    assert out["valid"] is False
    assert out["error"].startswith("handled:")
