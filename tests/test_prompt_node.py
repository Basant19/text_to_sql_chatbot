# tests/test_prompt_node.py
import os
import sys
import tempfile
import traceback

# Ensure project root on sys.path so imports work when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest  # noqa: E402

from app.graph.nodes.prompt_node import PromptNode  # noqa: E402
from app.logger import get_logger  # noqa: E402

logger = get_logger("test_prompt_node")


def test_prompt_node_default_builder():
    schemas = {
        "people": {"columns": ["id", "name"], "sample_rows": [{"id": 1, "name": "Alice"}]},
        "orders": {"columns": ["id", "amount"], "sample_rows": [{"id": 10, "amount": 99}]},
    }
    docs = [
        {"id": "d1", "score": 0.9, "text": "People doc", "meta": {"path": "/data/people.csv", "table_name": "people"}}
    ]
    few_shot = [{"question": "List names", "sql": "SELECT name FROM people"}]

    node = PromptNode()
    prompt = node.run("Show me names", schemas, retrieved_docs=docs, few_shot=few_shot)

    assert isinstance(prompt, str)
    # accept several possible phrasings, case-insensitive
    assert "user query" in prompt.lower() or "user question" in prompt.lower()

    assert "people" in prompt.lower()
    assert "sample rows" in prompt.lower() or "sample" in prompt.lower()
    assert "SELECT name FROM people" in prompt or "SELECT name" in prompt


def test_prompt_node_custom_builder():
    # custom builder to assert injection works
    def fake_builder(user_query, schemas, retrieved_docs, few_shot):
        return f"Q:{user_query}|T:{','.join(schemas.keys())}|DOCS:{len(retrieved_docs)}|EX:{len(few_shot or [])}"

    schemas = {"u": {"columns": ["a"], "sample_rows": []}}
    docs = [{"id": "x", "score": 0.1, "text": "t"}]
    few_shot = [{"question": "q", "sql": "s"}]

    node = PromptNode(prompt_builder=fake_builder)
    out = node.run("hello", schemas, retrieved_docs=docs, few_shot=few_shot)

    assert out.startswith("Q:hello")
    assert "|T:u" in out
    assert "|DOCS:1" in out
    assert "|EX:1" in out


# ---------------- Standalone runner ----------------
def _run_as_script():
    print("Running prompt_node tests in standalone mode (no pytest).")
    successes = 0
    failures = 0

    with tempfile.TemporaryDirectory() as td:
        try:
            # default builder test
            schemas = {
                "people": {"columns": ["id", "name"], "sample_rows": [{"id": 1, "name": "Alice"}]},
            }
            node = PromptNode()
            prompt = node.run("Who are they?", schemas, retrieved_docs=[], few_shot=None)
            if isinstance(prompt, str) and "Who are they?" in prompt:
                print("✔ default builder: OK")
                successes += 1
            else:
                print("✖ default builder: FAIL")
                failures += 1

            # custom builder test
            def fake_builder(u, s, r, f):
                return "OK"
            node2 = PromptNode(prompt_builder=fake_builder)
            if node2.run("x", {}, [], None) == "OK":
                print("✔ custom builder injection: OK")
                successes += 1
            else:
                print("✖ custom builder injection: FAIL")
                failures += 1

        except Exception:
            print("✖ Exception during standalone tests")
            traceback.print_exc()
            failures += 1

    print(f"\nStandalone run complete. successes={successes}, failures={failures}")
    return failures == 0


if __name__ == "__main__":
    ok = _run_as_script()
    if not ok:
        sys.exit(1)
