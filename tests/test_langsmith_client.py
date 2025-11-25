import os
import sys
import traceback

# Ensure project root on sys.path so imports work when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest  # noqa: E402
from app import config  # noqa: E402
from app.langsmith_client import LangSmithClient  # noqa: E402
from app.exception import CustomException  # noqa: E402
from app.logger import get_logger  # noqa: E402

logger = get_logger("test_langsmith_client")


def _make_dummy_response(status=200, json_body=None, text_body=""):
    class DummyResp:
        def __init__(self, status, json_body, text_body):
            self.status_code = status
            self._json = json_body
            self.text = text_body

        def json(self):
            if self._json is None:
                raise ValueError("no json")
            return self._json

    return DummyResp(status, json_body, text_body)


def test_generate_success(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LANGSMITH_ENDPOINT", "https://fake.endpoint", raising=False)
    monkeypatch.setattr(config, "LANGSMITH_API_KEY", "fake-key", raising=False)
    monkeypatch.setattr(config, "LANGSMITH_PROJECT", "test-proj", raising=False)
    monkeypatch.setattr(config, "LANGSMITH_TRACING", True, raising=False)

    import requests
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _make_dummy_response(200, {"text": "hello from fake"}, "")

    monkeypatch.setattr(requests, "post", fake_post)

    client = LangSmithClient()
    out = client.generate("say hello", model="gpt-test", max_tokens=10)
    assert isinstance(out, dict)
    assert out["text"] == "hello from fake"
    assert "Authorization" in captured["headers"]
    assert "fake-key" in captured["headers"]["Authorization"]
    # URL no longer includes model in path, payload has model key
    assert captured["json"]["model"] == "gpt-test"


def test_generate_api_error_raises(monkeypatch):
    monkeypatch.setattr(config, "LANGSMITH_ENDPOINT", "https://fake.endpoint", raising=False)
    monkeypatch.setattr(config, "LANGSMITH_API_KEY", "fake-key", raising=False)
    monkeypatch.setattr(config, "LANGSMITH_TRACING", False, raising=False)

    import requests

    def fake_post_error(url, json=None, headers=None, timeout=None):
        return _make_dummy_response(500, {"error": "boom"}, "boom")

    monkeypatch.setattr(requests, "post", fake_post_error)

    client = LangSmithClient()
    with pytest.raises(CustomException):
        client.generate("this will fail", model="gpt-test", max_tokens=5)


# ---------------- Standalone runner ----------------
def _run_as_script():
    print("Running langsmith_client tests in standalone mode (no pytest).")
    successes = 0
    failures = 0

    original_endpoint = getattr(config, "LANGSMITH_ENDPOINT", None)
    original_key = getattr(config, "LANGSMITH_API_KEY", None)
    try:
        config.LANGSMITH_ENDPOINT = "https://fake.endpoint"
        config.LANGSMITH_API_KEY = "fake-key"
        config.LANGSMITH_TRACING = True

        import requests

        def fake_post(url, json=None, headers=None, timeout=None):
            return _make_dummy_response(200, {"text": "hi"}, "")

        requests.post = fake_post

        client = LangSmithClient()
        out = client.generate("hello", model="gpt-test")
        if out and out.get("text") == "hi":
            print("✔ generate success: OK")
            successes += 1
        else:
            print("✖ generate success: FAIL")
            failures += 1

        # simulate error
        def fake_post_error(url, json=None, headers=None, timeout=None):
            return _make_dummy_response(500, {"error": "boom"}, "boom")

        requests.post = fake_post_error
        try:
            client.generate("bad", model="gpt-test")
            print("✖ generate error handling: FAIL")
            failures += 1
        except Exception:
            print("✔ generate error handling: OK")
            successes += 1

    except Exception:
        print("✖ Exception during standalone tests")
        traceback.print_exc()
        failures += 1
    finally:
        if original_endpoint is None:
            try:
                delattr(config, "LANGSMITH_ENDPOINT")
            except Exception:
                pass
        else:
            config.LANGSMITH_ENDPOINT = original_endpoint

        if original_key is None:
            try:
                delattr(config, "LANGSMITH_API_KEY")
            except Exception:
                pass
        else:
            config.LANGSMITH_API_KEY = original_key

    print(f"\nStandalone run complete. successes={successes}, failures={failures}")
    return failures == 0


if __name__ == "__main__":
    ok = _run_as_script()
    if not ok:
        sys.exit(1)
