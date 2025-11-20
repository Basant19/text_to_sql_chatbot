import sys
import json
import time
from typing import Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("langsmith_client")


class LangSmithClient:
    """
    Tiny wrapper for LangSmith / LLM service.
    - Uses config.LANGSMITH_ENDPOINT and config.LANGSMITH_API_KEY by default.
    - Exposes generate(prompt, model, max_tokens) -> {"text": str, "raw": dict}
    - Keeps tracing flag (config.LANGSMITH_TRACING) to enable extra logging.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        endpoint: Optional[str] = None,
        tracing: Optional[bool] = None,
    ):
        try:
            self.api_key = api_key or getattr(config, "LANGSMITH_API_KEY", None)
            self.project = project or getattr(config, "LANGSMITH_PROJECT", None)
            self.endpoint = endpoint or getattr(config, "LANGSMITH_ENDPOINT", None)
            self.tracing = tracing if tracing is not None else getattr(config, "LANGSMITH_TRACING", False)

            if not self.api_key:
                raise ValueError("LANGSMITH API key is required")

            # normalize endpoint (no trailing slash)
            if self.endpoint and self.endpoint.endswith("/"):
                self.endpoint = self.endpoint[:-1]

            self._headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-LangSmith-Project": self.project or "",
            }

            logger.info(f"LangSmithClient initialized (endpoint={self.endpoint}, tracing={self.tracing})")
        except Exception as e:
            logger.exception("Failed to initialize LangSmithClient")
            raise CustomException(e, sys)

    def generate(self, prompt: str, model: str = "gemini-1.5-flash", max_tokens: int = 512, timeout: int = 30) -> Dict[str, Any]:
        """
        Generate text from the LLM.
        Returns {"text": <string>, "raw": <full response dict>}
        Raises CustomException on failure.
        Note: This implementation expects a JSON response. It is intentionally generic
        so tests can monkeypatch requests.post() and return different payloads.
        """
        start = time.time()
        try:
            # import here so tests can monkeypatch requests.post easily
            import requests  # type: ignore

            if not self.endpoint:
                raise ValueError("No LANGSMITH endpoint configured")

            url = f"{self.endpoint}/v1/generate"
            payload = {
                "model": model,
                "input": prompt,
                "max_tokens": int(max_tokens),
                "project": self.project,
            }

            if self.tracing:
                logger.info("LangSmith generate() called — tracing enabled")
                logger.debug(f"POST {url} payload={json.dumps(payload)[:1000]}")

            resp = requests.post(url, json=payload, headers=self._headers, timeout=timeout)

            # raise for status codes >= 400
            if resp.status_code >= 400:
                logger.error(f"LangSmith generate failed status={resp.status_code}")
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                raise RuntimeError(f"LangSmith API error: {resp.status_code} {detail}")

            try:
                data = resp.json()
            except Exception:
                # If the service returns plain text, wrap it
                text = resp.text
                data = {"text": text}

            # Heuristic: try common fields for generated content
            text = None
            # prefer 'text'
            if isinstance(data, dict):
                for key in ("text", "output", "result", "content"):
                    if key in data and isinstance(data[key], (str, list)):
                        if isinstance(data[key], list):
                            # join if list of strings
                            text = " ".join(map(str, data[key]))
                        else:
                            text = data[key]
                        break
                # some APIs nest: { "outputs": [ { "text": "..." } ] }
                if text is None and "outputs" in data and isinstance(data["outputs"], list) and data["outputs"]:
                    first = data["outputs"][0]
                    if isinstance(first, dict) and "text" in first:
                        text = first["text"]

            if text is None:
                # fallback: serialize the raw response
                text = json.dumps(data)

            runtime = time.time() - start
            logger.info(f"LangSmith generate completed in {runtime:.3f}s")
            return {"text": str(text), "raw": data}
        except CustomException:
            raise
        except Exception as e:
            logger.exception("LangSmith generate failed")
            raise CustomException(e, sys)


# Simple demo when run as a module (will not actually call remote by default in tests)
if __name__ == "__main__":
    import os
    import tempfile

    print("LangSmithClient demo")
    try:
        # create a client using config values; will raise if API key missing
        c = LangSmithClient()
        print("Client initialized. Endpoint:", getattr(c, "endpoint", None))
        # Do not call generate() here to avoid network calls in demo.
        print("Demo complete — generate() not invoked to avoid network calls.")
    except Exception as e:
        print("Demo error:", e)
        sys.exit(1)
