# app/langsmith_client.py
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
    Wrapper for LangSmith observability + fallback LLM generation.
    - generate(): Only for LLM calls (rarely used now; Gemini/Graph handles generation)
    - trace_run(): For LangSmith observability (preferred)
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

    # -------------------------------------------------------------------------
    # NEW — Observability-only tracing
    # -------------------------------------------------------------------------
    def trace_run(
        self,
        name: str,
        prompt: str,
        sql: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Sends a trace run to LangSmith for observability.
        Does NOT call LLM. Only logs metadata.

        Parameters
        ----------
        name : str
            Name of the run (e.g., "generate_sql", "retrieve_schema")
        prompt : str
            Prompt used in the node
        sql : Optional[str]
            Generated SQL (if available)
        metadata : dict
            Extra debug info (latency etc.)

        Returns: {"success": True, "run_id": <id>}
        """
        try:
            import requests  # type: ignore

            url = f"{self.endpoint}/v1/runs"
            payload = {
                "name": name,
                "project": self.project,
                "input": {"prompt": prompt},
                "output": {"sql": sql} if sql else {},
                "metadata": metadata or {},
            }

            if self.tracing:
                logger.info(f"[TRACE] Sending LangSmith trace: {name}")
                logger.debug(str(payload)[:1500])

            resp = requests.post(url, json=payload, headers=self._headers, timeout=10)

            if resp.status_code >= 400:
                logger.error(f"LangSmith trace_run failed ({resp.status_code})")
                return {"success": False, "error": resp.text}

            data = resp.json()
            return {"success": True, "run_id": data.get("id")}

        except Exception as e:
            logger.exception("trace_run failed")
            return {"success": False, "error": str(e)}

    # -------------------------------------------------------------------------
    # LLM generation (fallback only)
    # -------------------------------------------------------------------------
    def generate(self, prompt: str, model: str = "gemini-1.5-flash", max_tokens: int = 512, timeout: int = 30) -> Dict[str, Any]:
        """
        Prefer Gemini via LangGraph. Use this only as fallback.
        """
        start = time.time()
        try:
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
                try:
                    logger.debug(f"POST {url} payload={json.dumps(payload)[:1000]}")
                except Exception:
                    pass

            resp = requests.post(url, json=payload, headers=self._headers, timeout=timeout)

            if resp.status_code >= 400:
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                raise RuntimeError(f"LangSmith API error: {resp.status_code} {detail}")

            try:
                data = resp.json()
            except Exception:
                data = {"text": resp.text}

            text = None
            if isinstance(data, dict):
                for k in ("text", "output", "result", "content"):
                    if k in data:
                        v = data[k]
                        text = " ".join(v) if isinstance(v, list) else v
                        break

                if text is None and "outputs" in data:
                    first = data["outputs"][0]
                    text = first.get("text") if isinstance(first, dict) else None

            if text is None:
                text = json.dumps(data)

            runtime = time.time() - start
            logger.info(f"LangSmith generate completed in {runtime:.3f}s")

            return {"text": str(text), "raw": data}

        except Exception as e:
            logger.exception("LangSmith generate failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("LangSmithClient demo")
    try:
        c = LangSmithClient()
        print("Client initialised.")
    except Exception as e:
        print("Demo error:", e)
        sys.exit(1)
