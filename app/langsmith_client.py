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
    Wrapper for LangSmith observability (trace_run only).

    Design decisions:
      - By default this client is *observability-only* and should NOT be used as a generation provider.
      - trace_run(...) is the single supported operation for sending traces to LangSmith.
      - generate(...) is intentionally disabled by default. If you explicitly want to allow
        LangSmith to be used for generation (NOT recommended), set USE_LANGSMITH_FOR_GEN=true
        in your environment and provide LANGSMITH_API_KEY. The generate() method will then
        attempt a call but still logs a warning — you should prefer a dedicated provider like Gemini.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        endpoint: Optional[str] = None,
        tracing: Optional[bool] = None,
    ):
        try:
            # Read config defaults but do NOT require a key unconditionally.
            self.api_key = api_key or getattr(config, "LANGSMITH_API_KEY", None)
            self.project = project or getattr(config, "LANGSMITH_PROJECT", None)
            self.endpoint = endpoint or getattr(config, "LANGSMITH_ENDPOINT", None)
            # tracing flag controls verbose logging; separate from whether trace_run will attempt network call
            self.tracing = tracing if tracing is not None else getattr(config, "LANGSMITH_TRACING", False)

            if self.endpoint and self.endpoint.endswith("/"):
                self.endpoint = self.endpoint[:-1]

            # Prepare headers only if API key present; otherwise headers remain minimal
            if self.api_key:
                self._headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-LangSmith-Project": self.project or "",
                }
            else:
                # No API key — trace_run will be a no-op or return failure; do not throw here to allow local dev without LangSmith
                self._headers = {
                    "Content-Type": "application/json",
                    "X-LangSmith-Project": self.project or "",
                }
                if self.tracing:
                    logger.warning("LangSmithClient initialized without LANGSMITH_API_KEY; trace_run will not succeed until a key is provided")

            logger.info("LangSmithClient initialized (endpoint=%s, tracing=%s)", self.endpoint or "none", self.tracing)
        except Exception as e:
            logger.exception("Failed to initialize LangSmithClient")
            raise CustomException(e, sys)

    # -------------------------------------------------------------------------
    # Observability-only tracing
    # -------------------------------------------------------------------------
    def trace_run(
        self,
        name: str,
        prompt: str,
        sql: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a trace run to LangSmith for observability only (no generation).

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

        Returns
        -------
        dict
            {"success": True, "run_id": <id>} on success, otherwise {"success": False, "error": <message>}
        """
        # Quick local/no-key guard: do not attempt network calls if no endpoint or no API key.
        if not self.endpoint:
            logger.debug("LangSmithClient.trace_run: no endpoint configured; skipping trace")
            return {"success": False, "error": "No LANGSMITH_ENDPOINT configured"}

        if not self.api_key:
            logger.debug("LangSmithClient.trace_run: no API key configured; skipping trace")
            return {"success": False, "error": "No LANGSMITH_API_KEY configured"}

        try:
            import requests  # local import to keep dependency optional

            url = f"{self.endpoint}/v1/runs"
            payload = {
                "name": name,
                "project": self.project,
                "input": {"prompt": prompt},
                "output": {"sql": sql} if sql else {},
                "metadata": metadata or {},
                "timestamp": time.time(),
            }

            if self.tracing:
                logger.info("[TRACE] Sending LangSmith trace: %s", name)
                try:
                    logger.debug("Trace payload preview: %s", json.dumps(payload)[:1500])
                except Exception:
                    logger.debug("Trace payload (non-serializable)")

            resp = requests.post(url, json=payload, headers=self._headers, timeout=10)

            if resp.status_code >= 400:
                logger.error("LangSmith trace_run failed (status=%s): %s", resp.status_code, resp.text)
                return {"success": False, "error": resp.text}

            try:
                data = resp.json()
            except Exception:
                data = {"raw": resp.text}

            return {"success": True, "run_id": data.get("id") or data.get("run_id") or None}
        except Exception as e:
            logger.exception("LangSmithClient.trace_run failed")
            return {"success": False, "error": str(e)}

    # -------------------------------------------------------------------------
    # Disabled LLM generation (safety-first)
    # -------------------------------------------------------------------------
    def generate(self, prompt: str, model: str = "gemini-2.5-flash", max_tokens: int = 512, timeout: int = 30) -> Dict[str, Any]:
        """
        Intentionally disabled by default. LangSmith should NOT be used for generation in the
        recommended architecture. If you truly intend to allow LangSmith as a generator,
        set USE_LANGSMITH_FOR_GEN=true in your environment and provide LANGSMITH_API_KEY.

        If enabled, this method will attempt a minimal call to the LangSmith generate endpoint.
        Otherwise it raises a CustomException to prevent accidental usage.
        """
        use_langsmith_for_gen = getattr(config, "USE_LANGSMITH_FOR_GEN", False)
        if not use_langsmith_for_gen:
            msg = (
                "LangSmithClient.generate() is disabled by default. To enable, set USE_LANGSMITH_FOR_GEN=true "
                "and provide LANGSMITH_API_KEY in the environment. Recommended: use GeminiClient for generation."
            )
            logger.error(msg)
            raise CustomException(msg)

        # If operator explicitly opted in, perform a guarded call.
        if not self.endpoint or not self.api_key:
            raise CustomException("LANGSMITH_ENDPOINT and LANGSMITH_API_KEY must be configured to use LangSmith for generation.")

        start = time.time()
        try:
            import requests  # local import

            url = f"{self.endpoint}/v1/generate"
            payload = {
                "model": model,
                "input": prompt,
                "max_tokens": int(max_tokens),
                "project": self.project,
            }

            if self.tracing:
                logger.warning("LangSmithClient.generate() invoked — USE_LANGSMITH_FOR_GEN is true. This is not recommended.")
                try:
                    logger.debug("LangSmith generate payload preview: %s", json.dumps(payload)[:1000])
                except Exception:
                    pass

            resp = requests.post(url, json=payload, headers=self._headers, timeout=timeout)
            if resp.status_code >= 400:
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                logger.error("LangSmith generate failed (status=%s): %s", resp.status_code, detail)
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

                if text is None and "outputs" in data and isinstance(data["outputs"], list) and data["outputs"]:
                    first = data["outputs"][0]
                    if isinstance(first, dict):
                        text = first.get("text") or first.get("content") or first.get("output")

            if text is None:
                text = json.dumps(data)

            runtime = time.time() - start
            logger.info("LangSmith generate completed in %.3fs", runtime)

            return {"text": str(text), "raw": data}
        except Exception as e:
            logger.exception("LangSmith generate failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("LangSmithClient demo (trace-only)")
    try:
        c = LangSmithClient()
        print("Client initialised. trace_run is available.")
    except Exception as e:
        print("Demo error:", e)
        sys.exit(1)
