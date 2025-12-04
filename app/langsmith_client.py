#D:\text_to_sql_bot\app\langsmith_client.py
import sys
import json
import time
from typing import Dict, Any, Optional, List

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("langsmith_client")


class LangSmithClient:
    """
    Wrapper for LangSmith observability (trace_run only).

    Improvements in this update:
      - Detailed diagnostic logging (masked headers) to help track 404/authorization issues
      - Endpoint fallback attempts (common path variants) to handle slightly different host setups
      - Temporary disable on persistent client errors (401/403/404) to avoid log spam
      - Configurable cooldown before retrying after a persistent error
      - Tolerant return values (dict with success/error) instead of raising on HTTP errors
      - Optional lightweight retry for transient network/server errors (5xx)
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

            # cooldown (seconds) after persistent failures before retrying
            self._retry_cooldown = int(getattr(config, "LANGSMITH_RETRY_COOLDOWN", 300))

            # disabled state to avoid noisy repeated failures
            self._disabled = False
            self._disabled_until: Optional[float] = None
            self._last_error: Optional[str] = None

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

            # Precompute endpoint candidates (common variants). When making the call we will try them in order.
            self._endpoint_candidates: List[str] = []
            if self.endpoint:
                self._endpoint_candidates.extend(
                    [
                        f"{self.endpoint}/v1/runs",
                        f"{self.endpoint}/api/v1/runs",
                        f"{self.endpoint}/v1/trace",
                        f"{self.endpoint}/v1/traces",
                    ]
                )

            logger.info("LangSmithClient initialized (endpoint=%s, tracing=%s)", self.endpoint or "none", self.tracing)
        except Exception as e:
            logger.exception("Failed to initialize LangSmithClient")
            raise CustomException(e, sys)

    # --------------------------
    # Internal helpers
    # --------------------------
    def _is_temporarily_disabled(self) -> bool:
        """
        Returns True if client is currently disabled due to a prior persistent failure.
        If a cooldown has expired, re-enable automatically.
        """
        if not self._disabled:
            return False
        if self._disabled_until is None:
            return True
        if time.time() >= self._disabled_until:
            # re-enable
            logger.info("LangSmithClient: cooldown expired; re-enabling trace attempts")
            self._disabled = False
            self._disabled_until = None
            self._last_error = None
            return False
        return True

    def _mark_disabled(self, reason: str) -> None:
        """
        Mark tracing as temporarily disabled and set a retry window.
        """
        self._disabled = True
        self._disabled_until = time.time() + max(1, int(self._retry_cooldown))
        self._last_error = reason
        logger.warning("LangSmithClient: disabling tracing for %ds due to error: %s", self._retry_cooldown, reason)

    def _mask_headers_for_log(self, headers: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in (headers or {}).items():
            if k.lower() == "authorization" and isinstance(v, str):
                # mask everything after 'Bearer '
                if v.lower().startswith("bearer "):
                    out[k] = "Bearer *****"
                else:
                    out[k] = "*****"
            else:
                out[k] = v
        return out

    # -------------------------------------------------------------------------
    # Observability-only tracing
    # -------------------------------------------------------------------------
    def trace_run(
        self,
        name: str,
        prompt: str,
        sql: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """
        Send a trace run to LangSmith for observability only (no generation).

        Returns a dict {"success": True, "run_id": ...} on success,
        otherwise {"success": False, "error": ...}.

        Behavior notes:
         - If LANGSMITH_ENDPOINT is not configured, returns error immediately.
         - If LANGSMITH_API_KEY not provided, returns error immediately.
         - On 401/403/404 the client disables itself for a cooldown window.
         - Will attempt endpoint candidate list; logs diagnostic info (masked headers, payload preview).
         - For transient 5xx/network errors it will try a couple of attempts before returning failure.
        """
        # Quick local/no-key guard: do not attempt network calls if no endpoint or no API key.
        if not self._endpoint_candidates:
            logger.debug("LangSmithClient.trace_run: no endpoint configured; skipping trace")
            return {"success": False, "error": "No LANGSMITH_ENDPOINT configured"}

        if not self.api_key:
            logger.debug("LangSmithClient.trace_run: no API key configured; skipping trace")
            return {"success": False, "error": "No LANGSMITH_API_KEY configured"}

        if self._is_temporarily_disabled():
            logger.debug("LangSmithClient.trace_run: tracing temporarily disabled (reason=%s)", self._last_error)
            return {"success": False, "error": f"Tracing temporarily disabled: {self._last_error}"}

        try:
            import requests  # local import to keep dependency optional

            payload = {
                "name": name,
                "project": self.project,
                "input": {"prompt": prompt},
                "output": {"sql": sql} if sql else {},
                "metadata": metadata or {},
                "timestamp": time.time(),
            }

            # show masked header + small payload preview when tracing/logging enabled
            if self.tracing:
                logger.info("[TRACE] Sending LangSmith trace: %s", name)
                try:
                    logger.debug("Trace headers: %s", json.dumps(self._mask_headers_for_log(self._headers)))
                    logger.debug("Trace payload preview: %s", json.dumps(payload)[:1500])
                except Exception:
                    logger.debug("Trace payload (non-serializable)")

            last_err = None
            # Try candidate endpoints in order; useful if a user configured endpoint without the /v1/runs suffix
            for url in self._endpoint_candidates:
                # small retry loop for transient errors
                attempts = 0
                while attempts < 2:
                    attempts += 1
                    if self.tracing:
                        logger.debug("Attempting LangSmith trace POST to %s (attempt %d)", url, attempts)
                    try:
                        resp = requests.post(url, json=payload, headers=self._headers, timeout=timeout)
                    except Exception as e:
                        last_err = str(e)
                        logger.warning("LangSmith trace POST to %s failed (network/exception): %s", url, e)
                        # for network errors, try next attempt or next candidate
                        time.sleep(0.5 * attempts)
                        continue

                    # got an HTTP response
                    status = resp.status_code
                    body_text = None
                    try:
                        body_text = resp.text
                    except Exception:
                        body_text = "<unreadable>"

                    # Caps the length of logged body to avoid huge logs
                    body_preview = (body_text[:1000] + "...") if body_text and len(body_text) > 1000 else body_text

                    if status >= 400:
                        # parse json body if possible
                        detail = None
                        try:
                            detail = resp.json()
                        except Exception:
                            detail = body_preview or f"HTTP {status}"

                        logger.error("LangSmith trace_run failed for %s (status=%s) body=%s", url, status, body_preview)

                        # persistent client errors -> disable
                        if status in (401, 403, 404):
                            self._mark_disabled(f"HTTP {status}: {detail}")
                            return {"success": False, "error": detail}

                        # for server errors (5xx) try again (up to attempts) then fallback
                        last_err = detail
                        # if we might retry the same URL, continue the loop; otherwise break to next candidate
                        if 500 <= status < 600 and attempts < 2:
                            time.sleep(0.5 * attempts)
                            continue
                        break

                    # success path
                    try:
                        data = resp.json()
                    except Exception:
                        data = {"raw": body_text}

                    run_id = data.get("id") or data.get("run_id") or data.get("data", {}).get("id") if isinstance(data, dict) else None
                    logger.info("LangSmith trace_run successful (url=%s run_id=%s)", url, run_id)
                    return {"success": True, "run_id": run_id}

                # end attempts for this url -> try next candidate

            # all candidates exhausted
            err = last_err or "All endpoint candidates failed"
            logger.error("LangSmith trace_run: all endpoint candidates failed: %s", err)
            return {"success": False, "error": err}

        except Exception as e:
            # Network or other exception: don't disable immediately, but record error
            logger.exception("LangSmithClient.trace_run failed (unexpected exception)")
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
