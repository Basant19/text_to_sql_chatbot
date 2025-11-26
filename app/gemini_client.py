# app/gemini_client.py
import sys
import json
import time
from typing import Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("gemini_client")


class GeminiClient:
    """
    Wrapper to call Google Gemini / Generative Language model.

    Preference order:
      1. Use google-generativeai SDK if available (recommended).
      2. Fallback to a simple REST call to the Generative Language API.

    Constructor reads:
      - config.GEMINI_API_KEY (required)
      - config.GEMINI_ENDPOINT (optional - if you want to override the REST base URL)
      - config.GEMINI_MODEL (optional default model)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        default_model: Optional[str] = None,
        tracing: Optional[bool] = None,
    ):
        try:
            self.api_key = api_key or getattr(config, "GEMINI_API_KEY", None)
            self.endpoint = endpoint or getattr(config, "GEMINI_ENDPOINT", "")  # optional override
            self.default_model = default_model or getattr(config, "GEMINI_MODEL", "gemini-1.5")
            self.tracing = tracing if tracing is not None else getattr(config, "LANGSMITH_TRACING", False)

            if not self.api_key:
                raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for GeminiClient")

            # Try to initialize SDK if available. We'll keep a flag to know which backend to use.
            self._use_sdk = False
            try:
                # The SDK module name may be 'google.generativeai' or 'generativeai' depending on package.
                try:
                    import google.generativeai as genai  # type: ignore
                except Exception:
                    import generativeai as genai  # type: ignore
                # Configure SDK with API key (the SDK usage may vary; common pattern is configure)
                # We set attribute for later use
                genai.configure(api_key=self.api_key)  # some SDKs expose configure()
                self._genai = genai
                self._use_sdk = True
                logger.info("GeminiClient: using google-generativeai SDK")
            except Exception:
                self._use_sdk = False
                self._genai = None
                logger.info("GeminiClient: google-generativeai SDK not available, will use REST fallback")

            logger.info(f"GeminiClient initialized (model={self.default_model}, endpoint={self.endpoint or 'default'})")
        except Exception as e:
            logger.exception("Failed to initialize GeminiClient")
            raise CustomException(e, sys)

    def generate(self, prompt: str, model: Optional[str] = None, max_tokens: int = 512, timeout: int = 30) -> Dict[str, Any]:
        """
        Generate text using Gemini model.

        Returns:
            {"text": <string>, "raw": <raw response dict>}
        Raises:
            CustomException on errors.
        """
        start = time.time()
        try:
            model = model or self.default_model

            if self._use_sdk and self._genai is not None:
                # SDK path - try to call a common generate_text API. SDK shapes vary by version.
                try:
                    # many examples use: generativeai.generate_text(model=model, prompt=...)
                    resp = self._genai.generate_text(model=model, prompt=prompt, max_output_tokens=int(max_tokens))
                    # SDK may return an object; try to extract text and raw dict
                    text = None
                    raw = None
                    # if response is a dict-like
                    if isinstance(resp, dict):
                        raw = resp
                        # common fields:
                        for key in ("output", "text", "content", "candidates"):
                            if key in resp:
                                val = resp[key]
                                if isinstance(val, str):
                                    text = val
                                    break
                                if isinstance(val, list) and val:
                                    # join candidate text pieces
                                    text = " ".join(map(str, val))
                                    break
                    else:
                        # try to access attribute 'text' or 'candidates'
                        raw = resp
                        text = getattr(resp, "text", None) or getattr(resp, "output", None)
                        if not text:
                            # some SDK return nested structure
                            try:
                                # attempt best-effort conversion
                                raw_json = json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
                                # try typical keys
                                if isinstance(raw_json, dict):
                                    for key in ("output", "text", "content", "candidates"):
                                        if key in raw_json:
                                            t = raw_json[key]
                                            if isinstance(t, str):
                                                text = t
                                                break
                                            if isinstance(t, list) and t:
                                                text = " ".join(map(str, t))
                                                break
                                raw = raw_json
                            except Exception:
                                raw = {"result": str(resp)}
                                text = str(resp)

                    if text is None:
                        # fallback to stringified raw
                        text = str(raw)
                    runtime = time.time() - start
                    logger.info(f"GeminiClient (SDK) completed in {runtime:.3f}s")
                    return {"text": str(text), "raw": raw}
                except Exception as e:
                    logger.exception("GeminiClient SDK call failed; falling back to REST")
                    # fall through to REST path

            # REST fallback path
            # Default Google Generative Language REST endpoint:
            # https://generativelanguage.googleapis.com/v1beta2/models/{model}:generate
            # Allow override via self.endpoint (must NOT include trailing slash)
            base = self.endpoint.rstrip("/") if self.endpoint else "https://generativelanguage.googleapis.com/v1beta2"
            url = f"{base}/models/{model}:generate"

            payload = {
                "input": prompt,
                "max_output_tokens": int(max_tokens),
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            if self.tracing:
                logger.info("GeminiClient (REST) calling %s", url)
                try:
                    logger.debug("Payload preview: %s", json.dumps(payload)[:1000])
                except Exception:
                    logger.debug("Payload (non-serializable)")

            import requests  # local import so tests can monkeypatch

            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code >= 400:
                logger.error("GeminiClient REST failed status=%s", resp.status_code)
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                raise RuntimeError(f"Gemini API error: {resp.status_code} {detail}")

            try:
                data = resp.json()
            except Exception:
                data = {"text": resp.text}

            # heuristic: extract text
            text = None
            if isinstance(data, dict):
                # typical shapes: {"candidates":[{"output":"..."}, ...]} or {"output": "..."} or {"text":"..."}
                if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
                    first = data["candidates"][0]
                    if isinstance(first, dict) and "output" in first:
                        text = first["output"]
                for key in ("output", "text", "content", "result"):
                    if text:
                        break
                    if key in data and isinstance(data[key], (str, list)):
                        val = data[key]
                        if isinstance(val, list):
                            text = " ".join(map(str, val))
                        else:
                            text = val
                # nested 'outputs' sometimes used
                if text is None and "outputs" in data and isinstance(data["outputs"], list) and data["outputs"]:
                    first = data["outputs"][0]
                    if isinstance(first, dict) and "text" in first:
                        text = first["text"]

            if text is None:
                text = json.dumps(data)

            runtime = time.time() - start
            logger.info(f"GeminiClient (REST) completed in {runtime:.3f}s")
            return {"text": str(text), "raw": data}
        except CustomException:
            raise
        except Exception as e:
            logger.exception("GeminiClient.generate failed")
            raise CustomException(e, sys)


# quick demo when run as script (won't call remote if API key missing)
if __name__ == "__main__":
    print("GeminiClient demo")
    try:
        c = GeminiClient()
        print("Initialized GeminiClient. Default model:", c.default_model)
        print("Call generate() only if GEMINI_API_KEY is set to avoid network calls.")
    except Exception as e:
        print("Init failed:", e)
        sys.exit(1)
