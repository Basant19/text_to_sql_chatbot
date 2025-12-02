# app/gemini_client.py
import sys
import json
import time
from typing import Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("gemini_client")


# Config-driven defaults (fall back to safe values)
_DEFAULT_TIMEOUT = getattr(config, "GEMINI_REST_TIMEOUT", 30)
_DEFAULT_RETRIES = getattr(config, "GEMINI_REST_RETRIES", 2)
_DEFAULT_BACKOFF_FACTOR = getattr(config, "GEMINI_REST_BACKOFF", 0.5)
_DEFAULT_MAX_RAW_PREVIEW = getattr(config, "MAX_RAW_SUMMARY_CHARS", 800)


def _safe_serialize_raw(raw: Any) -> Any:
    """
    Try to turn provider 'raw' into a JSON-friendly structure.
    If not possible, return a truncated string preview.
    """
    try:
        # first try a full dump (when raw is dict-like or has __dict__)
        return json.loads(json.dumps(raw, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        try:
            s = str(raw)
            if len(s) > _DEFAULT_MAX_RAW_PREVIEW:
                return s[: _DEFAULT_MAX_RAW_PREVIEW] + "…"
            return s
        except Exception:
            return "<unserializable raw response>"


class GeminiClient:
    """
    Adapter for Google Gemini (Generative Models) with robust fallbacks:
      1. LangChain ChatGoogleGenerativeAI wrapper (preferred)
      2. google.generativeai SDK
      3. REST fallback to Generative Language API with retry
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        default_model: Optional[str] = None,
        tracing: Optional[bool] = None,
    ):
        try:
            # config fallbacks
            self.api_key = api_key or getattr(config, "GEMINI_API_KEY", None) or getattr(config, "GOOGLE_API_KEY", None)
            self.endpoint = endpoint or getattr(config, "GEMINI_ENDPOINT", "")
            self.default_model = default_model or getattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
            self.tracing = tracing if tracing is not None else getattr(config, "LANGSMITH_TRACING", False)

            # do not force raising here — some flows may stub GeminiClient during tests.
            if not self.api_key:
                logger.warning("GeminiClient initialized without API key; REST/SDK calls will fail if invoked.")

            # Try LangChain Chat wrapper
            self._use_langchain_llm = False
            self._llm = None
            try:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
                except Exception:
                    try:
                        from langchain.google_genai import ChatGoogleGenerativeAI  # type: ignore
                    except Exception:
                        from langchain.chat_models import ChatGoogleGenerativeAI  # type: ignore

                try:
                    # wrapper often expects google_api_key kwarg
                    self._llm = ChatGoogleGenerativeAI(model=self.default_model, google_api_key=self.api_key)
                except TypeError:
                    self._llm = ChatGoogleGenerativeAI(model=self.default_model, api_key=self.api_key)
                self._use_langchain_llm = True
                logger.info("GeminiClient: using ChatGoogleGenerativeAI (LangChain wrapper)")
            except Exception as e:
                self._use_langchain_llm = False
                self._llm = None
                logger.debug("GeminiClient: LangChain wrapper not available: %s", e)

            # Try google.generativeai SDK
            self._use_genai_sdk = False
            self._genai = None
            if not self._use_langchain_llm:
                try:
                    try:
                        import google.generativeai as genai  # type: ignore
                    except Exception:
                        import generativeai as genai  # type: ignore

                    # configure SDK if possible
                    try:
                        genai.configure(api_key=self.api_key)
                    except Exception:
                        try:
                            genai.api_key = self.api_key  # type: ignore
                        except Exception:
                            pass

                    self._genai = genai
                    self._use_genai_sdk = True
                    logger.info("GeminiClient: using google.generativeai SDK")
                except Exception as e:
                    self._use_genai_sdk = False
                    self._genai = None
                    logger.debug("GeminiClient: genai SDK not available: %s", e)

            logger.info(
                "GeminiClient initialized (model=%s, endpoint=%s, langchain=%s, genai=%s)",
                self.default_model,
                self.endpoint or "<default>",
                self._use_langchain_llm,
                self._use_genai_sdk,
            )

        except Exception as e:
            logger.exception("Failed to initialize GeminiClient")
            raise CustomException(e, sys)

    def _normalize_model_name(self, model: str) -> str:
        """
        Return a model identifier in a stable form.
        This function returns either:
          - 'models/xxx' (if caller passed 'xxx' or 'models/xxx')
          - preserves 'tunedModels/...' if present
        """
        try:
            if not isinstance(model, str) or not model:
                return model
            if model.startswith("models/") or model.startswith("tunedModels/"):
                return model
            return f"models/{model}"
        except Exception:
            return model

    def _build_model_path(self, norm_model: str) -> str:
        """
        Derive the model path segment to embed into REST url WITHOUT duplicating 'models/'.
        If norm_model already contains 'models/' or 'tunedModels/', use it as-is.
        """
        if not isinstance(norm_model, str) or not norm_model:
            return norm_model
        if norm_model.startswith("models/") or norm_model.startswith("tunedModels/"):
            return norm_model
        return f"models/{norm_model}"

    def generate(self, prompt: str, model: Optional[str] = None, max_tokens: int = 512, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate text from Gemini provider.

        Returns:
            {"text": <str>, "raw": <json-serializable or truncated string>}
        """
        start = time.time()
        timeout = timeout or _DEFAULT_TIMEOUT
        try:
            model = model or self.default_model
            norm_model = self._normalize_model_name(model)
            model_path = self._build_model_path(norm_model)

            # 1) LangChain wrapper path
            if self._use_langchain_llm and self._llm is not None:
                try:
                    resp = self._llm.invoke(prompt)
                    text = None
                    raw = resp

                    # Common extraction heuristics
                    text = getattr(resp, "content", None) or getattr(resp, "text", None)
                    if text is None:
                        gens = getattr(resp, "generations", None)
                        if isinstance(gens, (list, tuple)) and len(gens) > 0:
                            first = gens[0]
                            text = getattr(first, "text", None) or getattr(first, "content", None)
                            if text is None and isinstance(first, dict):
                                text = first.get("text") or first.get("content")

                    if text is None:
                        try:
                            raw_json = json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
                            raw = raw_json
                            for key in ("content", "text", "output", "candidates"):
                                if key in raw_json:
                                    val = raw_json[key]
                                    if isinstance(val, str):
                                        text = val
                                        break
                                    if isinstance(val, list) and val:
                                        if isinstance(val[0], dict):
                                            text = val[0].get("content") or val[0].get("output") or val[0].get("text")
                                            if text:
                                                break
                                        else:
                                            text = " ".join(map(str, val))
                                            break
                        except Exception:
                            text = str(resp)

                    runtime = time.time() - start
                    logger.info("GeminiClient (LangChain wrapper) completed in %.3fs", runtime)
                    return {"text": str(text or ""), "raw": _safe_serialize_raw(raw)}
                except Exception as e:
                    logger.exception("LangChain wrapper invoke failed, falling back: %s", e)
                    # fall through to SDK/REST

            # 2) genai SDK path
            if self._use_genai_sdk and self._genai is not None:
                try:
                    gen = self._genai
                    # attempt to call a common generate_text API shape
                    try:
                        resp = gen.generate_text(model=norm_model, prompt=prompt, max_output_tokens=int(max_tokens))
                    except Exception:
                        # some SDK versions have different function signatures; try a generic call
                        resp = gen.generate(prompt, model=norm_model, max_output_tokens=int(max_tokens))

                    text = None
                    raw = resp
                    if isinstance(resp, dict):
                        raw = resp
                        for key in ("output", "text", "content", "candidates"):
                            if key in resp:
                                val = resp[key]
                                if isinstance(val, str):
                                    text = val
                                    break
                                if isinstance(val, list) and val:
                                    if isinstance(val[0], dict):
                                        text = val[0].get("output") or val[0].get("content") or val[0].get("text")
                                        if text:
                                            break
                                    else:
                                        text = " ".join(map(str, val))
                                        break
                    else:
                        text = getattr(resp, "text", None) or getattr(resp, "output", None)
                        if not text:
                            try:
                                raw_json = json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
                                raw = raw_json
                                for key in ("output", "text", "content"):
                                    if key in raw_json:
                                        v = raw_json[key]
                                        text = v if isinstance(v, str) else " ".join(map(str, v)) if isinstance(v, list) else None
                                        if text:
                                            break
                            except Exception:
                                pass

                    if text is None:
                        text = str(raw)
                    runtime = time.time() - start
                    logger.info("GeminiClient (genai SDK) completed in %.3fs", runtime)
                    return {"text": str(text), "raw": _safe_serialize_raw(raw)}
                except Exception as e:
                    logger.exception("genai SDK call failed; falling back to REST: %s", e)
                    # fall through to REST

            # 3) REST fallback with requests.Session + retry
            base = self.endpoint.rstrip("/") if self.endpoint else "https://generativelanguage.googleapis.com/v1beta2"

            # model_path already includes 'models/' or 'tunedModels/' where appropriate
            url = f"{base}/{model_path}:generate"
            payload = {"input": prompt, "max_output_tokens": int(max_tokens)}

            # Headers and params: support both Bearer token (Authorization) and plain API key (query param)
            headers = {"Content-Type": "application/json"}
            params = {}
            if self.api_key:
                # Heuristic: if api_key looks like an OAuth access token or starts with 'Bearer ' use Authorization header.
                ak = str(self.api_key).strip()
                if ak.lower().startswith("bearer ") or ak.startswith("ya29.") or ak.count(".") >= 2:
                    # Common Google access tokens start with 'ya29.'; also accept explicit 'Bearer ' prefix
                    token = ak.replace("Bearer ", "").strip()
                    headers["Authorization"] = f"Bearer {token}"
                else:
                    # Otherwise send as query param (API key)
                    params["key"] = ak

            # local import to allow mocking in tests
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            session = requests.Session()
            retries = Retry(
                total=_DEFAULT_RETRIES,
                backoff_factor=_DEFAULT_BACKOFF_FACTOR,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=frozenset(["POST", "GET"]),
            )
            session.mount("https://", HTTPAdapter(max_retries=retries))
            session.mount("http://", HTTPAdapter(max_retries=retries))

            if self.tracing:
                logger.info("GeminiClient (REST) POST %s params=%s payload_len=%d headers=%s", url, params, len(json.dumps(payload)), list(headers.keys()))

            resp = session.post(url, json=payload, headers=headers, params=params, timeout=timeout)

            if resp.status_code >= 400:
                # include response body in logs for debugging
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                logger.error("GeminiClient REST error status=%s url=%s params=%s detail=%s", resp.status_code, url, params, detail)
                raise RuntimeError(f"Gemini API error: {resp.status_code} {detail}")

            try:
                data = resp.json()
            except Exception:
                data = {"text": resp.text}

            text = None
            raw = data
            if isinstance(data, dict):
                # support multiple possible response shapes
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
                # older SDK/REST shapes sometimes have 'outputs' array
                if text is None and "outputs" in data and isinstance(data["outputs"], list) and data["outputs"]:
                    first = data["outputs"][0]
                    if isinstance(first, dict):
                        text = first.get("text") or first.get("content") or first.get("output")
            if text is None:
                text = json.dumps(data)

            runtime = time.time() - start
            logger.info("GeminiClient (REST) completed in %.3fs", runtime)
            return {"text": str(text), "raw": _safe_serialize_raw(raw)}

        except CustomException:
            raise
        except Exception as e:
            logger.exception("GeminiClient.generate failed: %s", e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("GeminiClient demo")
    try:
        c = GeminiClient()
        print("Initialized GeminiClient. Default model:", c.default_model)
        print("Call generate() only if GEMINI_API_KEY is set to avoid network calls.")
    except Exception as e:
        print("Init failed:", e)
        sys.exit(1)
