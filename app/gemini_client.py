#D:\text_to_sql_bot\app\gemini_client.py
import sys
import json
import time
from typing import Any, Dict, Optional

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

    Returns a dict: {"text": str, "raw": json-serializable}
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

            if not self.api_key:
                logger.warning("GeminiClient initialized without API key; REST/SDK calls will fail if invoked.")

            # Try LangChain Chat wrapper
            self._use_langchain_llm = False
            self._llm = None
            try:
                # Try several import paths to support different langchain versions
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
                except Exception:
                    try:
                        from langchain.google_genai import ChatGoogleGenerativeAI  # type: ignore
                    except Exception:
                        try:
                            from langchain.chat_models import ChatGoogleGenerativeAI  # type: ignore
                        except Exception:
                            ChatGoogleGenerativeAI = None

                if ChatGoogleGenerativeAI:
                    try:
                        # wrapper often expects google_api_key kwarg
                        self._llm = ChatGoogleGenerativeAI(model=self.default_model, google_api_key=self.api_key)
                    except TypeError:
                        # some wrappers expect `api_key` or `openai_api_key` style
                        try:
                            self._llm = ChatGoogleGenerativeAI(model=self.default_model, api_key=self.api_key)
                        except Exception:
                            # fallback to model-only init
                            self._llm = ChatGoogleGenerativeAI(model=self.default_model)

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

                    try:
                        # preferred configure pattern
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

    def _normalize_model_name(self, model: Optional[str]) -> str:
        try:
            if not isinstance(model, str) or not model:
                return str(model or "")
            m = model
            # If user passed "models/xyz" keep as-is, otherwise prefix
            if m.startswith("models/") or m.startswith("tunedModels/"):
                return m
            return f"models/{m}"
        except Exception:
            return str(model or "")

    def _build_model_path(self, norm_model: str) -> str:
        if not isinstance(norm_model, str) or not norm_model:
            return norm_model
        if norm_model.startswith("models/") or norm_model.startswith("tunedModels/"):
            return norm_model
        return f"models/{norm_model}"

    def _try_extract_text(self, resp: Any) -> str:
        """Try multiple heuristics to extract a text string from a provider response."""
        text = None
        # Direct attributes
        for attr in ("content", "text", "output", "message", "output_text"):
            try:
                v = getattr(resp, attr, None)
                if v:
                    text = v
                    break
            except Exception:
                continue

        # LangChain LLMResult -> .generations (list of lists or list of Generation objects)
        if text is None:
            try:
                gens = getattr(resp, "generations", None)
                if isinstance(gens, (list, tuple)) and gens:
                    first = gens[0]
                    # first can be list of Generation objects or a Generation object
                    if isinstance(first, (list, tuple)) and first:
                        candidate = first[0]
                    else:
                        candidate = first
                    # candidate may have .text or .content
                    text = getattr(candidate, "text", None) or getattr(candidate, "content", None) or (candidate.get("text") if isinstance(candidate, dict) else None)
            except Exception:
                pass

        # If it's a dict-like structure
        if text is None:
            try:
                raw_json = json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
                for key in ("text", "content", "output", "result", "outputs", "candidates"):
                    if key in raw_json:
                        val = raw_json[key]
                        if isinstance(val, str):
                            text = val
                            break
                        if isinstance(val, list) and val:
                            if isinstance(val[0], dict):
                                # try common nested keys
                                for k2 in ("text", "content", "output"):
                                    if k2 in val[0]:
                                        text = val[0].get(k2)
                                        break
                                if text:
                                    break
                            else:
                                text = " ".join(map(str, val))
                                break
            except Exception:
                pass

        # Final fallback
        if text is None:
            try:
                text = str(resp)
            except Exception:
                text = ""

        return str(text or "")

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

            # Sanity: ensure max_tokens is a positive integer
            try:
                max_tokens = int(max_tokens)
                if max_tokens <= 0:
                    max_tokens = 1
            except Exception:
                max_tokens = 512

            # Log prompt preview for debugging
            try:
                logger.info("GeminiClient.generate: prompt preview (len=%d): %s", len(prompt or ""), (prompt or "")[: _DEFAULT_MAX_RAW_PREVIEW].replace("\n", "\\n"))
            except Exception:
                logger.debug("GeminiClient.generate: prompt preview unavailable")

            # 1) LangChain wrapper path
            if self._use_langchain_llm and self._llm is not None:
                try:
                    # Try several common call patterns for langchain wrappers
                    resp = None
                    try:
                        # Some wrappers implement .generate or .generate_messages
                        if hasattr(self._llm, "generate"):
                            # build a minimal interface if required
                            try:
                                resp = self._llm.generate([{"content": prompt}])
                            except Exception:
                                resp = self._llm.generate(prompt)
                        elif hasattr(self._llm, "__call__"):
                            resp = self._llm(prompt)
                        elif hasattr(self._llm, "invoke"):
                            resp = self._llm.invoke(prompt)
                        else:
                            # Last resort: call as function
                            resp = self._llm(prompt)
                    except Exception:
                        # As a safe fallback try invoke
                        try:
                            resp = self._llm.invoke(prompt)
                        except Exception as inner:
                            logger.exception("LangChain wrapper call patterns failed: %s", inner)
                            raise

                    raw = resp
                    text = self._try_extract_text(resp)

                    runtime = time.time() - start
                    logger.info("GeminiClient (LangChain wrapper) completed in %.3fs text_len=%d", runtime, len(text or ""))
                    logger.debug("GeminiClient (LangChain wrapper) raw=%s", _safe_serialize_raw(raw))
                    return {"text": text, "raw": _safe_serialize_raw(raw)}
                except Exception as e:
                    logger.exception("LangChain wrapper invoke failed, falling back: %s", e)
                    # fall through to SDK/REST

            # 2) genai SDK path
            if self._use_genai_sdk and self._genai is not None:
                try:
                    gen = self._genai
                    resp = None
                    try:
                        # Try multiple known SDK call shapes
                        if hasattr(gen, "generate_text"):
                            resp = gen.generate_text(model=norm_model, prompt=prompt, max_output_tokens=max_tokens)
                        elif hasattr(gen, "generate"):
                            resp = gen.generate(prompt=prompt, model=norm_model, max_output_tokens=max_tokens)
                        else:
                            # best-effort call
                            resp = gen.generate(prompt, model=norm_model, max_output_tokens=max_tokens)
                    except Exception as inner:
                        logger.exception("genai SDK call shape failed: %s", inner)
                        raise

                    raw = resp
                    text = ""
                    try:
                        if isinstance(resp, dict):
                            for key in ("output", "text", "content", "candidates"):
                                if key in resp:
                                    val = resp[key]
                                    if isinstance(val, str):
                                        text = val
                                        break
                                    if isinstance(val, list) and val:
                                        if isinstance(val[0], dict):
                                            text = val[0].get("output") or val[0].get("content") or val[0].get("text") or ""
                                            if text:
                                                break
                                        else:
                                            text = " ".join(map(str, val))
                                            break
                        else:
                            text = self._try_extract_text(resp)
                    except Exception:
                        text = self._try_extract_text(resp)

                    if not text:
                        text = str(raw)

                    runtime = time.time() - start
                    logger.info("GeminiClient (genai SDK) completed in %.3fs text_len=%d", runtime, len(text or ""))
                    logger.debug("GeminiClient (genai SDK) raw=%s", _safe_serialize_raw(raw))
                    return {"text": text, "raw": _safe_serialize_raw(raw)}
                except Exception as e:
                    logger.exception("genai SDK call failed; falling back to REST: %s", e)
                    # fall through to REST

            # 3) REST fallback with requests.Session + retry
            base = self.endpoint.rstrip("/") if self.endpoint else "https://generativelanguage.googleapis.com/v1beta2"

            url = f"{base}/{model_path}:generate"
            payload = {"input": prompt, "max_output_tokens": int(max_tokens)}

            headers = {"Content-Type": "application/json"}
            params = {}
            if self.api_key:
                ak = str(self.api_key).strip()
                if ak.lower().startswith("bearer ") or ak.startswith("ya29.") or ak.count(".") >= 2:
                    token = ak.replace("Bearer ", "").strip()
                    headers["Authorization"] = f"Bearer {token}"
                else:
                    params["key"] = ak

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
                try:
                    logger.info("GeminiClient (REST) POST %s params=%s payload_len=%d headers=%s", url, params, len(json.dumps(payload)), list(headers.keys()))
                except Exception:
                    logger.info("GeminiClient (REST) POST %s", url)

            resp = session.post(url, json=payload, headers=headers, params=params, timeout=timeout)

            if resp.status_code >= 400:
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

            text = ""
            raw = data
            try:
                if isinstance(data, dict):
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
                    if not text and "outputs" in data and isinstance(data["outputs"], list) and data["outputs"]:
                        first = data["outputs"][0]
                        if isinstance(first, dict):
                            text = first.get("text") or first.get("content") or first.get("output") or ""
                if not text:
                    text = json.dumps(data)
            except Exception:
                text = str(data)

            runtime = time.time() - start
            logger.info("GeminiClient (REST) completed in %.3fs text_len=%d", runtime, len(text or ""))
            logger.debug("GeminiClient (REST) raw=%s", _safe_serialize_raw(raw))
            return {"text": str(text or ""), "raw": _safe_serialize_raw(raw)}

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
