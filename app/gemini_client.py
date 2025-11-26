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
    Adapter for Google Gemini (Generative Models).

    Preferred flow (in order):
      1. LangChain's ChatGoogleGenerativeAI wrapper (if `langchain-google-genai` or compatible package is installed).
         This mirrors your snippet:
             llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', google_api_key=api_key)
             resp = llm.invoke(prompt)
             llm_output = resp.content

      2. google.generativeai SDK (if installed) as a fallback.

      3. REST fallback to the Generative Language HTTP API.

    The adapter normalizes responses to:
        {"text": <str>, "raw": <any>}
    so the rest of the app can be provider-agnostic.

    Important:
      - Reads API key and model defaults from app.config (GEMINI_API_KEY, GEMINI_MODEL).
      - Does not use LangSmith for generation. Tracing/logging is separate.
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
            self.api_key = api_key or getattr(config, "GEMINI_API_KEY", None)
            self.endpoint = endpoint or getattr(config, "GEMINI_ENDPOINT", "")
            self.default_model = default_model or getattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
            self.tracing = tracing if tracing is not None else getattr(config, "LANGSMITH_TRACING", False)

            if not self.api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for GeminiClient")

            # Try to initialize preferred LangChain Chat wrapper (ChatGoogleGenerativeAI) first.
            self._use_langchain_llm = False
            self._llm = None
            try:
                # Try a few import paths that users may have depending on installation
                try:
                    # preferred package: langchain-google-genai shim
                    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
                except Exception:
                    # alternative: some installations expose it under langchain.google_genai or langchain.chat_models
                    try:
                        from langchain.google_genai import ChatGoogleGenerativeAI  # type: ignore
                    except Exception:
                        # final fallback: try langchain.chat_models (may not exist)
                        from langchain.chat_models import ChatGoogleGenerativeAI  # type: ignore

                # instantiate LLM wrapper (it typically expects google_api_key or api_key kwarg)
                try:
                    # many wrappers accept google_api_key or api_key; try both safe ways
                    self._llm = ChatGoogleGenerativeAI(model=self.default_model, google_api_key=self.api_key)
                except TypeError:
                    # fallback if parameter name differs
                    self._llm = ChatGoogleGenerativeAI(model=self.default_model, api_key=self.api_key)
                self._use_langchain_llm = True
                logger.info("GeminiClient: using ChatGoogleGenerativeAI (LangChain wrapper) as primary backend")
            except Exception as e:
                self._use_langchain_llm = False
                self._llm = None
                logger.info("GeminiClient: ChatGoogleGenerativeAI not available or failed to initialize: %s", e)

            # If LangChain wrapper not available, try google.generativeai SDK
            self._use_genai_sdk = False
            self._genai = None
            if not self._use_langchain_llm:
                try:
                    try:
                        import google.generativeai as genai  # type: ignore
                    except Exception:
                        import generativeai as genai  # type: ignore
                    # configure SDK (many versions provide `configure` or set API key)
                    try:
                        genai.configure(api_key=self.api_key)
                    except Exception:
                        # some SDKs expect a top-level variable
                        try:
                            genai.api_key = self.api_key  # type: ignore
                        except Exception:
                            pass
                    self._genai = genai
                    self._use_genai_sdk = True
                    logger.info("GeminiClient: using google.generativeai SDK as secondary backend")
                except Exception as e:
                    self._use_genai_sdk = False
                    self._genai = None
                    logger.info("GeminiClient: google.generativeai SDK not available: %s", e)

            # If neither backends available, we'll use REST fallback when generate() is called.
            logger.info(
                "GeminiClient initialized (model=%s, endpoint=%s, langchain_llm=%s, genai_sdk=%s)",
                self.default_model,
                self.endpoint or "default",
                self._use_langchain_llm,
                self._use_genai_sdk,
            )
        except Exception as e:
            logger.exception("Failed to initialize GeminiClient")
            raise CustomException(e, sys)

    # ---------------------------
    # Model name normalization
    # ---------------------------
    def _normalize_model_name(self, model: str) -> str:
        """
        Ensure model string conforms to API expectation:
        - If model already starts with 'models/' or 'tunedModels/', return as-is.
        - Otherwise prefix with 'models/'.
        """
        try:
            if not isinstance(model, str) or not model:
                return model
            if model.startswith("models/") or model.startswith("tunedModels/"):
                return model
            return f"models/{model}"
        except Exception:
            return model

    # ---------------------------
    # Generate
    # ---------------------------
    def generate(self, prompt: str, model: Optional[str] = None, max_tokens: int = 512, timeout: int = 30) -> Dict[str, Any]:
        """
        Generate text from the configured Gemini provider.

        Returns:
            {"text": <str>, "raw": <raw response>}
        Raises:
            CustomException on unexpected errors.
        """
        start = time.time()
        try:
            model = model or self.default_model
            # Normalize model name for SDK & REST compatibility
            norm_model = self._normalize_model_name(model)

            # 1) Preferred LangChain Chat wrapper path (mirrors your snippet)
            if self._use_langchain_llm and self._llm is not None:
                try:
                    # The wrapper exposes an `invoke` method in your snippet.
                    # Accept either a simple string prompt or a list of messages (future extension).
                    resp = self._llm.invoke(prompt)
                    # Many wrappers return an object with `.content` or `.generations`. Normalize.
                    text = None
                    raw = resp

                    # common property `content` as per your snippet
                    text = getattr(resp, "content", None)
                    if text is None:
                        # try attributes used by some wrappers
                        text = getattr(resp, "text", None)
                    if text is None:
                        # sometimes resp has `.generations` or `.generations[0].text`
                        gens = getattr(resp, "generations", None)
                        if isinstance(gens, (list, tuple)) and len(gens) > 0:
                            first = gens[0]
                            text = getattr(first, "text", None) or getattr(first, "content", None)
                            if text is None and isinstance(first, dict):
                                text = first.get("text") or first.get("content")
                    if text is None:
                        # best-effort extraction for dict-like
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
                                        # if candidates list of dicts with 'content' or 'output'
                                        if isinstance(val[0], dict):
                                            text = val[0].get("content") or val[0].get("output") or val[0].get("text")
                                        else:
                                            text = " ".join(map(str, val))
                                        break
                        except Exception:
                            # as a final fallback, stringify resp
                            text = str(resp)

                    runtime = time.time() - start
                    logger.info("GeminiClient (LangChain wrapper) completed in %.3fs", runtime)
                    return {"text": str(text), "raw": raw}
                except Exception as e:
                    # If wrapper fails, log and fall through to SDK/REST fallback
                    logger.exception("GeminiClient: ChatGoogleGenerativeAI invoke failed, falling back: %s", e)

            # 2) google.generativeai SDK path
            if self._use_genai_sdk and self._genai is not None:
                try:
                    gen = self._genai
                    # Attempt SDK call with normalized model name
                    try:
                        resp = gen.generate_text(model=norm_model, prompt=prompt, max_output_tokens=int(max_tokens))
                    except ValueError as ve:
                        # Some genai versions raise ValueError complaining about model name format.
                        # Retry explicitly with normalized model name (defensive).
                        logger.warning("genai SDK ValueError (maybe model-name format). Retrying with normalized model: %s; error: %s", norm_model, ve)
                        resp = gen.generate_text(model=norm_model, prompt=prompt, max_output_tokens=int(max_tokens))
                    except Exception as inner_e:
                        # Log and re-raise to trigger REST fallback below
                        logger.exception("GeminiClient: genai.generate_text inner exception, will fall back to REST: %s", inner_e)
                        raise

                    # extract textual content
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
                                        # candidate objects
                                        text = val[0].get("output") or val[0].get("content") or val[0].get("text")
                                        if text:
                                            break
                                    else:
                                        text = " ".join(map(str, val))
                                        break
                    else:
                        # object-like response; try attribute access
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
                    return {"text": str(text), "raw": raw}
                except Exception as e:
                    logger.exception("GeminiClient: genai.generate_text failed; falling back to REST: %s", e)
                    # fall-through to REST

            # 3) REST fallback
            base = self.endpoint.rstrip("/") if self.endpoint else "https://generativelanguage.googleapis.com/v1beta2"
            # Use normalized model in the path (the API expects the model path component, e.g., "models/<id>")
            url = f"{base}/models/{norm_model}:generate"

            payload = {
                "input": prompt,
                "max_output_tokens": int(max_tokens),
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            if self.tracing:
                logger.info("GeminiClient (REST) POST %s", url)
                try:
                    logger.debug("Payload preview: %s", json.dumps(payload)[:1000])
                except Exception:
                    logger.debug("Payload (non-serializable)")

            import requests  # local import for easier test/mocking

            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code >= 400:
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                logger.error("GeminiClient REST call failed: status=%s detail=%s", resp.status_code, detail)
                raise RuntimeError(f"Gemini API error: {resp.status_code} {detail}")

            try:
                data = resp.json()
            except Exception:
                data = {"text": resp.text}

            # Heuristic extraction from REST response shapes
            text = None
            raw = data
            if isinstance(data, dict):
                # candidate / output patterns
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
                if text is None and "outputs" in data and isinstance(data["outputs"], list) and data["outputs"]:
                    first = data["outputs"][0]
                    if isinstance(first, dict):
                        # some shapes use 'text' inside outputs
                        text = first.get("text") or first.get("content") or first.get("output")
            if text is None:
                text = json.dumps(data)

            runtime = time.time() - start
            logger.info("GeminiClient (REST) completed in %.3fs", runtime)
            return {"text": str(text), "raw": raw}
        except CustomException:
            raise
        except Exception as e:
            logger.exception("GeminiClient.generate failed: %s", e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Simple CLI/demo: do not call network unless user has configured GEMINI_API_KEY
    print("GeminiClient demo")
    try:
        c = GeminiClient()
        print("Initialized GeminiClient. Default model:", c.default_model)
        print("Call generate() only if GEMINI_API_KEY is set to avoid network calls.")
    except Exception as e:
        print("Init failed:", e)
        sys.exit(1)
