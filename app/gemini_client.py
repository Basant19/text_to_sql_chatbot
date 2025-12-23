# D:\text_to_sql_bot\app\gemini_client.py
import sys
import json
import time
from typing import Any, Dict, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("gemini_client")

# Config-driven defaults
_DEFAULT_TIMEOUT = getattr(config, "GEMINI_REST_TIMEOUT", 30)
_DEFAULT_RETRIES = getattr(config, "GEMINI_REST_RETRIES", 2)
_DEFAULT_BACKOFF_FACTOR = getattr(config, "GEMINI_REST_BACKOFF", 0.5)
_DEFAULT_MAX_RAW_PREVIEW = getattr(config, "MAX_RAW_SUMMARY_CHARS", 800)


def _safe_serialize_raw(raw: Any) -> Any:
    """Safely serialize raw provider output."""
    try:
        return json.loads(json.dumps(raw, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        try:
            s = str(raw)
            return s[:_DEFAULT_MAX_RAW_PREVIEW] + "…" if len(s) > _DEFAULT_MAX_RAW_PREVIEW else s
        except Exception:
            return "<unserializable raw response>"


class GeminiClient:
    """
    Adapter for Google Gemini.

    Supports:
      - LangChain ChatGoogleGenerativeAI
      - google.generativeai SDK
      - REST fallback

    Public API:
      generate(prompt, model=None, max_tokens=..., timeout=...)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,          # ✅ ACCEPTS `model`
        default_model: Optional[str] = None,  # ✅ BACKWARD COMPAT
        endpoint: Optional[str] = None,
        tracing: Optional[bool] = None,
    ):
        try:
            # --- Resolve configuration ---
            self.api_key = (
                api_key
                or getattr(config, "GEMINI_API_KEY", None)
                or getattr(config, "GOOGLE_API_KEY", None)
            )

            # `model` takes precedence over `default_model`
            self.default_model = (
                model
                or default_model
                or getattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
            )

            self.endpoint = endpoint or getattr(config, "GEMINI_ENDPOINT", "")
            self.tracing = tracing if tracing is not None else getattr(config, "LANGSMITH_TRACING", False)

            if not self.api_key:
                logger.warning("GeminiClient initialized without API key")

            # --- LangChain wrapper ---
            self._use_langchain_llm = False
            self._llm = None

            try:
                ChatGoogleGenerativeAI = None
                for path in (
                    "langchain_google_genai",
                    "langchain.google_genai",
                    "langchain.chat_models",
                ):
                    try:
                        module = __import__(path, fromlist=["ChatGoogleGenerativeAI"])
                        ChatGoogleGenerativeAI = getattr(module, "ChatGoogleGenerativeAI", None)
                        if ChatGoogleGenerativeAI:
                            break
                    except Exception:
                        continue

                if ChatGoogleGenerativeAI:
                    try:
                        self._llm = ChatGoogleGenerativeAI(
                            model=self.default_model,
                            google_api_key=self.api_key,
                        )
                    except TypeError:
                        self._llm = ChatGoogleGenerativeAI(
                            model=self.default_model,
                            api_key=self.api_key,
                        )

                    self._use_langchain_llm = True
                    logger.info("GeminiClient using LangChain ChatGoogleGenerativeAI")
            except Exception as e:
                logger.debug("LangChain Gemini wrapper unavailable: %s", e)

            # --- google.generativeai SDK ---
            self._use_genai_sdk = False
            self._genai = None

            if not self._use_langchain_llm:
                try:
                    import google.generativeai as genai  # type: ignore
                    genai.configure(api_key=self.api_key)
                    self._genai = genai
                    self._use_genai_sdk = True
                    logger.info("GeminiClient using google.generativeai SDK")
                except Exception as e:
                    logger.debug("google.generativeai SDK unavailable: %s", e)

            logger.info(
                "GeminiClient initialized (model=%s, langchain=%s, genai=%s)",
                self.default_model,
                self._use_langchain_llm,
                self._use_genai_sdk,
            )

        except Exception as e:
            logger.exception("Failed to initialize GeminiClient")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate text using Gemini.

        Returns:
            {"text": str, "raw": json-safe}
        """
        start = time.time()
        timeout = timeout or _DEFAULT_TIMEOUT
        model = model or self.default_model

        try:
            # --- LangChain path ---
            if self._use_langchain_llm and self._llm:
                resp = self._llm.invoke(prompt)
                text = getattr(resp, "content", None) or str(resp)
                return {
                    "text": str(text),
                    "raw": _safe_serialize_raw(resp),
                }

            # --- SDK path ---
            if self._use_genai_sdk and self._genai:
                resp = self._genai.generate_text(
                    model=f"models/{model}",
                    prompt=prompt,
                    max_output_tokens=max_tokens,
                )
                text = getattr(resp, "result", None) or str(resp)
                return {
                    "text": str(text),
                    "raw": _safe_serialize_raw(resp),
                }

            raise RuntimeError("No Gemini backend available")

        except Exception as e:
            logger.exception("GeminiClient.generate failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    c = GeminiClient(model="gemini-2.5-flash")
    print("GeminiClient OK:", c.default_model)
