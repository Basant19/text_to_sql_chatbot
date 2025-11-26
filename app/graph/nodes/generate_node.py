# app/graph/nodes/generate_node.py

import sys
import time
import threading
from typing import Optional, Dict, Any, Union

from app.logger import get_logger
from app.exception import CustomException
from app import utils, config
from app.tools import Tools

logger = get_logger("generate_node")


class GenerateNode:
    """
    Node responsible for generating SQL from a prompt via an LLM.

    Key behaviors:
    - Supports agent-based or direct provider generation.
    - Observability via a tracer client (LangSmith), fire-and-forget.
    - Strict safeguards: tracer cannot act as generator unless explicitly enabled via config.
    """

    def __init__(
        self,
        tools: Optional[Tools] = None,
        client: Optional[Any] = None,
        provider_client: Optional[Any] = None,
        tracer_client: Optional[Any] = None,
        model: Optional[str] = None,
        max_tokens: int = 512,
    ):
        try:
            self.tools = tools or Tools()
            self.client = client
            self.provider_client = provider_client
            self.tracer_client = tracer_client
            self.model = model or getattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
            self.max_tokens = max_tokens

            logger.info(
                "GenerateNode initialized (model=%s, max_tokens=%s, has_provider=%s, has_tracer=%s)",
                self.model,
                self.max_tokens,
                bool(self.provider_client or self.client),
                bool(self.tracer_client),
            )
        except Exception as e:
            logger.exception("Failed to initialize GenerateNode")
            raise CustomException(e, sys)

    def _call_tracer(self, prompt: str, sql: Optional[str] = None) -> None:
        """
        Fire-and-forget tracer call for observability.
        Exceptions are logged but do not affect generation.
        """
        if not self.tracer_client:
            return

        def _trace():
            try:
                if hasattr(self.tracer_client, "trace_run"):
                    self.tracer_client.trace_run(
                        name="generate_node.run",
                        prompt=prompt,
                        sql=sql,
                        metadata={
                            "model": self.model,
                            "max_tokens": self.max_tokens,
                            "timestamp": time.time(),
                        },
                    )
                    return

                # fallback: tracer.generate only if explicitly enabled
                if getattr(config, "USE_LANGSMITH_FOR_GEN", False) and hasattr(self.tracer_client, "generate"):
                    logger.warning(
                        "Tracer.generate() called (USE_LANGSMITH_FOR_GEN=true). Not recommended for production."
                    )
                    try:
                        self.tracer_client.generate(prompt, model=self.model, max_tokens=1)
                    except TypeError:
                        self.tracer_client.generate(prompt)
            except Exception as e:
                logger.debug("Tracer error ignored: %s", e)

        threading.Thread(target=_trace, daemon=True).start()

    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Generate SQL from prompt.

        Returns:
        {
            "prompt": <prompt used>,
            "raw": <raw provider/agent output>,
            "sql": <extracted SQL string>
        }
        """
        start_time = time.time()
        try:
            raw_resp: Union[Dict[str, Any], str, None] = None
            sql: str = ""
            prompt_text: str = prompt

            # 1) Use agent if available
            if self.client and hasattr(self.client, "run"):
                logger.info("GenerateNode: using agent client run()")
                out = self.client.run(prompt)

                if isinstance(out, tuple):
                    sql, prompt_text, raw_resp = out
                elif isinstance(out, dict):
                    prompt_text = out.get("prompt", prompt)
                    raw_resp = out
                    sql = out.get("sql") or utils.extract_sql_from_text(out.get("text", "")) or ""
                else:
                    raw_resp = out
                    sql = utils.extract_sql_from_text(str(out)) or str(out).strip()

                self._call_tracer(prompt_text, sql=sql)

            else:
                # 2) Determine provider
                provider = self.provider_client or self.client
                if provider is None:
                    from app.gemini_client import GeminiClient
                    provider = GeminiClient()
                    logger.debug("GenerateNode: lazily instantiated GeminiClient")

                if not hasattr(provider, "generate") or not callable(provider.generate):
                    raise CustomException("Provider must implement generate(prompt, model, max_tokens)", sys)

                logger.info("GenerateNode: calling provider.generate() (model=%s)", self.model)
                out = provider.generate(prompt, model=self.model, max_tokens=self.max_tokens)
                raw_resp = out

                if isinstance(out, dict):
                    resp_text = out.get("text") or out.get("output") or out.get("content") or ""
                    sql = utils.extract_sql_from_text(resp_text) or str(resp_text).strip()
                else:
                    sql = utils.extract_sql_from_text(str(out)) or str(out).strip()

                self._call_tracer(prompt, sql=sql)

            runtime = time.time() - start_time
            logger.info("GenerateNode completed in %.3fs, sql_len=%d", runtime, len(sql or ""))
            return {"prompt": prompt_text, "raw": raw_resp, "sql": sql}

        except CustomException:
            raise
        except Exception as e:
            logger.exception("GenerateNode.run failed")
            raise CustomException(e, sys)
