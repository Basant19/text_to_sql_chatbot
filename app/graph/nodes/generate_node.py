# app/graph/nodes/generate_node.py
import sys
import time
import threading
from typing import Optional, Dict, Any, Union

from app.logger import get_logger
from app.exception import CustomException
from app import utils

logger = get_logger("generate_node")


class GenerateNode:
    """
    Node that calls an LLM to generate SQL from a prompt.

    Behavior:
      - If `client` has a .run() method (LangGraph agent), that is used and its output
        is trusted (keeps previous behavior for agents).
      - Otherwise, generation is done via a provider client that implements
        generate(prompt, model, max_tokens). The provider client can be passed via
        `provider_client` (preferred) or `client` (legacy).
      - Optionally a `tracer_client` (e.g. LangSmithClient) may be provided. It will be
        called for observability only (we call it in a background thread and ignore output).
        Any tracer errors are caught and logged — they won't affect inference.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        provider_client: Optional[Any] = None,
        tracer_client: Optional[Any] = None,
        model: str = "gpt",
        max_tokens: int = 512,
    ):
        try:
            # `client` kept for backward compatibility: it may be an agent or a provider.
            self.client = client
            # provider_client is explicit LLM provider (GeminiClient, etc.)
            self.provider_client = provider_client
            # tracer_client is optional (LangSmithClient) used for observability only
            self.tracer_client = tracer_client

            self.model = model
            self.max_tokens = max_tokens

            logger.info(
                f"GenerateNode initialized (model={self.model}, max_tokens={self.max_tokens}, "
                f"has_provider={bool(self.provider_client or self.client)}, has_tracer={bool(self.tracer_client)})"
            )
        except Exception as e:
            logger.exception("Failed to initialize GenerateNode")
            raise CustomException(e, sys)

    def _call_tracer(self, prompt: str, model: str, max_tokens: int, sql: Optional[str] = None) -> None:
        """
        Call the tracer client (LangSmith) for observability in a fire-and-forget manner.
        This spawns a daemon thread so tracing does not add latency to inference.

        Preferred tracer interface: tracer_client.trace_run(name, prompt, sql, metadata)
        Fallback (not recommended): tracer_client.generate(...) — kept only as a last-resort
        and will be ignored if tracer does not implement trace_run.

        Any exception in tracing is logged and ignored.
        """
        if not self.tracer_client:
            return

        def _do_trace():
            try:
                # Prefer dedicated trace_run (no inference)
                if hasattr(self.tracer_client, "trace_run") and callable(self.tracer_client.trace_run):
                    try:
                        self.tracer_client.trace_run(
                            name="generate_node.run",
                            prompt=prompt,
                            sql=sql,
                            metadata={"model": model, "max_tokens": max_tokens},
                        )
                    except Exception as e:
                        logger.warning(f"Tracer.trace_run() failed (ignored): {e}")
                    return

                # Fallback: very lightweight generate() call (max_tokens=1) if trace_run not available
                if hasattr(self.tracer_client, "generate") and callable(self.tracer_client.generate):
                    try:
                        self.tracer_client.generate(prompt, model=model, max_tokens=1)
                    except Exception as e:
                        logger.warning(f"Tracer.generate() fallback failed (ignored): {e}")
            except Exception as e:
                logger.debug(f"Tracer background task encountered error (ignored): {e}")

        try:
            t = threading.Thread(target=_do_trace, daemon=True)
            t.start()
        except Exception as e:
            logger.debug(f"Failed to start tracer thread (ignored): {e}")

    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Generate SQL from a prompt.

        Returns dict:
          {
            "prompt": <prompt used or agent-provided prompt>,
            "raw": <raw response from provider or agent (if any)>,
            "sql": <extracted SQL string>
          }
        """
        start_time = time.time()
        try:
            raw_resp: Union[Dict[str, Any], str, None] = None
            sql: str = ""
            prompt_text: str = prompt

            # 1) If client is an agent (has run), prefer it (keeps existing behavior)
            if self.client and hasattr(self.client, "run") and callable(self.client.run):
                logger.info("GenerateNode: using LangGraph agent run()")
                out = self.client.run(prompt)

                if isinstance(out, tuple):
                    # assume (sql, prompt_text, raw_resp)
                    sql, prompt_text, raw_resp = out
                elif isinstance(out, dict):
                    prompt_text = out.get("prompt", prompt)
                    raw_resp = out
                    sql = out.get("sql") or utils.extract_sql_from_text(out.get("text", "")) or ""
                else:
                    raw_resp = out
                    sql = utils.extract_sql_from_text(str(out)) or str(out).strip()

                # fire tracer asynchronously (do not block)
                try:
                    self._call_tracer(prompt_text, self.model, self.max_tokens, sql=sql)
                except Exception:
                    logger.debug("Tracer call after agent run failed (ignored)")

            else:
                # 2) Determine provider client (explicit provider_client preferred)
                provider = self.provider_client or self.client
                # If still None, try to import GeminiClient lazily (best-effort)
                if provider is None:
                    try:
                        from app.gemini_client import GeminiClient  # type: ignore
                        provider = GeminiClient()
                        logger.debug("GenerateNode: lazily instantiated GeminiClient as provider")
                    except Exception:
                        provider = None

                if provider is None:
                    raise CustomException("No LLM provider configured (provider_client or client required)", sys)

                # Ensure provider has generate()
                if not hasattr(provider, "generate") or not callable(provider.generate):
                    raise CustomException("Provider client must implement generate(prompt, model, max_tokens)", sys)

                # 3) Call provider to get generation (this is the real inference call)
                logger.info("GenerateNode: calling provider.generate() for inference")
                out = provider.generate(prompt, model=self.model, max_tokens=self.max_tokens)
                raw_resp = out

                # extract SQL from provider response
                if isinstance(out, dict):
                    # common field for text is 'text'; allow fallback to 'output'
                    resp_text = out.get("text") or out.get("output") or ""
                    sql = utils.extract_sql_from_text(resp_text) or str(resp_text).strip()
                else:
                    sql = utils.extract_sql_from_text(str(out)) or str(out).strip()

                # 4) Fire-and-forget tracer (LangSmith) for observability only.
                try:
                    self._call_tracer(prompt, self.model, self.max_tokens, sql=sql)
                except Exception:
                    logger.debug("Tracer call after provider.generate failed (ignored)")

            runtime = time.time() - start_time
            logger.info(f"GenerateNode: completed in {runtime:.3f}s, sql_len={len(sql)}")

            return {"prompt": prompt_text, "raw": raw_resp, "sql": sql}

        except CustomException:
            raise
        except Exception as e:
            logger.exception("GenerateNode.run failed")
            raise CustomException(e, sys)
