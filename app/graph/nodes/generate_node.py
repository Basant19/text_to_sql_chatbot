# app/graph/nodes/generate_node.py
import sys
import time
from typing import Optional, Dict, Any, Union

from app.logger import get_logger
from app.exception import CustomException
from app.langsmith_client import LangSmithClient
from app import utils

logger = get_logger("generate_node")


class GenerateNode:
    """
    Node that calls an LLM to generate SQL from a prompt.

    Supports two backends:
      1. LangSmithClient-like object with generate(prompt, model, max_tokens)
      2. LangGraph agent with run(prompt) returning tuple or dict

    LangGraph agent is preferred when available.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = "gpt",
        max_tokens: int = 512
    ):
        try:
            # client can be LangSmithClient or LangGraph agent
            self.client = client or LangSmithClient()
            self.model = model
            self.max_tokens = max_tokens
        except Exception as e:
            logger.exception("Failed to initialize GenerateNode")
            raise CustomException(e, sys)

    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Generate SQL from a prompt using the client or agent.

        Parameters
        ----------
        prompt : str
            The text prompt describing the desired SQL.

        Returns
        -------
        Dict[str, Any]
            {
                "prompt": <original or adjusted prompt>,
                "raw": <raw response from client/agent>,
                "sql": <extracted SQL string>
            }

        Raises
        ------
        CustomException
            If generation fails or client/agent is invalid.
        """
        start_time = time.time()
        try:
            raw_resp: Union[Dict[str, Any], str, None] = None
            sql: str = ""
            prompt_text: str = prompt

            # Prefer LangGraph agent if it has a 'run' method
            if hasattr(self.client, "run") and callable(self.client.run):
                logger.info("GenerateNode: using LangGraph agent run()")
                out = self.client.run(prompt)

                if isinstance(out, tuple):
                    # assume (sql, prompt_text, raw_resp)
                    sql, prompt_text, raw_resp = out
                elif isinstance(out, dict):
                    sql = out.get("sql") or utils.extract_sql_from_text(out.get("text", "")) or ""
                    prompt_text = out.get("prompt", prompt)
                    raw_resp = out
                else:
                    raw_resp = out
                    sql = utils.extract_sql_from_text(str(out)) or str(out).strip()

            # Fall back to LangSmithClient
            elif hasattr(self.client, "generate") and callable(self.client.generate):
                logger.info("GenerateNode: using LangSmithClient generate()")
                out = self.client.generate(prompt, model=self.model, max_tokens=self.max_tokens)

                if isinstance(out, dict):
                    raw_resp = out
                    sql = (
                        utils.extract_sql_from_text(out.get("text", ""))
                        or out.get("output", "")
                        or ""
                    )
                else:
                    raw_resp = out
                    sql = utils.extract_sql_from_text(str(out)) or str(out).strip()
            else:
                raise CustomException(
                    "Client must implement 'run' (LangGraph agent) or 'generate' (LangSmithClient)",
                    sys
                )

            runtime = time.time() - start_time
            logger.info(f"GenerateNode: completed in {runtime:.3f}s, sql_len={len(sql)}")

            return {
                "prompt": prompt_text,
                "raw": raw_resp,
                "sql": sql
            }

        except CustomException:
            raise
        except Exception as e:
            logger.exception("GenerateNode.run failed")
            raise CustomException(e, sys)
