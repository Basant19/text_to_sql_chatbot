# app/graph/nodes/prompt_node.py

import sys
from typing import Dict, Any, List, Optional, Callable

from app.logger import get_logger
from app.exception import CustomException
from app import utils
from app.tools import Tools

logger = get_logger("prompt_node")


class PromptNode:
    """
    Node that builds a prompt for the LLM.

    Combines:
      - user_query: str
      - schemas: mapping table_name -> {"columns": [...], "sample_rows": [...]}
      - retrieved_docs: list of {'id','score','text','meta'}
      - few_shot: optional list of examples

    Optional dependency injection:
      - prompt_builder: callable to build prompt (defaults to utils.build_prompt)
    """

    def __init__(self, tools: Optional[Tools] = None, prompt_builder: Optional[Callable[..., str]] = None):
        try:
            self._tools = tools or Tools()
            self._builder = prompt_builder or utils.build_prompt
        except Exception as e:
            logger.exception("Failed to initialize PromptNode")
            raise CustomException(e, sys)

    def run(
        self,
        user_query: str,
        schemas: Dict[str, Dict[str, Any]],
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        few_shot: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Build and return the prompt string.

        Parameters
        ----------
        user_query : str
            User's natural language query.
        schemas : Dict[str, Dict[str, Any]]
            CSV/table schemas and sample rows.
        retrieved_docs : Optional[List[Dict[str, Any]]]
            Retrieved RAG documents.
        few_shot : Optional[List[Dict[str, str]]]
            Few-shot examples to include.

        Returns
        -------
        str
            Prompt string ready for LLM generation.
        """
        try:
            retrieved_docs = retrieved_docs or []

            # Use injected builder function
            prompt = self._builder(user_query, schemas, retrieved_docs, few_shot)
            logger.info("PromptNode: Prompt built successfully")
            return prompt
        except CustomException:
            raise
        except Exception as e:
            logger.exception("PromptNode.run failed")
            raise CustomException(e, sys)
