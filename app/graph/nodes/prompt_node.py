# app/graph/nodes/prompt_node.py
import sys
from typing import Dict, Any, List, Optional, Callable

from app.logger import get_logger
from app.exception import CustomException
from app import utils

logger = get_logger("prompt_node")


class PromptNode:
    """
    Node that builds a prompt from:
      - user_query: str
      - schemas: mapping table_name -> {"columns": [...], "sample_rows": [...]}
      - retrieved_docs: list of {'id','score','text','meta'}
      - few_shot: optional list of examples

    You can inject a custom prompt_builder (callable) for testing.
    """

    def __init__(self, prompt_builder: Optional[Callable[..., str]] = None):
        try:
            # default builder uses utils.build_prompt
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
        """
        try:
            retrieved_docs = retrieved_docs or []
            prompt = self._builder(user_query, schemas, retrieved_docs, few_shot)
            logger.info("Prompt built successfully")
            return prompt
        except CustomException:
            raise
        except Exception as e:
            logger.exception("PromptNode.run failed")
            raise CustomException(e, sys)
