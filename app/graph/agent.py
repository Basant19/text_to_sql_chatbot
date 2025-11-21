# app/graph/agent.py

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class LangGraphAgent:
    """
    LangGraphAgent orchestrates nodes for SQL generation:
    - schema_map: dict of table -> columns & sample_rows
    - retrieved_docs: relevant docs from vector search
    - few_shot: optional examples for few-shot prompting
    - langsmith_client: LangSmithClient instance for LLM calls
    """

    def __init__(
        self,
        schema_map: Dict[str, Dict[str, Any]],
        retrieved_docs: List[Dict[str, Any]],
        few_shot: List[Dict[str, str]] = None,
        langsmith_client: Any = None,
    ):
        self.schema_map = schema_map
        self.retrieved_docs = retrieved_docs
        self.few_shot = few_shot or []
        self.langsmith_client = langsmith_client

    def run(self, user_query: str):
        """
        Simulates SQL generation via LangGraph flow.
        Returns:
            sql (str), prompt_text (str), raw_response (dict)
        """
        # Build prompt for LLM
        prompt_text = f"-- USER QUERY --\n{user_query}\n-- SCHEMAS --\n{self.schema_map}\n-- DOCS --\n{self.retrieved_docs}"
        if self.few_shot:
            prompt_text += f"\n-- FEW SHOT EXAMPLES --\n{self.few_shot}"

        # Call LLM via langsmith_client
        if self.langsmith_client:
            raw_response = self.langsmith_client.generate(prompt_text, model="gemini-1.5-flash", max_tokens=512)
            sql = raw_response.get("text", "").strip() if isinstance(raw_response, dict) else str(raw_response)
        else:
            raw_response = {"text": "SELECT * FROM dummy;"}
            sql = "SELECT * FROM dummy;"

        return sql, prompt_text, raw_response
