# app/graph/agent.py
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Agent:
    """
    A simple orchestration layer that:
    - Takes an ordered dict of node_name -> node_instance
    - Passes input between them step-by-step
    - Captures final output and intermediate steps for debugging
    """

    def __init__(self, nodes: Dict[str, Any]):
        if not isinstance(nodes, dict):
            raise ValueError("Nodes must be provided as a dictionary of {name: node_instance}")
        self.nodes = nodes

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Agent: Starting execution pipeline.")
        context = {"input": input_data, "steps": []}
        current_data = input_data
        last_output = None

        for name, node in self.nodes.items():
            logger.debug(f"Agent: Executing node '{name}'")
            if not hasattr(node, "run"):
                raise AttributeError(f"Node '{name}' does not implement run()")
            try:
                output = node.run(current_data)
            except Exception as exc:
                logger.exception(f"Error in node '{name}'")
                output = {"error": str(exc)}

            context["steps"].append({
                "node": name,
                "input": current_data,
                "output": output
            })

            last_output = output

            # Normalize what to pass next
            if isinstance(output, dict) and "result" in output:
                current_data = {"value": output.get("result")}
            else:
                current_data = output

        context["final_output"] = last_output
        logger.info("Agent: Pipeline completed.")
        return context


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
        few_shot: Optional[List[Dict[str, str]]] = None,
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
        # Build a simple prompt_text (real builder would call prompt_node)
        prompt_text = f"-- USER QUERY --\n{user_query}\n-- SCHEMAS --\n{self.schema_map}\n-- DOCS --\n{self.retrieved_docs}"
        if self.few_shot:
            prompt_text += f"\n-- FEW SHOT EXAMPLES --\n{self.few_shot}"

        # Call LLM via langsmith_client
        if self.langsmith_client and hasattr(self.langsmith_client, "generate"):
            raw_response = self.langsmith_client.generate(prompt_text, model="gemini-1.5-flash", max_tokens=512)
            sql = raw_response.get("text", "").strip() if isinstance(raw_response, dict) else str(raw_response)
        else:
            raw_response = {"text": "SELECT * FROM dummy;"}
            sql = "SELECT * FROM dummy;"

        return sql, prompt_text, raw_response
