# File: app/graph/agent.py
import logging
import time
from typing import Any, Dict, List, Optional

from app import config

logger = logging.getLogger(__name__)


class Agent:
    """
    Generic orchestrator that runs a sequence of nodes in order.

    - Takes dict node_name -> node_instance
    - Passes input between nodes
    - Captures outputs and logs intermediate steps
    """

    def __init__(self, nodes: Dict[str, Any]):
        if not isinstance(nodes, dict):
            raise ValueError("Nodes must be provided as a dictionary {name: node_instance}")
        self.nodes = nodes

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Agent: Starting execution pipeline.")
        context = {"input": input_data, "steps": []}
        current_data = input_data
        last_output = None

        for name, node in self.nodes.items():
            logger.debug("Agent: Executing node '%s'", name)
            if not hasattr(node, "run"):
                logger.error("Node '%s' missing run() method", name)
                raise AttributeError(f"Node '{name}' does not implement run()")
            try:
                output = node.run(current_data)
            except Exception as exc:
                logger.exception("Error in node '%s'", name)
                output = {"error": str(exc)}

            context["steps"].append({"node": name, "input": current_data, "output": output})
            last_output = output

            # Determine next input
            if isinstance(output, dict) and "result" in output:
                current_data = {"value": output.get("result")}
            else:
                current_data = output

        context["final_output"] = last_output
        logger.info("Agent: Pipeline completed.")
        return context


class LangGraphAgent:
    """
    Specialized agent for LLM-based SQL generation using LangGraph-style flow.

    Parameters
    ----------
    schema_map : Dict[str, Dict[str, Any]]
        Mapping table_name -> {"columns": [...], "sample_rows": [...]}
    retrieved_docs : List[Dict[str, Any]]
        Retrieved documents from vector search.
    few_shot : Optional[List[Dict[str, str]]]
        Few-shot examples for LLM prompt.
    provider_client : Optional[Any]
        LLM provider client with `generate(prompt, model, max_tokens)`.
    langsmith_client : Optional[Any]
        LangSmith client for observability/tracing.
    """

    def __init__(
        self,
        schema_map: Dict[str, Dict[str, Any]],
        retrieved_docs: List[Dict[str, Any]],
        few_shot: Optional[List[Dict[str, str]]] = None,
        provider_client: Optional[Any] = None,
        langsmith_client: Optional[Any] = None,
    ):
        self.schema_map = schema_map or {}
        self.retrieved_docs = retrieved_docs or []
        self.few_shot = few_shot or []
        self.provider_client = provider_client
        self.langsmith_client = langsmith_client

        self.default_model = getattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
        self.allow_langsmith_gen = getattr(config, "USE_LANGSMITH_FOR_GEN", False)

    def _build_prompt(self, user_query: str) -> str:
        """Construct prompt including schemas, docs, and few-shot examples."""
        prompt_parts = [
            "-- USER QUERY --",
            user_query,
            "-- SCHEMAS --",
            str(self.schema_map),
            "-- DOCS --",
            str(self.retrieved_docs),
        ]
        if self.few_shot:
            prompt_parts.extend(["-- FEW SHOT EXAMPLES --", str(self.few_shot)])
        return "\n".join(prompt_parts)

    def _extract_text(self, raw: Any) -> str:
        """Heuristic to extract main text from provider response."""
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, dict):
            for k in ("text", "output", "result", "content"):
                if k in raw:
                    v = raw[k]
                    if isinstance(v, list):
                        return " ".join(map(str, v)).strip()
                    return str(v).strip()
            outputs = raw.get("outputs")
            if isinstance(outputs, list) and outputs:
                first = outputs[0]
                if isinstance(first, dict) and "text" in first:
                    return str(first["text"]).strip()
            try:
                return str(raw)
            except Exception:
                return ""
        return str(raw).strip()

    def run(self, user_query: str):
        """
        Generate SQL for a user query using LLM.

        Returns:
            sql (str), prompt_text (str), raw_response (Any)
        """
        prompt_text = self._build_prompt(user_query)
        trace_metadata_base = {
            "schema_tables": list(self.schema_map.keys()),
            "retrieved_docs": len(self.retrieved_docs),
            "few_shot": len(self.few_shot),
        }

        # Pre-run trace
        if self.langsmith_client and hasattr(self.langsmith_client, "trace_run"):
            try:
                self.langsmith_client.trace_run(
                    name="langgraph.generate.start",
                    prompt=prompt_text,
                    sql=None,
                    metadata={"phase": "start", **trace_metadata_base},
                )
            except Exception:
                logger.exception("LangSmith trace_run (start) failed; continuing")

        raw_response = None
        sql = ""
        provider_used = "none"
        gen_start = time.time()

        try:
            # Primary: provider_client
            if self.provider_client and hasattr(self.provider_client, "generate"):
                provider_used = getattr(self.provider_client, "__class__", type(self.provider_client)).__name__
                logger.debug("Generating via provider_client (%s)", provider_used)
                raw_response = self.provider_client.generate(prompt_text, model=self.default_model, max_tokens=512)
            else:
                # fallback: LangSmith (if allowed)
                if self.langsmith_client and self.allow_langsmith_gen and hasattr(self.langsmith_client, "generate"):
                    provider_used = "LangSmith (opt-in)"
                    logger.warning("Using LangSmith.generate() because provider_client missing")
                    raw_response = self.langsmith_client.generate(prompt_text, model=self.default_model, max_tokens=512)
                else:
                    # final safe fallback
                    provider_used = "none"
                    raw_response = {"text": "SELECT * FROM dummy LIMIT 10;"}
                    logger.warning("No provider available; returning dummy SQL")

        except Exception as gen_exc:
            logger.exception("Generation call failed: %s", gen_exc)
            # Attempt fallback
            try:
                if provider_used != "LangSmith (opt-in)" and self.langsmith_client and self.allow_langsmith_gen:
                    logger.warning("Attempting LangSmith.generate() fallback")
                    raw_response = self.langsmith_client.generate(prompt_text, model=self.default_model, max_tokens=512)
                    provider_used = "LangSmith (fallback-after-error)"
                else:
                    raw_response = {"text": "SELECT * FROM dummy LIMIT 10;"}
                    provider_used = "none-fallback"
            except Exception:
                logger.exception("Fallback also failed; using dummy SQL")
                raw_response = {"text": "SELECT * FROM dummy LIMIT 10;"}
                provider_used = "none-fallback"

        gen_time = time.time() - gen_start
        sql = self._extract_text(raw_response)

        # Post-run trace
        if self.langsmith_client and hasattr(self.langsmith_client, "trace_run"):
            try:
                meta = {
                    **trace_metadata_base,
                    "provider": provider_used,
                    "gen_time_s": gen_time,
                    "success": bool(sql),
                }
                if not sql:
                    meta["error"] = "empty_response"
                self.langsmith_client.trace_run(
                    name="langgraph.generate.complete",
                    prompt=prompt_text,
                    sql=sql or None,
                    metadata=meta,
                )
            except Exception:
                logger.exception("LangSmith trace_run (complete) failed; ignoring")

        return sql, prompt_text, raw_response
