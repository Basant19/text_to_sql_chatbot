# app/graph/agent.py
import logging
import time
from typing import Any, Dict, List, Optional

from app import config

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
            logger.debug("Agent: Executing node '%s'", name)
            if not hasattr(node, "run"):
                raise AttributeError(f"Node '{name}' does not implement run()")
            try:
                output = node.run(current_data)
            except Exception as exc:
                logger.exception("Error in node '%s'", name)
                output = {"error": str(exc)}

            context["steps"].append({"node": name, "input": current_data, "output": output})
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
    LangGraphAgent orchestrates nodes for SQL generation.

    Parameters
    ----------
    schema_map : Dict[str, Dict[str, Any]]
        Mapping table_name -> {"columns": [...], "sample_rows": [...]}
    retrieved_docs : List[Dict[str, Any]]
        Retrieved documents from vector search.
    few_shot : Optional[List[Dict[str, str]]]
        Optional few-shot examples.
    provider_client : Optional[Any]
        Provider client (e.g., Gemini wrapper) which should expose generate(prompt, model, max_tokens)
    langsmith_client : Optional[Any]
        LangSmithClient used for observability/tracing (trace_run)
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

        # Default model from config
        self.default_model = getattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
        # Safety guard: disallow LangSmith for generation unless explicitly enabled
        self.allow_langsmith_gen = getattr(config, "USE_LANGSMITH_FOR_GEN", False)

    def _build_prompt(self, user_query: str) -> str:
        """
        Build a plain prompt used to ask the LLM. In production code, replace this with a
        proper PromptNode/prompt builder that formats schema, retrieved docs, and few-shot examples.
        """
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
        """
        Heuristic to extract the main text output from various provider response shapes.
        """
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, dict):
            # look for common keys
            for k in ("text", "output", "result", "content"):
                if k in raw:
                    v = raw[k]
                    if isinstance(v, list):
                        return " ".join(map(str, v)).strip()
                    return str(v).strip()
            # nested outputs
            outputs = raw.get("outputs")
            if isinstance(outputs, list) and outputs:
                first = outputs[0]
                if isinstance(first, dict) and "text" in first:
                    return str(first["text"]).strip()
            # fallback: try to stringify but keep it short
            try:
                return str(raw)
            except Exception:
                return ""
        # fallback for other types
        return str(raw).strip()

    def run(self, user_query: str):
        """
        Simulates SQL generation via LangGraph flow.

        Preferred path:
          - provider_client.generate(prompt, model, max_tokens)

        Fallback path:
          - If provider_client missing AND USE_LANGSMITH_FOR_GEN=true, use langsmith_client.generate()
          - Otherwise return a safe dummy SQL

        Observability:
          - Calls langsmith_client.trace_run() at start and completion if available.
          - Does NOT call langsmith_client.generate() unless USE_LANGSMITH_FOR_GEN is true.
        Returns:
            sql (str), prompt_text (str), raw_response (dict or other)
        """
        prompt_text = self._build_prompt(user_query)

        # Prepare metadata for tracing
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
            # Primary: provider_client.generate
            if self.provider_client and hasattr(self.provider_client, "generate"):
                provider_used = getattr(self.provider_client, "__class__", type(self.provider_client)).__name__
                logger.debug("LangGraphAgent: generating via provider_client (%s)", provider_used)
                raw_response = self.provider_client.generate(prompt_text, model=self.default_model, max_tokens=512)
            else:
                # provider_client missing: consider LangSmith only if operator explicitly allowed it
                if self.langsmith_client and self.allow_langsmith_gen and hasattr(self.langsmith_client, "generate"):
                    provider_used = "LangSmith (opt-in)"
                    logger.warning("LangGraphAgent: provider_client missing, using LangSmith.generate() because USE_LANGSMITH_FOR_GEN=true")
                    raw_response = self.langsmith_client.generate(prompt_text, model=self.default_model, max_tokens=512)
                else:
                    # No provider available: safe fallback
                    provider_used = "none"
                    raw_response = {"text": "SELECT * FROM dummy LIMIT 10;"}
                    logger.warning("LangGraphAgent: no provider available, returning dummy SQL")

        except Exception as gen_exc:
            logger.exception("Generation call failed in LangGraphAgent: %s", gen_exc)

            # Attempt fallback to LangSmith only if not already used and operator allowed it
            try:
                if provider_used != "LangSmith (opt-in)" and self.langsmith_client and self.allow_langsmith_gen and hasattr(self.langsmith_client, "generate"):
                    logger.warning("LangGraphAgent: attempting LangSmith.generate() fallback due to error")
                    raw_response = self.langsmith_client.generate(prompt_text, model=self.default_model, max_tokens=512)
                    provider_used = "LangSmith (fallback-after-error)"
                else:
                    # final safe fallback
                    raw_response = {"text": "SELECT * FROM dummy LIMIT 10;"}
                    provider_used = "none-fallback"
            except Exception:
                logger.exception("Fallback to LangSmith.generate() also failed; using dummy SQL")
                raw_response = {"text": "SELECT * FROM dummy LIMIT 10;"}
                provider_used = "none-fallback"

        gen_time = time.time() - gen_start

        # Extract SQL text
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
                    name="langgraph.generate.complete", prompt=prompt_text, sql=sql or None, metadata=meta
                )
            except Exception:
                logger.exception("LangSmith trace_run (complete) failed; ignoring")

        return sql, prompt_text, raw_response
