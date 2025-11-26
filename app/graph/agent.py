# app/graph/agent.py
import logging
import time
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
        self.schema_map = schema_map
        self.retrieved_docs = retrieved_docs or []
        self.few_shot = few_shot or []
        self.provider_client = provider_client
        self.langsmith_client = langsmith_client

    def _build_prompt(self, user_query: str) -> str:
        """
        Build a plain prompt used to ask the LLM. In your real code this should
        call the prompt_node/prompt builder to create a nicely formatted prompt.
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
            # fallback: stringify
            try:
                return str(raw)
            except Exception:
                return ""
        # fallback for other types
        return str(raw).strip()

    def run(self, user_query: str):
        """
        Simulates SQL generation via LangGraph flow.
        Prefer provider_client.generate() (Gemini) for generation.
        Falls back to langsmith_client.generate() only if provider missing.
        Uses langsmith_client.trace_run() (if provided) to record observability metadata.

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

        # Try provider_client first (preferred path)
        raw_response = None
        sql = ""

        # If langsmith_client is present, record a pre-run trace (start)
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

        # Attempt generation with provider_client
        provider_used = None
        gen_start = time.time()
        try:
            if self.provider_client and hasattr(self.provider_client, "generate"):
                provider_used = getattr(self.provider_client, "__class__", type(self.provider_client)).__name__
                logger.debug(f"LangGraphAgent: generating via provider_client ({provider_used})")
                raw_response = self.provider_client.generate(prompt_text, model="gemini-1.5-flash", max_tokens=512)
            elif self.langsmith_client and hasattr(self.langsmith_client, "generate"):
                provider_used = "LangSmith (fallback)"
                logger.debug("LangGraphAgent: provider_client missing, using LangSmith.generate() as fallback")
                raw_response = self.langsmith_client.generate(prompt_text, model="gemini-1.5-flash", max_tokens=512)
            else:
                # No provider available: produce a dummy safe SQL response
                provider_used = "none"
                raw_response = {"text": "SELECT * FROM dummy LIMIT 10;"}
                logger.warning("LangGraphAgent: no provider available, returning dummy SQL")
        except Exception as e:
            gen_err = e
            logger.exception("Generation call failed in LangGraphAgent")
            # Try to capture fallback to LangSmith if not already used
            if provider_used != "LangSmith (fallback)" and self.langsmith_client and hasattr(self.langsmith_client, "generate"):
                try:
                    raw_response = self.langsmith_client.generate(prompt_text, model="gemini-1.5-flash", max_tokens=512)
                    provider_used = "LangSmith (fallback-after-error)"
                except Exception:
                    logger.exception("Fallback to LangSmith.generate() also failed")
                    raw_response = {"text": "SELECT * FROM dummy LIMIT 10;"}
                    provider_used = "none-fallback"
            else:
                raw_response = {"text": "SELECT * FROM dummy LIMIT 10;"}
                provider_used = "none-fallback"

        gen_time = time.time() - gen_start

        # Extract SQL text
        sql = self._extract_text(raw_response)

        # Post-run trace with latency and status
        if self.langsmith_client and hasattr(self.langsmith_client, "trace_run"):
            try:
                meta = {
                    **trace_metadata_base,
                    "provider": provider_used,
                    "gen_time_s": gen_time,
                    "success": True if sql else False,
                }
                # include error info when generation failed
                if not sql:
                    meta["error"] = "empty_response"
                self.langsmith_client.trace_run(name="langgraph.generate.complete", prompt=prompt_text, sql=sql or None, metadata=meta)
            except Exception:
                logger.exception("LangSmith trace_run (complete) failed; ignoring")

        return sql, prompt_text, raw_response
