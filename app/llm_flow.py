# app/llm_flow.py

import sys
import time
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import schema_store, vector_search, sql_executor, config
from app.langsmith_client import LangSmithClient
from app.utils import build_prompt
from app.graph.agent import LangGraphAgent

logger = get_logger("llm_flow")


def generate_sql_from_query(
    user_query: str,
    csv_names: List[str],
    run_query: bool = False,
    top_k: int = 5,
    langsmith_client: Optional[LangSmithClient] = None,
    use_langgraph: bool = True,
    few_shot: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Generate SQL for a given user query using RAG + an LLM provider (Gemini).
    LangSmith is used only for observability/tracing.

    Parameters
    ----------
    user_query : str
        Natural language query from the user.
    csv_names : List[str]
        List of CSV/table names to consider for retrieval.
    run_query : bool
        If True, execute the generated SQL against DuckDB.
    top_k : int
        Number of top retrieved documents to consider for prompt building.
    langsmith_client : Optional[LangSmithClient]
        LangSmith client instance used solely for tracing. Created if None.
    use_langgraph : bool
        Whether to prefer LangGraph agent over direct LLM call.
    few_shot : Optional[List[Dict[str, str]]]
        Optional few-shot examples to include in prompt.

    Returns
    -------
    Dict[str, Any]
        {
            "prompt": str,
            "sql": str,
            "valid": bool,
            "execution": Optional[Dict[str, Any]],
            "error": Optional[str],
            "raw": Any
        }
    """
    start_time = time.time()
    try:
        # --- Construct LangSmith tracing client (used only for observability) ---
        langsmith_client = langsmith_client or LangSmithClient()

        # --- Construct provider client (Gemini) ---
        # Import lazily so tests can monkeypatch/replace gemini client as needed.
        try:
            from app.gemini_client import GeminiClient  # type: ignore

            # Try to get GEMINI_API_KEY from config (fall back to GOOGLE_API_KEY)
            gemini_key = getattr(config, "GEMINI_API_KEY", None) or getattr(config, "GOOGLE_API_KEY", None)
            provider_client = GeminiClient(api_key=gemini_key)
            logger.info("Gemini provider client initialized for generation")
        except Exception as e:
            # If Gemini client isn't available, keep provider_client = None and fallback to LangSmith.generate
            provider_client = None
            logger.warning(f"Could not initialize GeminiClient; falling back to LangSmith.generate(): {e}")

        # ----------------------
        # Load Schemas
        # ----------------------
        ss = schema_store.SchemaStore()
        schemas_map: Dict[str, Dict[str, Any]] = {}
        for name in csv_names:
            cols = ss.get_schema(name) or []
            samples = ss.get_sample_rows(name) or []
            schemas_map[name] = {"columns": cols, "sample_rows": samples}
            if not cols:
                logger.warning(f"Schema not found for CSV/table '{name}'")

        # ----------------------
        # Retrieve relevant docs (RAG)
        # ----------------------
        retrieved_docs: List[Dict[str, Any]] = []
        try:
            vs = vector_search.VectorSearch(embedding_fn=None)
            docs = vs.search(user_query, top_k=top_k) or []
            if docs:
                filtered = [
                    d for d in docs
                    if not csv_names or any(
                        cn.lower() in ((d.get("meta", {}).get("path", "") or "").lower()) or
                        cn.lower() in ((d.get("meta", {}).get("table_name", "") or "").lower())
                        for cn in csv_names
                    )
                ]
                retrieved_docs = filtered if filtered else docs
        except Exception:
            logger.exception("Vector search retrieval failed; proceeding without retrieved docs")

        # ----------------------
        # Generate SQL (via LangGraph or direct provider)
        # ----------------------
        sql: str = ""
        prompt_text: str = ""
        raw_response: Any = None

        # Build prompt once (LangGraph agent may re-use it or rebuild)
        built_prompt = build_prompt(user_query, schemas_map, retrieved_docs, few_shot=few_shot)

        if use_langgraph:
            # Pass both provider (Gemini) and langsmith (tracing) into the agent.
            # Agent is expected to prefer provider_client for generation and call langsmith_client.trace_run for observability.
            agent = LangGraphAgent(
                schema_map=schemas_map,
                retrieved_docs=retrieved_docs,
                few_shot=few_shot,
                provider_client=provider_client,
                langsmith_client=langsmith_client,
            )
            try:
                gen_out = agent.run(user_query)
            except TypeError:
                # Backwards compatibility if agent signature is older: try old kwargs
                agent = LangGraphAgent(
                    schema_map=schemas_map,
                    retrieved_docs=retrieved_docs,
                    few_shot=few_shot,
                    langsmith_client=langsmith_client,
                )
                gen_out = agent.run(user_query)

            # Agent may return tuple (sql, prompt_text, raw) or dict
            if isinstance(gen_out, tuple):
                sql, prompt_text, raw_response = gen_out
            elif isinstance(gen_out, dict):
                sql = gen_out.get("sql") or gen_out.get("text") or ""
                prompt_text = gen_out.get("prompt", built_prompt)
                raw_response = gen_out
            else:
                raw_response = gen_out
                sql = str(gen_out).strip()

        else:
            # Direct provider path: call Gemini provider if available, otherwise LangSmith.generate as fallback.
            prompt_text = built_prompt
            gen_start = time.time()
            if provider_client is not None and hasattr(provider_client, "generate") and callable(provider_client.generate):
                try:
                    resp = provider_client.generate(prompt_text, model="gemini-1.5-flash", max_tokens=512)
                except Exception as e:
                    logger.exception("Gemini provider.generate failed; falling back to LangSmith.generate()")
                    resp = langsmith_client.generate(prompt_text, model="gemini-1.5-flash", max_tokens=512)
            else:
                # fallback to LangSmith.generate if provider not available
                resp = langsmith_client.generate(prompt_text, model="gemini-1.5-flash", max_tokens=512)

            gen_runtime = time.time() - gen_start
            raw_response = resp
            if isinstance(resp, dict):
                sql = resp.get("text", "").strip()
            else:
                sql = str(resp).strip()

            # Trace the generation run to LangSmith (observability)
            try:
                # Include generation latency and method used in metadata
                meta = {"top_k": top_k, "provider": "gemini" if provider_client is not None else "langsmith", "gen_time_s": gen_runtime}
                langsmith_client.trace_run(name="generate_sql", prompt=prompt_text, sql=sql, metadata=meta)
            except Exception:
                logger.exception("Failed to send trace_run to LangSmith after direct generation")

        # If LangGraph path finished, still send a trace (agent likely already traced, but safe to ensure)
        if use_langgraph:
            try:
                meta = {"top_k": top_k, "method": "langgraph"}
                langsmith_client.trace_run(name="generate_sql", prompt=built_prompt, sql=sql, metadata=meta)
            except Exception:
                logger.exception("Failed to send trace_run to LangSmith after LangGraph generation")

        # ----------------------
        # Validate & optionally run
        # ----------------------
        valid = bool(sql)
        execution_result: Optional[Any] = None
        error_msg: Optional[str] = None

        if run_query and valid:
            try:
                execution_result = sql_executor.execute_sql(sql, read_only=True, limit=100)
            except CustomException as ce:
                valid = False
                error_msg = str(ce)
                logger.warning(f"SQL execution failed: {error_msg}")

        runtime = time.time() - start_time
        logger.info(f"llm_flow: SQL generated in {runtime:.3f}s (valid={valid})")

        return {
            "prompt": prompt_text,
            "sql": sql,
            "valid": valid,
            "execution": execution_result,
            "error": error_msg,
            "raw": raw_response,
        }

    except CustomException:
        raise
    except Exception as e:
        logger.exception("LLM flow failed unexpectedly")
        raise CustomException(e, sys)
