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
    LangSmith is used only for observability/tracing (trace_run only) unless explicitly opted-in.

    Returns a dict:
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
        # If not provided, create a local client (it may be a noop if no key/endpoint configured).
        try:
            langsmith_client = langsmith_client or LangSmithClient()
        except Exception as e:
            logger.warning("Failed to initialize LangSmithClient for tracing: %s", e)
            langsmith_client = None

        # --- Construct provider client (Gemini) lazily ---
        provider_client = None
        try:
            from app.gemini_client import GeminiClient  # type: ignore

            gemini_key = getattr(config, "GEMINI_API_KEY", None) or getattr(config, "GOOGLE_API_KEY", None)
            try:
                provider_client = GeminiClient(api_key=gemini_key) if gemini_key else GeminiClient()
                logger.info("Gemini provider client initialized for generation")
            except Exception as e:
                provider_client = None
                logger.warning("GeminiClient instantiation failed; provider_client set to None: %s", e)
        except Exception:
            provider_client = None
            logger.debug("GeminiClient import not available; provider_client remains None")

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
                logger.warning("Schema not found for CSV/table '%s'", name)

        # ----------------------
        # Retrieve relevant docs (RAG)
        # ----------------------
        retrieved_docs: List[Dict[str, Any]] = []
        try:
            vs = vector_search.VectorSearch(embedding_fn=None)
            docs = vs.search(user_query, top_k=top_k) or []
            if docs:
                # Filter by csv_names if provided (best-effort matching against metadata)
                if csv_names:
                    filtered = [
                        d
                        for d in docs
                        if any(
                            cn.lower() in ((d.get("meta", {}).get("path", "") or "").lower())
                            or cn.lower() in ((d.get("meta", {}).get("table_name", "") or "").lower())
                            for cn in csv_names
                        )
                    ]
                    retrieved_docs = filtered if filtered else docs
                else:
                    retrieved_docs = docs
        except Exception:
            logger.exception("Vector search retrieval failed; proceeding without retrieved docs")

        # ----------------------
        # Build prompt
        # ----------------------
        built_prompt = build_prompt(user_query, schemas_map, retrieved_docs, few_shot=few_shot)

        # Trace start of the flow (observability only)
        try:
            if langsmith_client and hasattr(langsmith_client, "trace_run"):
                langsmith_client.trace_run(
                    name="llm_flow.start",
                    prompt=built_prompt,
                    sql=None,
                    metadata={"top_k": top_k, "use_langgraph": use_langgraph, "retrieved_docs": len(retrieved_docs)},
                )
        except Exception:
            logger.debug("llm_flow: trace_run(start) failed (ignored)")

        # ----------------------
        # Generate SQL (LangGraph preferred)
        # ----------------------
        sql: str = ""
        prompt_text: str = ""
        raw_response: Any = None
        try:
            if use_langgraph:
                # Provide both provider and tracer to LangGraphAgent; agent itself handles preferred provider vs tracer fallback logic.
                agent = LangGraphAgent(
                    schema_map=schemas_map,
                    retrieved_docs=retrieved_docs,
                    few_shot=few_shot,
                    provider_client=provider_client,
                    langsmith_client=langsmith_client,
                )
                gen_out = agent.run(user_query)
                # Agent returns (sql, prompt_text, raw_response) as documented
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
                # Direct provider path (no LangGraph)
                prompt_text = built_prompt
                gen_start = time.time()
                # Primary: Gemini provider
                if provider_client is not None and hasattr(provider_client, "generate"):
                    try:
                        raw_response = provider_client.generate(prompt_text, model=getattr(config, "GEMINI_MODEL", "gemini-2.5-flash"), max_tokens=512)
                    except Exception:
                        logger.exception("Gemini provider.generate failed; attempting LangSmith fallback (if enabled)")
                        # Only call LangSmith.generate if operator explicitly opted-in
                        if langsmith_client and getattr(config, "USE_LANGSMITH_FOR_GEN", False) and hasattr(langsmith_client, "generate"):
                            raw_response = langsmith_client.generate(prompt_text, model=getattr(config, "GEMINI_MODEL", "gemini-2.5-flash"), max_tokens=512)
                        else:
                            raw_response = {"text": ""}
                else:
                    # No provider: only use LangSmith.generate if operator opted-in
                    if langsmith_client and getattr(config, "USE_LANGSMITH_FOR_GEN", False) and hasattr(langsmith_client, "generate"):
                        raw_response = langsmith_client.generate(prompt_text, model=getattr(config, "GEMINI_MODEL", "gemini-2.5-flash"), max_tokens=512)
                    else:
                        raw_response = {"text": ""}

                gen_runtime = time.time() - gen_start
                if isinstance(raw_response, dict):
                    sql = (raw_response.get("text") or raw_response.get("output") or "").strip()
                else:
                    sql = str(raw_response).strip()

                # Trace direct generation result (observability)
                try:
                    if langsmith_client and hasattr(langsmith_client, "trace_run"):
                        langsmith_client.trace_run(
                            name="generate_sql.direct",
                            prompt=prompt_text,
                            sql=sql or None,
                            metadata={"top_k": top_k, "provider": "gemini" if provider_client else "langsmith", "gen_time_s": gen_runtime},
                        )
                except Exception:
                    logger.debug("llm_flow: trace_run(direct) failed (ignored)")

        except Exception as e:
            logger.exception("Generation failed in llm_flow")
            raw_response = {"error": str(e)}
            sql = ""

        # Final trace indicating completion
        try:
            if langsmith_client and hasattr(langsmith_client, "trace_run"):
                langsmith_client.trace_run(
                    name="llm_flow.complete",
                    prompt=built_prompt,
                    sql=sql or None,
                    metadata={"top_k": top_k, "use_langgraph": use_langgraph, "retrieved_docs": len(retrieved_docs), "sql_len": len(sql or "")},
                )
        except Exception:
            logger.debug("llm_flow: trace_run(complete) failed (ignored)")

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
                logger.warning("SQL execution failed: %s", error_msg)

        runtime = time.time() - start_time
        logger.info("llm_flow: SQL generated in %.3fs (valid=%s)", runtime, valid)

        return {
            "prompt": prompt_text or built_prompt,
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
