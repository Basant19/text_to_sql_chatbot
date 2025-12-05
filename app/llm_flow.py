# app/llm_flow.py
import sys
import time
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import schema_store, sql_executor, config
from app.langsmith_client import LangSmithClient
from app.utils import build_prompt

# Prefer the singleton accessor for VectorSearch
try:
    from app.vector_search import get_instance as get_vector_search_instance
except Exception:
    get_vector_search_instance = None  # type: ignore

# Try to get a packaged helper that returns a usable embedding function (tools.py may expose it)
_EMBEDDING_FN_PROVIDER = None
try:
    from app import tools  # type: ignore

    if hasattr(tools, "get_embedding_fn"):
        _EMBEDDING_FN_PROVIDER = getattr(tools, "get_embedding_fn")
    elif hasattr(tools, "embedding_fn"):
        _EMBEDDING_FN_PROVIDER = lambda: getattr(tools, "embedding_fn")
except Exception:
    _EMBEDDING_FN_PROVIDER = None

# LangGraphAgent import (optional)
try:
    from app.graph.agent import LangGraphAgent  # type: ignore
except Exception:
    LangGraphAgent = None  # type: ignore

logger = get_logger("llm_flow")


def _init_vector_search():
    """
    Initialize / return a VectorSearch singleton instance.
    Tries to obtain an embedding_fn from tools if available and sets dim if tools publish it.
    """
    if get_vector_search_instance is None:
        return None
    embedding_fn = None
    dim = None
    try:
        if _EMBEDDING_FN_PROVIDER:
            # provider may be callable that returns a function or the function itself
            ef = _EMBEDDING_FN_PROVIDER()
            if callable(ef):
                embedding_fn = ef
            elif callable(_EMBEDDING_FN_PROVIDER):
                embedding_fn = _EMBEDDING_FN_PROVIDER
        # If tools exposes a DIM constant, pick it up
        try:
            dim = getattr(__import__("app.tools", fromlist=[""]), "EMBEDDING_DIM", None)
        except Exception:
            dim = None
    except Exception:
        embedding_fn = None
        dim = None
    # instantiate singleton (get_instance handles None embedding_fn)
    try:
        if dim:
            vs = get_vector_search_instance(embedding_fn=embedding_fn, index_path=getattr(config, "FAISS_INDEX_PATH", "./faiss/index.faiss"), dim=dim)
        else:
            vs = get_vector_search_instance(embedding_fn=embedding_fn, index_path=getattr(config, "FAISS_INDEX_PATH", "./faiss/index.faiss"))
        return vs
    except Exception:
        logger.exception("Failed to initialize VectorSearch singleton")
        return None


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
        # --- LangSmith tracing client (observability) ---
        try:
            langsmith_client = langsmith_client or LangSmithClient()
        except Exception as e:
            logger.warning("Failed to initialize LangSmithClient for tracing: %s", e)
            langsmith_client = None

        # --- Provider client (Gemini) lazily imported in LangGraphAgent or later when needed ---
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
        # Load Schemas from SchemaStore
        # ----------------------
        ss = schema_store.SchemaStore()
        schemas_map: Dict[str, Dict[str, Any]] = {}
        for name in csv_names or []:
            cols = ss.get_schema(name) or []
            samples = ss.get_sample_rows(name) or []
            schemas_map[name] = {"columns": cols, "sample_rows": samples}
            if not cols:
                logger.warning("Schema not found for CSV/table '%s'", name)

        # ----------------------
        # Retrieve relevant docs (RAG) using VectorSearch singleton
        # ----------------------
        retrieved_docs: List[Dict[str, Any]] = []
        try:
            vs = _init_vector_search()
            if vs is not None:
                docs = vs.search(user_query, top_k=top_k) or []
            else:
                docs = []
            # docs are expected to be list of {"id":..., "score":..., "meta": {...}}
            if docs:
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
        prompt_text: str = ""
        sql: str = ""
        raw_response: Any = None

        # Trace start
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
        # Generate SQL (prefer LangGraphAgent)
        # ----------------------
        try:
            if use_langgraph and LangGraphAgent is not None:
                # agent handles provider/tracer internally
                agent = LangGraphAgent(
                    schema_map=schemas_map,
                    retrieved_docs=retrieved_docs,
                    few_shot=few_shot,
                    provider_client=provider_client,
                    langsmith_client=langsmith_client,
                )
                gen_out = agent.run(user_query)
                if isinstance(gen_out, tuple):
                    sql, prompt_text, raw_response = gen_out
                elif isinstance(gen_out, dict):
                    sql = gen_out.get("sql") or gen_out.get("text") or ""
                    prompt_text = gen_out.get("prompt", built_prompt)
                    raw_response = gen_out
                else:
                    raw_response = gen_out
                    sql = str(gen_out or "").strip()
            else:
                # Direct provider path
                prompt_text = built_prompt
                gen_start = time.time()
                # Primary: Gemini provider
                if provider_client is not None and hasattr(provider_client, "generate"):
                    try:
                        raw_response = provider_client.generate(prompt_text, model=getattr(config, "GEMINI_MODEL", "gemini-2.5-flash"), max_tokens=512)
                    except Exception:
                        logger.exception("Gemini provider.generate failed; attempting LangSmith fallback (if enabled)")
                        if langsmith_client and getattr(config, "USE_LANGSMITH_FOR_GEN", False) and hasattr(langsmith_client, "generate"):
                            raw_response = langsmith_client.generate(prompt_text, model=getattr(config, "GEMINI_MODEL", "gemini-2.5-flash"), max_tokens=512)
                        else:
                            raw_response = {"text": ""}
                else:
                    if langsmith_client and getattr(config, "USE_LANGSMITH_FOR_GEN", False) and hasattr(langsmith_client, "generate"):
                        raw_response = langsmith_client.generate(prompt_text, model=getattr(config, "GEMINI_MODEL", "gemini-2.5-flash"), max_tokens=512)
                    else:
                        raw_response = {"text": ""}

                gen_runtime = time.time() - gen_start
                if isinstance(raw_response, dict):
                    sql = (raw_response.get("text") or raw_response.get("output") or "").strip()
                else:
                    sql = str(raw_response).strip()

                # Trace direct generation
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

        # Final trace complete
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
