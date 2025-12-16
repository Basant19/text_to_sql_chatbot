# app/llm_flow.py
from __future__ import annotations

import sys
import time
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import schema_store, sql_executor, config
from app.langsmith_client import LangSmithClient
from app.utils import build_prompt

# Optional VectorSearch singleton
try:
    from app.vector_search import get_instance as get_vector_search_instance
except Exception:
    get_vector_search_instance = None  # type: ignore

# Optional embedding provider
_EMBEDDING_FN_PROVIDER = None
try:
    from app import tools

    if hasattr(tools, "get_embedding_fn"):
        _EMBEDDING_FN_PROVIDER = tools.get_embedding_fn
    elif hasattr(tools, "embedding_fn"):
        _EMBEDDING_FN_PROVIDER = lambda: tools.embedding_fn
except Exception:
    _EMBEDDING_FN_PROVIDER = None

# Optional LangGraph agent
try:
    from app.graph.agent import LangGraphAgent
except Exception:
    LangGraphAgent = None  # type: ignore

logger = get_logger("llm_flow")


# ------------------------------------------------------------------
# VectorSearch bootstrap (safe)
# ------------------------------------------------------------------
def _init_vector_search():
    if get_vector_search_instance is None:
        logger.info("VectorSearch not available")
        return None

    try:
        embedding_fn = None
        if _EMBEDDING_FN_PROVIDER:
            ef = _EMBEDDING_FN_PROVIDER()
            if callable(ef):
                embedding_fn = ef

        dim = getattr(
            __import__("app.tools", fromlist=[""]), "EMBEDDING_DIM", None
        )

        kwargs = {
            "embedding_fn": embedding_fn,
            "index_path": getattr(config, "FAISS_INDEX_PATH", "./faiss/index.faiss"),
        }
        if dim:
            kwargs["dim"] = dim

        return get_vector_search_instance(**kwargs)

    except Exception:
        logger.exception("VectorSearch initialization failed")
        return None


# ------------------------------------------------------------------
# Main entrypoint
# ------------------------------------------------------------------
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
    Orchestrates:
      - Schema loading
      - Vector retrieval (RAG)
      - SQL generation (LangGraph or direct)
      - Optional execution
    """
    start_time = time.time()

    try:
        # --------------------------------------------------------------
        # LangSmith (observability only, never required)
        # --------------------------------------------------------------
        try:
            langsmith_client = langsmith_client or LangSmithClient()
        except Exception:
            langsmith_client = None
            logger.debug("LangSmith disabled")

        # --------------------------------------------------------------
        # Provider client (Gemini)
        # --------------------------------------------------------------
        provider_client = None
        try:
            from app.gemini_client import GeminiClient

            api_key = getattr(config, "GEMINI_API_KEY", None) or getattr(
                config, "GOOGLE_API_KEY", None
            )
            provider_client = GeminiClient(api_key=api_key) if api_key else GeminiClient()
        except Exception:
            logger.warning("GeminiClient unavailable")
            provider_client = None

        # --------------------------------------------------------------
        # Load schemas
        # --------------------------------------------------------------
        ss = schema_store.SchemaStore()
        schemas_map: Dict[str, Dict[str, Any]] = {}

        for name in csv_names or []:
            schemas_map[name] = {
                "columns": ss.get_schema(name) or [],
                "sample_rows": ss.get_sample_rows(name) or [],
            }

        if not schemas_map:
            return {
                "prompt": "",
                "sql": "",
                "valid": False,
                "execution": None,
                "error": "No schemas available",
                "raw": None,
            }

        # --------------------------------------------------------------
        # Vector retrieval (RAG)
        # --------------------------------------------------------------
        retrieved_docs: List[Dict[str, Any]] = []
        try:
            vs = _init_vector_search()
            if vs:
                retrieved_docs = vs.search(user_query, top_k=top_k) or []
        except Exception:
            logger.exception("Vector search failed")

        # --------------------------------------------------------------
        # Prompt construction
        # --------------------------------------------------------------
        prompt_text = build_prompt(
            user_query=user_query,
            schemas=schemas_map,
            retrieved=retrieved_docs,
            few_shot=few_shot,
        )

        # --------------------------------------------------------------
        # Trace start
        # --------------------------------------------------------------
        if langsmith_client:
            try:
                langsmith_client.trace_run(
                    name="llm_flow.start",
                    prompt=prompt_text,
                    sql=None,
                    metadata={"use_langgraph": use_langgraph},
                )
            except Exception:
                pass

        # --------------------------------------------------------------
        # SQL generation
        # --------------------------------------------------------------
        sql = ""
        raw_response: Any = None

        try:
            if use_langgraph and LangGraphAgent:
                agent = LangGraphAgent(
                    schema_map=schemas_map,
                    retrieved_docs=retrieved_docs,
                    few_shot=few_shot,
                    provider_client=provider_client,
                    langsmith_client=langsmith_client,
                )
                out = agent.run(user_query)

                if isinstance(out, dict):
                    sql = out.get("sql", "")
                    raw_response = out
                else:
                    sql = str(out).strip()
                    raw_response = out

            else:
                if provider_client and hasattr(provider_client, "generate"):
                    raw_response = provider_client.generate(
                        prompt_text,
                        model=getattr(config, "GEMINI_MODEL", "gemini-2.5-flash"),
                        max_tokens=512,
                    )
                else:
                    raw_response = {"text": ""}

                if isinstance(raw_response, dict):
                    sql = (raw_response.get("text") or "").strip()
                else:
                    sql = str(raw_response).strip()

        except Exception as e:
            logger.exception("SQL generation failed")
            raw_response = {"error": str(e)}
            sql = ""

        # --------------------------------------------------------------
        # Final trace
        # --------------------------------------------------------------
        if langsmith_client:
            try:
                langsmith_client.trace_run(
                    name="llm_flow.complete",
                    prompt=prompt_text,
                    sql=sql or None,
                )
            except Exception:
                pass

        # --------------------------------------------------------------
        # Validate & execute
        # --------------------------------------------------------------
        valid = bool(sql)
        execution = None
        error = None

        if run_query and valid:
            try:
                execution = sql_executor.execute_sql(
                    sql, read_only=True, limit=100
                )
            except CustomException as ce:
                valid = False
                error = str(ce)

        logger.info(
            "llm_flow finished in %.3fs (valid=%s)",
            time.time() - start_time,
            valid,
        )

        return {
            "prompt": prompt_text,
            "sql": sql,
            "valid": valid,
            "execution": execution,
            "error": error,
            "raw": raw_response,
        }

    except Exception as e:
        logger.exception("llm_flow fatal error")
        raise CustomException(e, sys)
