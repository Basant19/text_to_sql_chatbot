# app/llm_flow.py
from __future__ import annotations

import sys
import time
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.schema_store import SchemaStore
from app import sql_executor, config
from app.langsmith_client import LangSmithClient
from app.utils import build_prompt

logger = get_logger("llm_flow")

# Optional VectorSearch
try:
    from app.vector_search import get_instance as get_vector_search_instance
except Exception:
    get_vector_search_instance = None  # type: ignore

# Optional LangGraph agent
try:
    from app.graph.agent import LangGraphAgent
except Exception:
    LangGraphAgent = None  # type: ignore


# ============================================================
# Intent Routing
# ============================================================
def route_intent(user_input: str) -> str:
    """
    Decide how to route user input.

    Returns
    -------
    str
        One of: "sql", "chat"

    Heuristics
    ----------
    - Mentions of tables, columns, data → SQL
    - "summarize", "explain", "describe" → RAG / chat
    - Default → chat
    """
    if not user_input:
        return "chat"

    text = user_input.lower()

    sql_signals = [
        "select",
        "count",
        "average",
        "sum",
        "group by",
        "order by",
        "from ",
        "join ",
        "table",
        "column",
        "rows",
    ]

    if any(sig in text for sig in sql_signals):
        return "sql"

    return "chat"


# ============================================================
# VectorSearch bootstrap (lazy & safe)
# ============================================================
def _init_vector_search():
    """
    Lazily initialize VectorSearch.

    This function:
    - MUST NOT raise
    - MUST NOT mutate global state
    """
    if get_vector_search_instance is None:
        logger.debug("VectorSearch not available")
        return None

    try:
        return get_vector_search_instance(
            index_path=getattr(
                config,
                "FAISS_INDEX_PATH",
                "./faiss/index.faiss",
            )
        )
    except Exception:
        logger.exception("VectorSearch initialization failed")
        return None


# ============================================================
# MAIN ENTRYPOINT
# ============================================================
def handle_user_query(
    *,
    user_input: str,
    csv_names: Optional[List[str]] = None,
    run_query: bool = False,
    top_k: int = 5,
    langsmith_client: Optional[LangSmithClient] = None,
    use_langgraph: bool = True,
) -> Dict[str, Any]:
    """
    Unified LLM entrypoint for UI and API.

    This function handles:
    - SQL generation & execution
    - Document summarization (RAG)
    - General chat

    Returns (normalized)
    --------------------
    {
        "route": "sql" | "chat",
        "sql": Optional[str],
        "execution": Optional[dict],
        "text": Optional[str],
        "error": Optional[str],
        "meta": dict
    }
    """

    start_time = time.time()
    route = route_intent(user_input)

    logger.info("Routing user query | route=%s", route)

    try:
        # --------------------------------------------------
        # Provider client (Gemini)
        # --------------------------------------------------
        from app.gemini_client import GeminiClient

        gemini = GeminiClient()

        # --------------------------------------------------
        # LangSmith (optional)
        # --------------------------------------------------
        try:
            langsmith_client = langsmith_client or LangSmithClient()
        except Exception:
            langsmith_client = None

        # ==================================================
        # CHAT / SUMMARIZATION PATH
        # ==================================================
        if route == "chat":
            logger.debug("Handling general chat request")

            response = gemini.chat(user_input)

            return {
                "route": "chat",
                "sql": None,
                "execution": None,
                "text": response,
                "error": None,
                "meta": {
                    "runtime_sec": round(time.time() - start_time, 4),
                },
            }

        # ==================================================
        # SQL / RAG PATH
        # ==================================================
        logger.debug("Handling SQL / RAG request")

        csv_names = csv_names or []

        # Load schemas
        ss = SchemaStore.get_instance()
        schemas: Dict[str, Dict[str, Any]] = {}

        for name in csv_names:
            schemas[name] = {
                "columns": ss.get_schema(name) or [],
                "sample_rows": ss.get_sample_rows(name) or [],
            }

        if not schemas:
            return {
                "route": "sql",
                "sql": None,
                "execution": None,
                "text": None,
                "error": "No schemas available",
                "meta": {},
            }

        # Vector retrieval (optional)
        retrieved_docs: List[Dict[str, Any]] = []
        vs = _init_vector_search()
        if vs:
            try:
                retrieved_docs = vs.search(user_input, top_k=top_k) or []
            except Exception:
                logger.exception("Vector search failed")

        # Build prompt
        prompt = build_prompt(
            user_query=user_input,
            schemas=schemas,
            retrieved=retrieved_docs,
        )

        # LangGraph preferred
        sql = ""
        raw = None

        if use_langgraph and LangGraphAgent:
            agent = LangGraphAgent(
                schema_map=schemas,
                retrieved_docs=retrieved_docs,
                provider_client=gemini,
                langsmith_client=langsmith_client,
            )
            out = agent.run(user_input)
            if isinstance(out, dict):
                sql = (out.get("sql") or "").strip()
                raw = out
            else:
                sql = str(out).strip()
                raw = out
        else:
            raw = gemini.generate(prompt)
            sql = str(raw).strip()

        execution = None
        error = None

        if run_query and sql:
            try:
                execution = sql_executor.execute_sql(
                    sql,
                    read_only=True,
                    limit=100,
                )
            except Exception as e:
                error = str(e)

        logger.info(
            "llm_flow completed | route=sql | valid=%s | time=%.3fs",
            bool(sql),
            time.time() - start_time,
        )

        return {
            "route": "sql",
            "sql": sql,
            "execution": execution,
            "text": None,
            "error": error,
            "meta": {
                "retrieved_docs": len(retrieved_docs),
                "runtime_sec": round(time.time() - start_time, 4),
            },
        }

    except Exception as e:
        logger.exception("llm_flow fatal error")
        raise CustomException(e, sys)