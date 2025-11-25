# app/llm_flow.py

import sys
import time
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import schema_store, vector_search, sql_executor
from app.langsmith_client import LangSmithClient
from app.utils import build_prompt
from app.graph.agent import LangGraphAgent

logger = get_logger("llm_flow")


def generate_sql_from_query(
    user_query: str,
    csv_names: List[str],
    run_query: bool = False,
    top_k: int = 5,
    client: Optional[LangSmithClient] = None,
    use_langgraph: bool = True,
    few_shot: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Generate SQL for a given user query using RAG + LLM.

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
    client : Optional[LangSmithClient]
        LangSmith client instance; created if None.
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
        client = client or LangSmithClient()

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
        # Retrieve relevant docs
        # ----------------------
        retrieved_docs: List[Dict[str, Any]] = []
        try:
            vs = vector_search.VectorSearch(embedding_fn=None)
            docs = vs.search(user_query, top_k=top_k) or []
            if docs:
                filtered = [
                    d for d in docs
                    if not csv_names or any(
                        cn.lower() in (d.get("meta", {}).get("path", "").lower() or
                                       d.get("meta", {}).get("table_name", "").lower())
                        for cn in csv_names
                    )
                ]
                retrieved_docs = filtered if filtered else docs
        except Exception:
            logger.exception("Vector search retrieval failed; proceeding without retrieved docs")

        # ----------------------
        # Generate SQL
        # ----------------------
        sql: str = ""
        prompt_text: str = ""
        raw_response: Any = None

        if use_langgraph:
            agent = LangGraphAgent(
                schema_map=schemas_map,
                retrieved_docs=retrieved_docs,
                few_shot=few_shot,
                langsmith_client=client
            )
            sql, prompt_text, raw_response = agent.run(user_query)
        else:
            prompt_text = build_prompt(user_query, schemas_map, retrieved_docs, few_shot=few_shot)
            resp = client.generate(prompt_text, model="gemini-1.5-flash", max_tokens=512)
            raw_response = resp
            if isinstance(resp, dict):
                sql = resp.get("text", "").strip()
            else:
                sql = str(resp).strip()

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
