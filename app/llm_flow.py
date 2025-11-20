import sys
import time
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import schema_store, vector_search, sql_executor
from app.langsmith_client import LangSmithClient

# LangGraph imports
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
    Main SQL generation flow.

    Steps:
      1. Gather schemas from SchemaStore
      2. Retrieve relevant docs via VectorSearch
      3. Generate SQL via LangGraph agent (if enabled) or direct LLM call
      4. Validate SQL and optionally execute
    Returns dict with:
      prompt, sql, valid, execution result, error, raw LLM response
    """
    start = time.time()
    try:
        client = client or LangSmithClient()

        # --- Load schemas ---
        ss = schema_store.SchemaStore()
        schemas_map: Dict[str, Dict[str, Any]] = {}
        for name in csv_names:
            cols = ss.get_schema(name)
            samples = ss.get_sample_rows(name)
            schemas_map[name] = {"columns": cols or [], "sample_rows": samples or []}
            if not cols:
                logger.warning(f"Schema not found for {name}")

        # --- Retrieve relevant documents ---
        retrieved_docs: List[Dict[str, Any]] = []
        try:
            vs = vector_search.VectorSearch(embedding_fn=None)
            docs = vs.search(user_query, top_k=top_k)
            if docs:
                filtered = [
                    d for d in docs
                    if not csv_names or any(
                        cn in (d.get("meta", {}).get("path", "") or d.get("meta", {}).get("table_name", ""))
                        for cn in csv_names
                    )
                ]
                retrieved_docs = filtered if filtered else docs
        except Exception:
            logger.exception("Vector search retrieval failed; proceeding without docs")

        # --- Generate SQL ---
        raw_response: Any
        sql: str
        prompt_text: str

        if use_langgraph:
            # Use LangGraph agent
            agent = LangGraphAgent(
                schema_map=schemas_map,
                retrieved_docs=retrieved_docs,
                few_shot=few_shot,
                langsmith_client=client
            )
            sql, prompt_text, raw_response = agent.run(user_query)
        else:
            # Direct LLM call
            from app.utils import build_prompt
            prompt_text = build_prompt(user_query, schemas_map, retrieved_docs, few_shot=few_shot)
            resp = client.generate(prompt_text, model="gemini-1.5-flash", max_tokens=512)
            raw_response = resp
            sql = resp.get("text", "").strip() if isinstance(resp, dict) else str(resp)

        # --- Validate SQL ---
        valid = True
        if not sql:
            valid = False

        execution_result = None
        error_msg = None

        if run_query and valid:
            try:
                execution_result = sql_executor.execute_sql(sql, read_only=True, limit=100)
            except CustomException as ce:
                valid = False
                error_msg = str(ce)

        runtime = time.time() - start
        logger.info(f"llm_flow: generated SQL in {runtime:.3f}s (valid={valid})")

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
        logger.exception("LLM flow failed")
        raise CustomException(e, sys)
