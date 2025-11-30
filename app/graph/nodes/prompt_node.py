# app/graph/nodes/prompt_node.py
import sys
from typing import Dict, Any, List, Optional, Callable

from app.logger import get_logger
from app.exception import CustomException
from app import utils

logger = get_logger("prompt_node")


def _default_prompt_builder(user_query: str, schemas: Dict[str, Dict[str, Any]], retrieved_docs: List[Dict[str, Any]], few_shot: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Simple deterministic prompt builder used as a fallback.
    It constructs:
      - A compact 'Schema' block listing each canonical table and its columns (and up to 3 sample rows)
      - Optional retrieved docs block
      - The user's question
    """
    parts = []
    parts.append("You are a SQL assistant. Use the exact table and column names below when writing SQL.")
    parts.append("")
    # Schema block
    if schemas:
        parts.append("Schema:")
        for table, info in schemas.items():
            parts.append(f"Table: {table}")
            cols = info.get("columns") or []
            if cols:
                parts.append("Columns: " + ", ".join(cols))
            # sample rows (show as CSV-like)
            sample_rows = info.get("sample_rows") or []
            if sample_rows:
                parts.append("Sample rows:")
                # sample rows might be dicts (if stored that way) or lists; format defensively
                for r in sample_rows[:3]:
                    if isinstance(r, dict):
                        kv = ", ".join(f"{k}={v}" for k, v in r.items())
                        parts.append(f" - {kv}")
                    elif isinstance(r, (list, tuple)):
                        parts.append(" - " + ", ".join(str(x) for x in r))
                    else:
                        parts.append(" - " + str(r))
            parts.append("")  # blank line between tables
    else:
        parts.append("Schema: (no schema information available)")

    # Retrieved docs (RAG)
    if retrieved_docs:
        parts.append("Relevant documents:")
        for doc in retrieved_docs[:5]:
            title = doc.get("meta", {}).get("title") or doc.get("id") or "<doc>"
            snippet = doc.get("text") or doc.get("content") or ""
            parts.append(f"- {title}: {snippet[:200].replace('\\n', ' ')}")
        parts.append("")

    # Few-shot examples if provided (pass through)
    if few_shot:
        parts.append("Examples:")
        for ex in few_shot:
            # expected shape: {"user":"...", "sql":"..."}
            if isinstance(ex, dict):
                u = ex.get("user") or ex.get("query") or ""
                s = ex.get("sql") or ex.get("assistant") or ""
                parts.append(f"User: {u}")
                parts.append(f"SQL Example: {s}")
                parts.append("")

    # User query
    parts.append("User question:")
    parts.append(user_query)
    parts.append("")
    parts.append("Rules: Output only valid SQL wrapped in a single ```sql block when returning SQL. Use only the tables and columns above. If you cannot produce SQL, explain why briefly.")

    return "\n".join(parts)


class PromptNode:
    """
    Node that builds a prompt for the LLM.

    Combines:
      - user_query: str
      - schemas: mapping table_name -> {"columns": [...], "sample_rows": [...]}
      - retrieved_docs: list of {'id','score','text','meta'}
      - few_shot: optional list of examples

    Uses utils.build_prompt if available; otherwise falls back to a deterministic builder.
    """

    def __init__(self, prompt_builder: Optional[Callable[..., str]] = None):
        try:
            # prefer an explicitly provided builder, else try utils.build_prompt, else fallback
            if prompt_builder:
                self._builder = prompt_builder
            else:
                self._builder = getattr(utils, "build_prompt", None) or _default_prompt_builder
        except Exception as e:
            logger.exception("Failed to initialize PromptNode")
            raise CustomException(e, sys)

    def run(
        self,
        user_query: str,
        schemas: Dict[str, Dict[str, Any]],
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        few_shot: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Build and return the prompt string.

        Ensures schema is formatted as canonical table names + columns and
        reduces sample rows to a small number (1-3).
        """
        try:
            retrieved_docs = retrieved_docs or []
            # sanitize schemas: ensure values are simple lists/dicts and cap sample rows
            clean_schemas: Dict[str, Dict[str, Any]] = {}
            for table, info in (schemas or {}).items():
                cols = info.get("columns") or []
                samples = info.get("sample_rows") or []
                # normalize sample rows (if dict rows -> keep, else show as list)
                normalized_samples = []
                for r in samples[:3]:
                    # try to keep mapping form if possible
                    if isinstance(r, dict):
                        normalized_samples.append({k: str(v) for k, v in r.items()})
                    elif isinstance(r, (list, tuple)):
                        # convert to list of strings using column names if possible
                        if cols and len(cols) == len(r):
                            normalized_samples.append({cols[i]: str(r[i]) for i in range(len(r))})
                        else:
                            normalized_samples.append([str(x) for x in r])
                    else:
                        normalized_samples.append(str(r))
                clean_schemas[table] = {"columns": cols, "sample_rows": normalized_samples}

            # Build prompt using the selected builder
            prompt = self._builder(user_query, clean_schemas, retrieved_docs, few_shot)
            logger.info("PromptNode: Prompt built successfully")
            return prompt
        except CustomException:
            raise
        except Exception as e:
            logger.exception("PromptNode.run failed; falling back to default builder")
            # fallback to deterministic builder to at least produce a good prompt
            try:
                return _default_prompt_builder(user_query, schemas or {}, retrieved_docs or [], few_shot)
            except Exception as ex:
                logger.exception("PromptNode fallback builder failed")
                raise CustomException(ex, sys)
