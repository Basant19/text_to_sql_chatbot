import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# -------------------------------
# SQL Helpers
# -------------------------------

def extract_sql_from_text(text: str) -> str:
    """
    Extract SQL statements from LLM output.
    Removes explanation text and keeps only SQL.
    """
    # naive extraction: look for text between ```sql ... ``` or last SELECT
    sql_match = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        sql = sql_match.group(1).strip()
    else:
        # fallback: try to find SELECT statements
        select_match = re.search(r"(SELECT.*?)(;|$)", text, re.IGNORECASE | re.DOTALL)
        sql = select_match.group(1).strip() if select_match else text.strip()
    logger.debug(f"Extracted SQL: {sql}")
    return sql

def is_safe_sql(sql: str) -> bool:
    """
    Basic safety check: no destructive statements allowed.
    """
    unsafe_keywords = ["DROP", "DELETE", "ALTER", "TRUNCATE", "UPDATE", "INSERT"]
    for kw in unsafe_keywords:
        if re.search(rf"\b{kw}\b", sql, re.IGNORECASE):
            logger.warning(f"Unsafe SQL detected: {kw}")
            return False
    return True

def limit_sql_rows(sql: str, limit: int = 100) -> str:
    """
    Wrap SQL to enforce max rows.
    """
    return f"SELECT * FROM ({sql}) AS _sub LIMIT {limit}"


# -------------------------------
# Prompt / LLM Helpers
# -------------------------------

def format_few_shot_examples(examples: List[Dict[str, str]]) -> str:
    """
    Convert few-shot examples into prompt-ready text
    """
    formatted = []
    for ex in examples:
        q = ex.get("query", "")
        a = ex.get("sql", "")
        formatted.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(formatted)

def format_retrieved_docs(docs: List[Dict[str, str]]) -> str:
    """
    Format retrieved docs for prompt inclusion
    """
    formatted = []
    for doc in docs:
        title = doc.get("title", "Document")
        content = doc.get("content", "")
        formatted.append(f"{title}:\n{content}")
    return "\n\n".join(formatted)

def build_prompt(
    user_query: str,
    schemas: Dict[str, List[str]],
    retrieved_docs: Optional[List[Dict[str, str]]] = None,
    few_shot: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Build full prompt for LLM: schema + retrieved docs + user query + few-shot examples
    """
    schema_text = flatten_schema(schemas)
    docs_text = format_retrieved_docs(retrieved_docs) if retrieved_docs else ""
    few_shot_text = format_few_shot_examples(few_shot) if few_shot else ""
    
    prompt_parts = [part for part in [few_shot_text, schema_text, docs_text] if part]
    prompt = "\n\n".join(prompt_parts)
    prompt += f"\n\nUser Query: {user_query}\nSQL:"
    return prompt


# -------------------------------
# CSV / Schema Helpers
# -------------------------------

def flatten_schema(schema: Dict[str, List[str]]) -> str:
    """
    Convert schema dict into human-readable text for prompt
    """
    parts = []
    for table, columns in schema.items():
        cols = ", ".join(columns)
        parts.append(f"Table: {table} | Columns: {cols}")
    return "\n".join(parts)

def preview_sample_rows(rows: List[Dict[str, str]], max_preview: int = 3) -> str:
    """
    Convert a few rows into human-readable text
    """
    preview = rows[:max_preview]
    lines = []
    for row in preview:
        lines.append(", ".join(f"{k}={v}" for k, v in row.items()))
    return "\n".join(lines)
