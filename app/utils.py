# D:\text_to_sql_bot\app\utils.py
import re
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


# -------------------------------
# SQL Extraction & Safety Helpers
# -------------------------------

def extract_sql_from_text(text: str) -> str:
    """
    Extract SQL from an LLM response.
    Strategy:
      1. Prefer fenced ```sql ... ``` blocks.
      2. Otherwise return the first SELECT ... (until semicolon or end).
      3. Otherwise return trimmed text.
    """
    if not text:
        return ""

    # 1) fenced code block with 'sql' language
    m = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if m:
        sql = m.group(1).strip()
        logger.debug("extract_sql_from_text: found fenced sql block")
        return sql

    # 2) fenced code block without language
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        cand = m.group(1).strip()
        if re.search(r"\bSELECT\b", cand, re.IGNORECASE):
            logger.debug("extract_sql_from_text: found fenced block with SELECT")
            return cand

    # 3) find first SELECT ... ; or end
    m = re.search(r"(SELECT\b.*?)(;|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        sql = m.group(1).strip()
        logger.debug("extract_sql_from_text: found inline SELECT")
        return sql

    # 4) fallback to full text (trim)
    logger.debug("extract_sql_from_text: falling back to full text")
    return text.strip()


def is_safe_sql(sql: str) -> bool:
    """
    Disallow destructive SQL statements.
    Returns False if unsafe keyword present as a whole word.
    """
    if not sql:
        return True
    unsafe_keywords = [
        "DROP", "DELETE", "ALTER", "TRUNCATE",
        "UPDATE", "INSERT", "CREATE", "ATTACH", "DETACH"
    ]
    for kw in unsafe_keywords:
        if re.search(rf"\b{kw}\b", sql, re.IGNORECASE):
            logger.warning(f"Unsafe SQL detected: {kw}")
            return False
    return True


def limit_sql_rows(sql: str, limit: int = 100) -> str:
    """
    Wrap SQL so that returned rows are limited.
    """
    sql = sql.strip().rstrip(";")
    return f"SELECT * FROM ({sql}) AS _sub LIMIT {int(limit)}"


def is_select_query(sql: str) -> bool:
    """
    Return True if SQL starts with SELECT (ignoring whitespace/comments).
    """
    if not sql:
        return False
    s = sql.strip()
    return bool(re.match(r"^\s*\(?\s*SELECT\b", s, re.IGNORECASE))


def _normalize_identifier(name: str) -> str:
    """
    Normalize table name: strip quotes/backticks and schema prefixes.
    """
    name = name.strip()
    if (name.startswith('"') and name.endswith('"')) or (name.startswith("'") and name.endswith("'")):
        name = name[1:-1]
    if name.startswith("`") and name.endswith("`"):
        name = name[1:-1]
    if "." in name:
        name = name.split(".")[-1]
    return name


def validate_tables_in_sql(sql: str, schemas: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Return list of table names referenced in SQL that are NOT present in schemas.
    """
    if not sql:
        return []

    known = {t.lower() for t in schemas.keys()}
    tokens = re.findall(r"\b(?:FROM|JOIN|INTO|UPDATE)\s+([^\s,;()]+)", sql, re.IGNORECASE)
    missing = []
    for tok in tokens:
        nm = _normalize_identifier(tok).lower().split()[0]
        if nm and nm not in known and nm not in missing:
            missing.append(nm)
    return missing


# -------------------------------
# Prompt / LLM Helpers
# -------------------------------

def format_few_shot_examples(examples: Optional[List[Dict[str, str]]]) -> str:
    """
    Convert few-shot examples into prompt-ready text.
    """
    if not examples:
        return ""
    out = []
    for ex in examples:
        q = ex.get("query") or ex.get("question") or ""
        a = ex.get("sql") or ex.get("answer") or ""
        out.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(out)


def format_retrieved_docs(docs: Optional[List[Dict[str, Any]]]) -> str:
    """
    Format retrieved documents (RAG) for prompt.
    Expected doc shape: {"id":..., "score":..., "text":..., "meta": {...}}
    """
    if not docs:
        return ""
    parts = []
    for d in docs:
        title = d.get("meta", {}).get("table_name") or d.get("meta", {}).get("title") or d.get("id") or "Document"
        score = d.get("score")
        text = d.get("text") or d.get("meta", {}).get("text") or ""
        if score is not None:
            parts.append(f"{title} (score={score:.3f}):\n{text}")
        else:
            parts.append(f"{title}:\n{text}")
    return "\n\n".join(parts)


def flatten_schema(schema: Dict[str, Any]) -> str:
    """
    Convert schema mapping into readable text.
    """
    if not schema:
        return ""
    parts = []
    for table, meta in schema.items():
        if isinstance(meta, dict):
            cols = meta.get("columns") or []
        else:
            cols = meta or []
        col_text = ", ".join(map(str, cols))
        parts.append(f"Table: {table} | Columns: {col_text}")
        if isinstance(meta, dict):
            sample_rows = meta.get("sample_rows") or []
            if sample_rows:
                preview = preview_sample_rows(sample_rows, max_preview=2)
                parts.append(f"  sample rows: {preview}")
    return "\n".join(parts)


def preview_sample_rows(rows: List[Dict[str, Any]], max_preview: int = 3) -> str:
    """
    Render a small human-friendly preview of sample rows.
    """
    if not rows:
        return ""
    preview = rows[:max_preview]
    lines = []
    for r in preview:
        kvs = ", ".join(f"{k}={v}" for k, v in r.items())
        lines.append(kvs)
    return " | ".join(lines)


def build_prompt(
    user_query: str,
    schemas: Dict[str, Any],
    retrieved_docs: Optional[List[Dict[str, Any]]] = None,
    few_shot: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Build prompt with:
      - few-shot examples
      - flattened schema
      - retrieved documents
      - user query
    """
    parts = []

    fs = format_few_shot_examples(few_shot)
    if fs:
        parts.append(fs)

    schema_text = flatten_schema(schemas)
    if schema_text:
        parts.append(schema_text)

    docs_text = format_retrieved_docs(retrieved_docs)
    if docs_text:
        parts.append("Retrieved Documents:\n" + docs_text)

    parts.append(f"User Query: {user_query}\nSQL:")
    prompt = "\n\n".join(parts)
    return prompt
