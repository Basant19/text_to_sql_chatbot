import re
import logging
from typing import List, Dict, Optional, Any, Set, Tuple

logger = logging.getLogger(__name__)

# try optional dependency
def _have_sqlglot() -> bool:
    try:
        import sqlglot  # type: ignore
        return True
    except Exception:
        return False


_SQLGLOT_AVAILABLE = _have_sqlglot()


# -------------------------------
# Normalization helpers
# -------------------------------
def _strip_quotes(name: str) -> str:
    if not isinstance(name, str):
        name = str(name or "")
    name = name.strip()
    if (name.startswith('"') and name.endswith('"')) or (name.startswith("'") and name.endswith("'")):
        name = name[1:-1]
    if name.startswith("`") and name.endswith("`"):
        name = name[1:-1]
    return name


def _canonicalize_name(name: str) -> str:
    """
    Canonicalize table/alias names for matching:
      - strip quotes/backticks
      - drop directory/file extension if present
      - lowercase and strip
    """
    if not name:
        return ""
    name = _strip_quotes(name)
    # drop schema prefixes
    if "." in name:
        name = name.split(".")[-1]
    # replace any non-word sequences with underscore
    name = re.sub(r"[^\w]+", "_", name)
    return name.lower().strip()


# -------------------------------
# SQL extraction helpers
# -------------------------------
def extract_sql_from_text(text: str) -> str:
    """
    Extract SQL from an LLM response.
    Strategy:
      1. Prefer fenced ```sql ... ``` blocks.
      2. Otherwise prefer fenced code blocks with SELECT inside.
      3. Otherwise return the first SELECT ... (until semicolon or end).
      4. Otherwise return trimmed text.
    """
    if not text:
        return ""

    # 1) fenced code block with 'sql' language
    m = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if m:
        sql = m.group(1).strip()
        logger.debug("extract_sql_from_text: found fenced sql block")
        return sql

    # 2) fenced code block without language but with SELECT
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        cand = m.group(1).strip()
        if re.search(r"\bSELECT\b", cand, re.IGNORECASE):
            logger.debug("extract_sql_from_text: found fenced block with SELECT")
            return cand

    # 3) first SELECT ... ; or end
    m = re.search(r"(SELECT\b.*?)(;|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        sql = m.group(1).strip()
        logger.debug("extract_sql_from_text: found inline SELECT")
        return sql

    # 4) fallback
    logger.debug("extract_sql_from_text: falling back to full text")
    return text.strip()


def _parse_with_sqlglot(sql: str):
    """
    Helper: return parsed tree or None if parse fails.
    """
    try:
        import sqlglot  # type: ignore
        parsed = sqlglot.parse_one(sql, read=None)
        return parsed
    except Exception as e:
        logger.debug("sqlglot.parse_one failed: %s", e)
        return None


def extract_table_and_column_tokens(sql: str) -> Tuple[Set[str], Set[str]]:
    """
    Return two sets: (tables_set, columns_set) extracted from SQL.
    Uses sqlglot if available for robust parsing; falls back to regex heuristics.
    Extracted values are canonicalized (lowercase, stripped).
    """
    if not sql:
        return set(), set()

    tables: Set[str] = set()
    cols: Set[str] = set()

    if _SQLGLOT_AVAILABLE:
        try:
            parsed = _parse_with_sqlglot(sql)
            if parsed is not None:
                # lazy import of expressions
                from sqlglot.expressions import Table, Column, Alias  # type: ignore

                # Tables (FROM, JOIN)
                for t in parsed.find_all(Table):
                    try:
                        # Table nodes often expose .this or .name
                        name = getattr(t, "name", None) or getattr(t, "this", None)
                        if name is None:
                            name = t.sql() if hasattr(t, "sql") else str(t)
                        name = _canonicalize_name(str(name))
                        if name:
                            tables.add(name)
                    except Exception:
                        continue

                # Aliases (give alias names as possible table refs)
                for a in parsed.find_all(Alias):
                    try:
                        alias_name = None
                        try:
                            # sqlglot >=23: a.alias is Identifier
                            if hasattr(a, "alias") and a.alias:
                                alias_name = getattr(a.alias, "name", None) or str(a.alias)
                        except Exception:
                            alias_name = None

                        # sqlglot older versions: alias under args
                        if not alias_name and "alias" in getattr(a, "args", {}):
                            alias_arg = a.args.get("alias")
                            if alias_arg:
                                alias_name = getattr(alias_arg, "name", None) or str(alias_arg)

                        if alias_name:
                            alias_name = _canonicalize_name(str(alias_name))
                            if alias_name:
                                tables.add(alias_name)
                    except Exception:
                        pass

                # Columns
                for c in parsed.find_all(Column):
                    try:
                        # Column nodes often expose .name or .this
                        col_name = getattr(c, "name", None) or getattr(c, "this", None)
                        if col_name is None:
                            token_text = c.sql() if hasattr(c, "sql") else str(c)
                            # take last part after dot
                            if "." in token_text:
                                col_name = token_text.split(".")[-1]
                            else:
                                col_name = token_text
                        col_name = _canonicalize_name(str(col_name))
                        if col_name:
                            cols.add(col_name)
                        # if qualifier present, add to tables set (use only Column.table)
                        try:
                            qual = getattr(c, "table", None)
                            if qual:
                                # qual may be Identifier; extract .name if present
                                qname = getattr(qual, "name", None) or str(qual)
                                qname = _canonicalize_name(str(qname))
                                if qname:
                                    tables.add(qname)
                        except Exception:
                            pass
                    except Exception:
                        continue

                return tables, cols
        except Exception as e:
            logger.debug("sqlglot extraction failed: %s", e, exc_info=True)
            # fallthrough to regex fallback

    # Regex fallback for table names
    table_tokens = re.findall(r"\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+([^\s,;()]+)", sql, re.IGNORECASE)
    for tok in table_tokens:
        nm = _canonicalize_name(tok.split()[0])
        if nm:
            tables.add(nm)

    # Regex fallback for columns in SELECT list
    select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql, re.IGNORECASE | re.DOTALL)
    if select_match:
        cols_text = select_match.group(1)
        # split on commas not inside parentheses (basic)
        parts = re.split(r",(?![^\(]*\))", cols_text)
        for p in parts:
            p = p.strip()
            # remove trailing "AS alias" or aliasing
            p = re.sub(r"\s+AS\s+.+$", "", p, flags=re.IGNORECASE)
            # remove function wrappers like COUNT(col) -> col (take inner token)
            fn_inner = re.search(r"[A-Za-z_][\w]*\s*\(\s*([A-Za-z0-9_\.\*]+)\s*\)", p)
            token = None
            if fn_inner:
                token = fn_inner.group(1)
            else:
                m = re.search(r"([A-Za-z_][A-Za-z0-9_\.]*)\b", p)
                if m:
                    token = m.group(1)

            if not token:
                continue

            # ignore wildcards & function names
            token_l = token.lower()
            if token_l == "*" or token_l in {"count", "sum", "avg", "max", "min", "distinct"}:
                continue

            # take last part of qualified token (table.col -> col)
            token = token.split(".")[-1]
            token = _canonicalize_name(token)
            if token:
                cols.add(token)

    return tables, cols


def extract_table_names_from_sql(sql: str) -> List[str]:
    """
    Return a list of referenced table tokens (canonicalized) in the SQL.
    """
    tset, _ = extract_table_and_column_tokens(sql)
    return sorted(tset)


def extract_column_names_from_sql(sql: str) -> List[str]:
    """
    Return a list of referenced column tokens (canonicalized) in the SQL.
    """
    _, cset = extract_table_and_column_tokens(sql)
    return sorted(cset)


# -------------------------------
# Schema matching helpers
# -------------------------------
def _build_name_index_from_schemas(schemas: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Given SchemaStore._store-like mapping, build a lookup:
      lower_name -> store_key
    Includes keys: store_key, canonical, friendly, aliases, and basename of path.
    """
    index: Dict[str, str] = {}
    if not schemas:
        return index

    for store_key, meta in schemas.items():
        try:
            # key itself
            index[_canonicalize_name(store_key)] = store_key
            if isinstance(meta, dict):
                canonical = meta.get("canonical") or meta.get("canonical_name") or meta.get("table_name")
                if canonical:
                    index[_canonicalize_name(canonical)] = store_key
                friendly = meta.get("friendly") or meta.get("table_name")
                if friendly:
                    index[_canonicalize_name(friendly)] = store_key
                aliases = meta.get("aliases") or []
                if isinstance(aliases, (list, tuple)):
                    for a in aliases:
                        if a:
                            index[_canonicalize_name(a)] = store_key
                path = meta.get("path")
                if path:
                    base = _canonicalize_name(path.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0])
                    if base:
                        index[base] = store_key
            else:
                # meta not dict: treat as simple list (older shape)
                pass
        except Exception:
            continue
    return index


def find_matching_schema_key(name: str, schemas: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """
    Try to find a store key in schemas that matches 'name'.
    Returns the store_key (the key used in SchemaStore._store) or None.

    Matching strategy:
      - canonicalize input, exact lookup in name index
      - substring/contains checks
      - fuzzy matching using difflib (if no exact)
    """
    if not name or not schemas:
        return None
    import difflib

    tgt = _canonicalize_name(name)
    if not tgt:
        return None

    index = _build_name_index_from_schemas(schemas)
    if not index:
        return None

    # exact
    if tgt in index:
        return index[tgt]

    # contains/substr matches
    for k in index.keys():
        if tgt == k or tgt in k or k in tgt:
            return index[k]

    # fuzzy
    candidates = list(index.keys())
    match = difflib.get_close_matches(tgt, candidates, n=1, cutoff=0.7)
    if match:
        return index[match[0]]

    return None


# -------------------------------
# Validation helpers used by ValidateNode
# -------------------------------
def validate_tables_in_sql(sql: str, schemas: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Return list of table names referenced in SQL that are NOT present in schemas.
    This function returns the canonicalized missing tokens (lowercase).
    """
    if not sql:
        return []
    referenced = set(extract_table_names_from_sql(sql))
    missing = []
    for r in referenced:
        if not find_matching_schema_key(r, schemas):
            missing.append(r)
    return missing


def validate_columns_in_sql(sql: str, schemas: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Return list of columns referenced in SQL that are not present in the (resolved) schemas.
    Uses extraction + lookup; returns lowercased column tokens.
    """
    if not sql:
        return []
    referenced_cols = set(extract_column_names_from_sql(sql))
    if not referenced_cols:
        return []

    # Build map store_key -> set(lowered column names)
    cols_map: Dict[str, Set[str]] = {}
    for store_key, meta in (schemas or {}).items():
        try:
            cols = []
            if isinstance(meta, dict):
                # Prefer normalized tokens for matching; fall back to original column names
                cols = meta.get("columns_normalized") or [ _canonicalize_name(str(c)) for c in (meta.get("columns") or []) ]
            elif isinstance(meta, (list, tuple)):
                cols = meta
            cols_map[store_key] = { _canonicalize_name(str(c)) for c in cols if c is not None }
        except Exception:
            cols_map[store_key] = set()

    # union all cols
    union_cols = set().union(*cols_map.values()) if cols_map else set()

    missing = []
    for col in referenced_cols:
        # skip wildcard and obvious functions or numeric tokens
        if not col or col == "*" or col.isdigit():
            continue
        if col.lower() in {"count", "sum", "avg", "max", "min", "distinct"}:
            continue
        if col in union_cols:
            continue
        missing.append(col)
    return missing


# -------------------------------
# Safety helpers
# -------------------------------
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


def is_select_query(sql: str) -> bool:
    """
    Return True if SQL is a top-level SELECT statement (basic check).
    Uses sqlglot if available for better detection.
    """
    if not sql:
        return False
    s = sql.strip()
    # quick regex check
    if re.match(r"^\s*\(?\s*SELECT\b", s, re.IGNORECASE):
        return True
    if _SQLGLOT_AVAILABLE:
        try:
            parsed = _parse_with_sqlglot(sql)
            if parsed is not None:
                # top-level expression class name - e.g., Select
                name = parsed.__class__.__name__.lower()
                return "select" in name
        except Exception:
            pass
    return False


def limit_sql_rows(sql: str, limit: int = 100) -> str:
    """
    Return a SQL that enforces a row limit by wrapping the original query.
    """
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        return sql
    return f"SELECT * FROM ({sql}) AS _sub LIMIT {int(limit)}"


def exceeds_row_limit(sql: str, limit: int) -> bool:
    """
    Check if SQL has a LIMIT clause exceeding the specified limit.
    Basic regex; if ambiguous, returns False.
    """
    if not sql:
        return False
    m = re.search(r"\bLIMIT\s+(\d+)", sql, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1)) > int(limit)
        except Exception:
            return False
    return False


def check_forbidden_tables(sql: str, forbidden: List[str]) -> List[str]:
    """
    Return list of forbidden table tokens used in the SQL (case-insensitive match on token boundaries).
    """
    forbidden_used = []
    if not sql or not forbidden:
        return forbidden_used
    for tbl in forbidden:
        if re.search(rf"\b{re.escape(tbl)}\b", sql, re.IGNORECASE):
            forbidden_used.append(tbl)
    return forbidden_used


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
        meta = d.get("meta") or {}
        title = meta.get("table_name") or meta.get("title") or d.get("id") or "Document"
        score = d.get("score")
        text = d.get("text") or meta.get("text") or ""
        if score is not None:
            parts.append(f"{title} (score={score:.3f}):\n{text}")
        else:
            parts.append(f"{title}:\n{text}")
    return "\n\n".join(parts)


def flatten_schema(schema: Any) -> str:
    """
    Convert schema mapping or list into readable text for prompts/UI.
    Accepts SchemaStore._store (dict) or CSVLoader.load_and_extract() list of metadata dicts.
    """
    if not schema:
        return ""
    parts = []

    if isinstance(schema, list):
        # list of metadata dicts
        for meta in schema:
            try:
                name = meta.get("canonical_name") or meta.get("table_name") or (meta.get("path") or "").split("/")[-1]
                cols = meta.get("columns") or []
                parts.append(f"Table: {name} | Columns: {', '.join(map(str, cols))}")
                sample_rows = meta.get("sample_rows") or []
                if sample_rows:
                    parts.append("  sample rows: " + preview_sample_rows(sample_rows, max_preview=2))
            except Exception:
                continue
        return "\n".join(parts)

    if isinstance(schema, dict):
        for table, meta in schema.items():
            try:
                if isinstance(meta, dict):
                    # show human-readable original column names if available, otherwise normalized tokens
                    cols_display = meta.get("columns") or meta.get("columns_normalized") or []
                    parts.append(f"Table: {meta.get('canonical') or table} | Columns: {', '.join(map(str, cols_display))}")
                    sample_rows = meta.get("sample_rows") or []
                    if sample_rows:
                        parts.append("  sample rows: " + preview_sample_rows(sample_rows, max_preview=2))
                else:
                    # meta is simple list of columns
                    cols = meta or []
                    parts.append(f"Table: {table} | Columns: {', '.join(map(str, cols))}")
            except Exception:
                continue
        return "\n".join(parts)

    # unknown shape
    return str(schema)


def preview_sample_rows(rows: List[Dict[str, Any]], max_preview: int = 3) -> str:
    """
    Render a small human-friendly preview of sample rows.
    """
    if not rows:
        return ""
    preview = rows[:max_preview]
    lines = []
    for r in preview:
        if isinstance(r, dict):
            kvs = ", ".join(f"{k}={v}" for k, v in r.items())
        elif isinstance(r, (list, tuple)):
            kvs = ", ".join(str(x) for x in r)
        else:
            kvs = str(r)
        lines.append(kvs)
    return " | ".join(lines)


def build_prompt(
    user_query: str,
    schemas: Any,
    retrieved_docs: Optional[List[Dict[str, Any]]] = None,
    few_shot: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Build prompt with:
      - few-shot examples
      - flattened schema
      - retrieved documents
      - user query

    `schemas` can be either SchemaStore._store (dict) or list-of-metadata from CSVLoader.
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

    # Safety instruction: force model to use only canonical table/column names provided above.
    # This reduces hallucination where the model invents table names such as `apps`.
    parts.append(
        "IMPORTANT: Use ONLY the table and column names listed above exactly as written. "
        "Do NOT invent or use any other table names (for example: 'apps')."
    )

    prompt = "\n\n".join(parts)
    return prompt
