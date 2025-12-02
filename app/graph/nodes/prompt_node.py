# app/graph/nodes/prompt_node.py
import sys
from typing import Dict, Any, List, Optional, Callable, Iterable, Tuple

from app.logger import get_logger
from app.exception import CustomException
from app import utils

logger = get_logger("prompt_node")


def _as_store_map(schemas: Any) -> Dict[str, Dict[str, Any]]:
    """
    Normalize different possible shapes of 'schemas' into a mapping:
      store_key -> metadata dict with keys:
        - canonical (canonical name)
        - friendly (friendly name)
        - aliases (list)
        - path (optional)
        - columns (list of original headers)
        - columns_normalized (optional list of normalized column names)
        - sample_rows (optional)
    Accepts:
      - dict of store_key -> metadata (new SchemaStore._store)
      - list of metadata dicts returned by CSVLoader.load_and_extract()
      - legacy dict: table -> {"columns": [...], ...}
    """
    out: Dict[str, Dict[str, Any]] = {}

    if not schemas:
        return out

    # If input is a list of metadata dicts (CSVLoader.load_and_extract output)
    if isinstance(schemas, list):
        for meta in schemas:
            try:
                if not isinstance(meta, dict):
                    continue
                path = meta.get("path")
                # determine store key: prefer canonical_name or canonical_name fallback to filename stem
                canonical = meta.get("canonical_name") or meta.get("canonical") or meta.get("table_name")
                if not canonical:
                    if path:
                        canonical = meta.get("canonical_name") or meta.get("canonical") or (meta.get("original_name") or meta.get("path") or "")
                        canonical = canonical.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0]
                    else:
                        canonical = meta.get("table_name") or "table"
                store_key = canonical
                out[store_key] = {
                    "canonical": canonical,
                    "friendly": meta.get("table_name") or canonical,
                    "aliases": meta.get("aliases") or [],
                    "path": path,
                    "columns": meta.get("columns") or [],
                    "columns_normalized": meta.get("columns_normalized") or [],
                    "sample_rows": meta.get("sample_rows") or [],
                }
            except Exception:
                logger.exception("Failed to normalize schema metadata item; skipping", exc_info=True)
        return out

    # If input is a dict-like mapping (SchemaStore._store)
    if isinstance(schemas, dict):
        for key, meta in schemas.items():
            try:
                if not isinstance(meta, dict):
                    # legacy shape: value may be columns list -> convert
                    if isinstance(meta, list):
                        out[key] = {
                            "canonical": key,
                            "friendly": key,
                            "aliases": [],
                            "path": None,
                            "columns": meta,
                            "columns_normalized": [],
                            "sample_rows": [],
                        }
                    else:
                        continue
                else:
                    canonical = meta.get("canonical") or meta.get("canonical_name") or key
                    friendly = meta.get("friendly") or meta.get("table_name") or canonical
                    aliases = meta.get("aliases") or []
                    path = meta.get("path")
                    columns = meta.get("columns") or []
                    columns_normalized = meta.get("columns_normalized") or []
                    sample_rows = meta.get("sample_rows") or meta.get("samples") or []
                    out[key] = {
                        "canonical": canonical,
                        "friendly": friendly,
                        "aliases": aliases,
                        "path": path,
                        "columns": columns,
                        "columns_normalized": columns_normalized,
                        "sample_rows": sample_rows,
                    }
            except Exception:
                logger.exception("Failed to normalize schema entry for key=%s; skipping", key, exc_info=True)
        return out

    # Fallback: unsupported type -> return empty mapping
    return out


def _format_sample_row(row: Any, cols: Optional[List[str]] = None) -> str:
    """
    Make a compact one-line representation of a sample row.
    If row is dict -> key=value pairs; if list/tuple -> join using cols if provided.
    """
    try:
        if isinstance(row, dict):
            items = []
            for k, v in row.items():
                items.append(f"{k}={v}")
            return ", ".join(items)
        if isinstance(row, (list, tuple)):
            if cols and len(cols) == len(row):
                pairs = [f"{cols[i]}={row[i]}" for i in range(len(row))]
                return ", ".join(pairs)
            return ", ".join([str(x) for x in row])
        return str(row)
    except Exception:
        return str(row)


def _default_prompt_builder(
    user_query: str,
    schemas: Dict[str, Dict[str, Any]],
    retrieved_docs: List[Dict[str, Any]],
    few_shot: Optional[List[Dict[str, str]]] = None,
    max_tables: int = 10,
) -> str:
    """
    Deterministic prompt builder.

    Produces:
      - instructions (use canonical names)
      - compact Schema block (canonical, friendly, aliases, columns, normalized columns)
      - short machine-friendly schema snippet (one-line per table)
      - optional retrieved docs block
      - few-shot examples
      - user question and rules

    This prompt intentionally shows canonical names and explicit column lists so the LLM
    is guided to use exact identifiers that the validator expects.
    """
    parts: List[str] = []
    parts.append("You are a SQL assistant. Use the exact **canonical** table and column names shown in the Schema block when writing SQL.")
    parts.append("")
    # Schema block (compact)
    if schemas:
        parts.append("Schema (use these exact names):")
        count = 0
        # schema lines: also produce a machine-friendly compact snippet to help LLM reliably reference names
        snippet_lines: List[str] = []
        for store_key, info in schemas.items():
            if count >= max_tables:
                parts.append(f"... (omitted {len(schemas) - max_tables} more tables) ...")
                break
            canonical = info.get("canonical") or store_key
            friendly = info.get("friendly") or canonical
            aliases = info.get("aliases") or []
            columns = info.get("columns") or []
            cols_norm = info.get("columns_normalized") or []
            sample_rows = info.get("sample_rows") or []

            parts.append(f"Table (canonical): {canonical}")
            parts.append(f"Friendly name: {friendly}")
            if aliases:
                parts.append(f"Aliases: {', '.join(aliases)}")
            if columns:
                parts.append("Columns: " + ", ".join(columns))
            if cols_norm:
                parts.append("Columns (normalized): " + ", ".join(cols_norm))
            if sample_rows:
                parts.append("Sample rows:")
                for r in sample_rows[:3]:
                    parts.append(" - " + _format_sample_row(r, columns))
            parts.append("")  # blank line
            snippet_lines.append(f"{canonical}({', '.join(columns)})")
            count += 1

        # machine-friendly schema snippet (compact, useful for LLM to copy)
        if snippet_lines:
            parts.append("Schema snippet (compact):")
            parts.append("; ".join(snippet_lines))
            parts.append("")
    else:
        parts.append("Schema: (no schema information available)")

    # Retrieved docs (RAG)
    if retrieved_docs:
        parts.append("Relevant documents:")
        for doc in retrieved_docs[:5]:
            title = (doc.get("meta") or {}).get("title") or doc.get("id") or "<doc>"
            snippet = doc.get("text") or doc.get("content") or ""
            # one-liner snippet
            safe_snip = snippet.replace("\n", " ").strip()
            if len(safe_snip) > 200:
                safe_snip = safe_snip[:197] + "..."
            parts.append(f"- {title}: {safe_snip}")
        parts.append("")

    # Few-shot examples
    if few_shot:
        parts.append("Examples (do not repeat these exact SQLs unless appropriate):")
        for ex in few_shot:
            if isinstance(ex, dict):
                u = ex.get("user") or ex.get("query") or ""
                s = ex.get("sql") or ex.get("assistant") or ""
                parts.append("User: " + u)
                parts.append("SQL Example:")
                parts.append("```sql")
                parts.append(s.strip())
                parts.append("```")
                parts.append("")

    # User query
    parts.append("User question:")
    parts.append(user_query.strip())
    parts.append("")
    parts.append(
        "Rules:\n"
        "- Output a single valid SQL statement (preferably in a ```sql fenced block). Use only canonical table and column names shown above.\n"
        "- If you must reference a column not listed, explain why and avoid fabricating column names.\n"
        "- Only produce read-only SELECT queries; do not produce INSERT/UPDATE/DELETE statements.\n"
        "- If the query may return a large number of rows, add a LIMIT or explain that you will apply a limit."
    )
    parts.append("")
    parts.append("If you cannot write a valid SQL using the given schema, explain in one short sentence why.")

    return "\n".join(parts)


class PromptNode:
    """
    Build prompts for the LLM from user query, schema metadata, retrieved docs and few-shot examples.

    Public API:
      run(user_query, schemas, retrieved_docs=None, few_shot=None) -> prompt string

    `schemas` can be:
      - the SchemaStore._store dict (store_key -> metadata)
      - a list of metadata dicts (CSVLoader.load_and_extract output)
      - a legacy dict mapping table -> {"columns": [...]}
    """
    def __init__(self, prompt_builder: Optional[Callable[..., str]] = None):
        try:
            if prompt_builder:
                self._builder = prompt_builder
            else:
                # prefer utils.build_prompt if implemented; else fallback
                self._builder = getattr(utils, "build_prompt", None) or _default_prompt_builder
        except Exception as e:
            logger.exception("Failed to initialize PromptNode")
            raise CustomException(e, sys)

    def run(
        self,
        user_query: str,
        schemas: Any,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        few_shot: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Build and return the prompt string.
        """
        try:
            retrieved_docs = retrieved_docs or []
            # normalize schemas into store_map
            store_map = _as_store_map(schemas)

            # prepare a clean view for the builder: keep necessary keys only and cap sample rows
            clean_schemas: Dict[str, Dict[str, Any]] = {}
            for k, info in store_map.items():
                cols = info.get("columns") or []
                cols_norm = info.get("columns_normalized") or []
                samples = info.get("sample_rows") or []
                # normalize sample rows to safe strings/dicts (max 3)
                normalized_samples = []
                for r in samples[:3]:
                    try:
                        if isinstance(r, dict):
                            normalized_samples.append({str(k): str(v) for k, v in r.items()})
                        elif isinstance(r, (list, tuple)):
                            if cols and len(cols) == len(r):
                                normalized_samples.append({cols[i]: str(r[i]) for i in range(len(r))})
                            else:
                                normalized_samples.append([str(x) for x in r])
                        else:
                            normalized_samples.append(str(r))
                    except Exception:
                        normalized_samples.append(str(r))
                clean_schemas[k] = {
                    "canonical": info.get("canonical") or k,
                    "friendly": info.get("friendly") or info.get("canonical") or k,
                    "aliases": info.get("aliases") or [],
                    "path": info.get("path"),
                    "columns": cols,
                    "columns_normalized": cols_norm,
                    "sample_rows": normalized_samples,
                }

            # build prompt
            prompt = self._builder(user_query, clean_schemas, retrieved_docs or [], few_shot)
            logger.info("PromptNode: Prompt built successfully")
            return prompt
        except CustomException:
            raise
        except Exception as e:
            logger.exception("PromptNode.run failed; falling back to default builder", exc_info=True)
            try:
                return _default_prompt_builder(user_query, _as_store_map(schemas), retrieved_docs or [], few_shot)
            except Exception as ex:
                logger.exception("PromptNode fallback also failed")
                raise CustomException(ex, sys)
