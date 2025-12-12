#D:\text_to_sql_bot\app\graph\nodes\prompt_node.py
from __future__ import annotations
import sys
import logging
from typing import Dict, Any, List, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

logger = get_logger("prompt_node")
LOG = logging.getLogger(__name__)


class PromptNode:
    """
    PromptNode:
      - Constructs a final prompt for the LLM using schema context, user question, and retrieved evidence.
      - Returns {"prompt": <str>, "pieces": {"schema":..., "evidence":..., "question":...}}
      - Ensures the prompt contains canonical table/column lists and a strong constraint to produce only a SELECT.
    """

    def __init__(self, tools: Optional[Tools] = None, max_context_chars: int = 4000):
        try:
            self._tools = tools or Tools()
            self.max_context_chars = int(max_context_chars)
        except Exception as e:
            logger.exception("Failed to initialize PromptNode")
            raise CustomException(e, sys)

    def run(self, question: str, schema_context: Dict[str, Dict[str, Any]], retrieved: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Returns {"prompt": <str>, "pieces": {"schema":..., "evidence":..., "question":...}}

        - `schema_context` expected shape:
             { "canonical_table_name": {"columns": ["col1","col2"], "sample_rows": [...]}, ... }
        - `retrieved` expected to be list of dicts with keys: text, score (optional), meta (optional)
        """
        try:
            pieces: Dict[str, Any] = {}

            # Build schema snippet with canonical names and columns
            schema_snippets: List[str] = []
            allowed_tables: List[str] = []
            table_columns: Dict[str, List[str]] = {}

            if schema_context:
                for table, info in (schema_context or {}).items():
                    allowed_tables.append(table)
                    cols = info.get("columns") or []
                    # normalize column list to strings (shorten if huge)
                    cols_strs = [str(c) for c in cols]
                    table_columns[table] = cols_strs
                    col_str = ", ".join(cols_strs[:50]) if cols_strs else "(no columns found)"

                    # sample rows: compact preview
                    sample = info.get("sample_rows") or []
                    sample_str = ""
                    if sample:
                        preview = []
                        for r in sample[:2]:
                            if isinstance(r, dict):
                                kv_pairs = []
                                # show first up-to-4 key/value pairs
                                for k, v in list(r.items())[:4]:
                                    kv_pairs.append(f"{k}={v}")
                                preview.append(";".join(kv_pairs))
                            else:
                                preview.append(str(r))
                        sample_str = " Samples: " + " | ".join(preview)
                    schema_snippets.append(f"Table `{table}`: columns: {col_str}.{sample_str}")
                pieces["schema"] = "\n".join(schema_snippets)
            else:
                pieces["schema"] = "No schema context available."
                table_columns = {}

            # Evidence / retrieved
            evidence_texts: List[str] = []
            if retrieved:
                for r in retrieved[:10]:
                    try:
                        t = r.get("text") if isinstance(r, dict) else str(r)
                    except Exception:
                        t = str(r)[:300]
                    meta = r.get("meta") if isinstance(r, dict) else {}
                    # meta may contain canonical table
                    tag = ""
                    try:
                        tag = meta.get("canonical") or meta.get("table") or meta.get("source") or ""
                    except Exception:
                        tag = ""
                    # score may be missing
                    score = r.get("score") if isinstance(r, dict) else None
                    score_str = f"{float(score):.4f}" if score is not None else "n/a"
                    evidence_texts.append(f"[score={score_str}] ({tag}) { (t or '')[:300] }")
            pieces["evidence"] = "\n".join(evidence_texts) if evidence_texts else "No retrieved evidence."

            pieces["question"] = question

            # Compose explicit allowed tables/columns section
            schema_allow_lines: List[str] = []
            if table_columns:
                schema_allow_lines.append("ALLOWED TABLES AND COLUMNS (use only these names exactly):")
                for tbl, cols in table_columns.items():
                    # show up to 100 columns in prompt; if many, warn
                    col_display = ", ".join(cols[:200]) if cols else "(no columns)"
                    schema_allow_lines.append(f"- {tbl}: {col_display}")
            else:
                schema_allow_lines.append("No schema information available; do not invent table or column names.")

            allowed_section = "\n".join(schema_allow_lines)

            # Strong instruction block to avoid invalid SQL
            constraints = [
                "IMPORTANT CONSTRAINTS (READ CAREFULLY):",
                "- Only produce ONE valid SQL statement, and it must be a single SELECT query (no INSERT/UPDATE/DELETE/DDL).",
                "- Do NOT reference any table or column name that is not listed in the 'ALLOWED TABLES AND COLUMNS' section above.",
                "- If the user's question cannot be answered using the listed tables/columns, say you cannot answer and ask for clarification.",
                "- Use canonical table names exactly as shown (case-insensitive matching is acceptable but prefer exact names).",
                "- Return only the SQL statement in your final answer, with no surrounding explanation (for machine consumption).",
                "- Example output (exact format):",
                "  SELECT col1, col2 FROM `your_table` WHERE col3 = 'value';",
                "- If you cannot answer, respond with: CANNOT_ANSWER (and nothing else).",
            ]
            constraints_text = "\n".join(constraints)

            # Compose prompt: order matters (instructions -> schema -> evidence -> question)
            prompt_parts = [
                "You are an assistant that answers SQL questions over uploaded CSV files.",
                constraints_text,
                allowed_section,
                "Relevant data snippets (examples from CSVs):",
                pieces["evidence"],
                "User question:",
                pieces["question"],
            ]
            prompt = "\n\n".join([p for p in prompt_parts if p is not None])

            # Trim to max_context_chars naively (prefer to trim evidence)
            if len(prompt) > self.max_context_chars:
                # keep header, constraints, allowed_section and question, trim evidence
                header = "\n\n".join([prompt_parts[0], prompt_parts[1], prompt_parts[2], "User question:", prompt_parts[-1]])
                remaining = max(0, self.max_context_chars - len(header) - 20)
                evidence = pieces["evidence"]
                if remaining <= 0:
                    # fallback: hard truncate the whole prompt
                    prompt = header[: self.max_context_chars]
                else:
                    pieces["evidence"] = evidence[:remaining]
                    prompt = "\n\n".join([prompt_parts[0], prompt_parts[1], prompt_parts[2], "Relevant data snippets:", pieces["evidence"], "User question:", pieces["question"]])

            return {"prompt": prompt, "pieces": pieces}

        except Exception as e:
            logger.exception("PromptNode.run failed")
            raise CustomException(e, sys)
