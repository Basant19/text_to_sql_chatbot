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
    PromptNode
    ----------
    Responsibilities:
      - Build a final LLM prompt from:
          * user question
          * canonical schema context
          * optional retrieved evidence
      - Enforce strict SQL-only constraints
      - Be GraphBuilder-safe (keyword-only run signature)
    """

    def __init__(self, tools: Optional[Tools] = None, max_context_chars: int = 4000):
        try:
            self._tools = tools or Tools()
            self.max_context_chars = max(500, int(max_context_chars))
            LOG.info("PromptNode initialized (max_context_chars=%s)", self.max_context_chars)
        except Exception as e:
            logger.exception("PromptNode initialization failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # IMPORTANT: keyword-only signature (GraphBuilder compatibility)
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        question: Optional[str] = None,
        schema_context: Optional[Dict[str, Dict[str, Any]]] = None,
        retrieved: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
              "prompt": <final_prompt_str>,
              "pieces": {
                  "schema": <str>,
                  "evidence": <str>,
                  "question": <str>
              }
            }
        """
        try:
            # -----------------------------
            # Normalize inputs defensively
            # -----------------------------
            question = (question or "").strip()
            schema_context = schema_context if isinstance(schema_context, dict) else {}
            retrieved = retrieved if isinstance(retrieved, list) else []

            pieces: Dict[str, Any] = {}

            # -----------------------------
            # Schema section
            # -----------------------------
            schema_lines: List[str] = []
            table_columns: Dict[str, List[str]] = {}

            for table, info in schema_context.items():
                if not isinstance(info, dict):
                    continue

                cols = info.get("columns") or []
                cols = [str(c) for c in cols]
                table_columns[table] = cols

                col_str = ", ".join(cols[:50]) if cols else "(no columns found)"

                sample_rows = info.get("sample_rows") or []
                sample_preview: List[str] = []

                for row in sample_rows[:2]:
                    if isinstance(row, dict):
                        kvs = [f"{k}={v}" for k, v in list(row.items())[:4]]
                        sample_preview.append("; ".join(kvs))
                    else:
                        sample_preview.append(str(row))

                sample_str = f" Samples: {' | '.join(sample_preview)}" if sample_preview else ""
                schema_lines.append(f"Table `{table}`: columns: {col_str}.{sample_str}")

            pieces["schema"] = (
                "\n".join(schema_lines)
                if schema_lines
                else "No schema context available."
            )

            # -----------------------------
            # Retrieved evidence section
            # -----------------------------
            evidence_lines: List[str] = []

            for r in retrieved[:10]:
                if not isinstance(r, dict):
                    continue

                text = str(r.get("text", ""))[:300]
                meta = r.get("meta") or {}
                tag = (
                    meta.get("canonical")
                    or meta.get("table")
                    or meta.get("source")
                    or ""
                )
                score = r.get("score")
                score_str = f"{float(score):.4f}" if score is not None else "n/a"

                evidence_lines.append(f"[score={score_str}] ({tag}) {text}")

            pieces["evidence"] = (
                "\n".join(evidence_lines)
                if evidence_lines
                else "No retrieved evidence."
            )

            pieces["question"] = question or "No user question provided."

            # -----------------------------
            # Allowed tables & columns
            # -----------------------------
            allow_lines: List[str] = []

            if table_columns:
                allow_lines.append(
                    "ALLOWED TABLES AND COLUMNS (use only these names exactly):"
                )
                for tbl, cols in table_columns.items():
                    col_display = ", ".join(cols[:200]) if cols else "(no columns)"
                    allow_lines.append(f"- {tbl}: {col_display}")
            else:
                allow_lines.append(
                    "No schema information available; do NOT invent tables or columns."
                )

            allowed_section = "\n".join(allow_lines)

            # -----------------------------
            # Hard constraints
            # -----------------------------
            constraints = "\n".join(
                [
                    "IMPORTANT CONSTRAINTS:",
                    "- Output ONLY ONE SQL SELECT statement.",
                    "- No INSERT, UPDATE, DELETE, DDL, or multiple statements.",
                    "- Use ONLY tables and columns listed above.",
                    "- If the question cannot be answered, output exactly: CANNOT_ANSWER",
                    "- Return ONLY SQL, no explanations.",
                ]
            )

            # -----------------------------
            # Final prompt assembly
            # -----------------------------
            prompt_parts = [
                "You are an assistant that generates SQL over CSV-backed tables.",
                constraints,
                allowed_section,
                "Relevant data snippets:",
                pieces["evidence"],
                "User question:",
                pieces["question"],
            ]

            prompt = "\n\n".join(prompt_parts)

            # -----------------------------
            # Safe trimming (evidence first)
            # -----------------------------
            if len(prompt) > self.max_context_chars:
                LOG.warning("Prompt exceeds max_context_chars, trimming evidence")
                base_parts = prompt_parts[:3] + ["User question:", pieces["question"]]
                base_prompt = "\n\n".join(base_parts)

                remaining = self.max_context_chars - len(base_prompt) - 20
                if remaining > 0:
                    trimmed_evidence = pieces["evidence"][:remaining]
                    prompt = "\n\n".join(
                        [
                            prompt_parts[0],
                            prompt_parts[1],
                            prompt_parts[2],
                            "Relevant data snippets:",
                            trimmed_evidence,
                            "User question:",
                            pieces["question"],
                        ]
                    )
                else:
                    prompt = base_prompt[: self.max_context_chars]

            return {
                "prompt": prompt,
                "pieces": pieces,
            }

        except Exception as e:
            logger.exception("PromptNode.run failed")
            raise CustomException(e, sys)
