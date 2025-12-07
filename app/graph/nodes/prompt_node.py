# app/graph/nodes/prompt_node.py
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
      - This node is intentionally simple; downstream systems can replace with templating logic.
    """

    def __init__(self, tools: Optional[Tools] = None, max_context_chars: int = 4000):
        try:
            self._tools = tools or Tools()
            self.max_context_chars = max_context_chars
        except Exception as e:
            logger.exception("Failed to initialize PromptNode")
            raise CustomException(e, sys)

    def run(self, question: str, schema_context: Dict[str, Dict[str, Any]], retrieved: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Returns {"prompt": <str>, "pieces": {"schema":..., "evidence":..., "question":...}}
        """
        try:
            pieces: Dict[str, Any] = {}
            # Build schema snippet
            schema_snippets: List[str] = []
            for table, info in (schema_context or {}).items():
                cols = info.get("columns") or []
                sample = info.get("sample_rows") or []
                col_str = ", ".join(cols[:20]) if cols else "(no columns found)"
                sample_str = ""
                if sample:
                    # show up to 2 sample rows in compact form
                    sample_preview = []
                    for r in sample[:2]:
                        if isinstance(r, dict):
                            kv = ";".join([f"{k}={v}" for k, v in list(r.items())[:4]])
                            sample_preview.append(kv)
                        else:
                            sample_preview.append(str(r))
                    sample_str = " Samples: " + " | ".join(sample_preview)
                schema_snippets.append(f"Table `{table}`: columns: {col_str}.{sample_str}")
            pieces["schema"] = "\n".join(schema_snippets) if schema_snippets else "No schema context available."

            # Evidence / retrieved
            evidence_texts: List[str] = []
            if retrieved:
                for r in retrieved[:10]:
                    t = r.get("text") or ""
                    meta = r.get("meta") or {}
                    tag = meta.get("canonical") or meta.get("table") or meta.get("source") or ""
                    evidence_texts.append(f"[score={r.get('score'):.4f}] ({tag}) {t[:300]}")
            pieces["evidence"] = "\n".join(evidence_texts) if evidence_texts else "No retrieved evidence."

            pieces["question"] = question

            # Compose prompt: keep things short to respect LLM context
            prompt_parts = [
                "You are an assistant that answers SQL questions over uploaded CSVs.",
                "Schema information (tables & columns):",
                pieces["schema"],
                "Relevant data snippets:",
                pieces["evidence"],
                "User question:",
                pieces["question"],
                "Constraints: only use the uploaded CSVs. If unsure, ask for clarification.",
            ]
            prompt = "\n\n".join([p for p in prompt_parts if p])

            # Trim to max_context_chars naively (prefer to trim evidence)
            if len(prompt) > self.max_context_chars:
                # preserve schema + question, trim evidence
                base = "\n\n".join([prompt_parts[0], prompt_parts[1], prompt_parts[2], prompt_parts[5]])
                evidence = pieces["evidence"]
                allowed_evidence = max(0, self.max_context_chars - len(base) - 200)
                if allowed_evidence < 0:
                    prompt = base[: self.max_context_chars]
                else:
                    pieces["evidence"] = evidence[:allowed_evidence]
                    prompt = "\n\n".join([prompt_parts[0], prompt_parts[1], prompt_parts[2], "Relevant data snippets:", pieces["evidence"], prompt_parts[5]])
            return {"prompt": prompt, "pieces": pieces}
        except Exception as e:
            logger.exception("PromptNode.run failed")
            raise CustomException(e, sys)
