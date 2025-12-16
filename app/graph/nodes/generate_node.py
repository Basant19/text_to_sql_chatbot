#D:\text_to_sql_bot\app\graph\nodes\generate_node.py
from __future__ import annotations

import sys
import logging
import re
import json
from typing import Dict, Any, Optional, List

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

logger = get_logger("generate_node")
LOG = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _provider_name(provider: object) -> str:
    try:
        return provider.__class__.__name__
    except Exception:
        return str(type(provider))


# ------------------------------------------------------------------
# Generate Node
# ------------------------------------------------------------------
class GenerateNode:
    """
    GenerateNode
    ------------
    Responsibilities:
      - Build an LLM prompt from query + schema + retrieved context
      - Call the LLM provider (if configured)
      - ALWAYS return a syntactically valid SQL string
        so downstream nodes do not silently fail
    """

    def __init__(
        self,
        tools: Optional[Tools] = None,
        provider_client: Optional[Any] = None,
    ):
        try:
            self._tools = tools or Tools()

            # 🔥 SINGLE SOURCE OF TRUTH FOR PROVIDER
            if provider_client is not None:
                self._provider = provider_client
            elif hasattr(self._tools, "get_provider_client"):
                self._provider = self._tools.get_provider_client()
            else:
                self._provider = None

            LOG.info(
                "GenerateNode initialized (provider=%s)",
                _provider_name(self._provider) if self._provider else None,
            )

        except Exception as e:
            logger.exception("GenerateNode initialization failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        query: str,
        schemas: Dict[str, Any],
        retrieved: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        parts: List[str] = [
            "You are a Text-to-SQL generator.",
            "Generate a SAFE, READ-ONLY SQL SELECT query.",
            "Do NOT use INSERT, UPDATE, DELETE, DROP, or ALTER.",
            "",
            "User Question:",
            query,
            "",
            "Database Schemas:",
            json.dumps(schemas, indent=2),
            "",
        ]

        if retrieved:
            parts.append("Relevant Context:")
            for r in retrieved:
                txt = r.get("text")
                if txt:
                    parts.append(f"- {txt}")

        parts.append("")
        parts.append("Return ONLY the SQL query. No explanation.")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Keyword-only run (GraphBuilder safe)
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        query: Optional[str] = None,
        schemas: Optional[Dict[str, Any]] = None,
        retrieved: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            "sql": "<generated sql>",
            "raw": <provider raw output>,
            "metadata": {...}
          }
        """

        try:
            # ------------------------------------------------------------------
            # Schema auto-resolution (CRITICAL FIX)
            # ------------------------------------------------------------------
            if schemas is None and self._tools:
                schemas = {}
                for name in self._tools.list_csvs():
                    cols = self._tools.get_schema(name)
                    if cols:
                        schemas[name] = cols

            if not query or not schemas:
                LOG.warning(
                    "GenerateNode: missing input | query=%s | schemas=%s",
                    query,
                    bool(schemas),
                )
                return {
                    "sql": "",
                    "raw": None,
                    "metadata": {"error": "missing_input"},
                }

            # ------------------------------------------------------------------
            # 🔥 NO PROVIDER → SAFE FALLBACK SQL (KEY FIX)
            # ------------------------------------------------------------------
            if not self._provider:
                table = next(iter(schemas.keys()))
                fallback_sql = f"SELECT * FROM {table} LIMIT 10"

                LOG.warning(
                    "GenerateNode: no LLM provider configured, using fallback SQL (%s)",
                    table,
                )

                return {
                    "sql": fallback_sql,
                    "raw": None,
                    "metadata": {
                        "warning": "no_provider_fallback",
                        "table": table,
                    },
                }

            # ------------------------------------------------------------------
            # Build prompt
            # ------------------------------------------------------------------
            prompt = self._build_prompt(query, schemas, retrieved)

            # ------------------------------------------------------------------
            # Provider call
            # ------------------------------------------------------------------
            try:
                if callable(self._provider):
                    raw = self._provider(prompt)
                elif hasattr(self._provider, "generate"):
                    raw = self._provider.generate(prompt)
                elif hasattr(self._provider, "run"):
                    raw = self._provider.run(prompt)
                else:
                    raise RuntimeError("Unsupported provider interface")

            except Exception as e:
                LOG.exception("Provider call failed; falling back to safe SQL")

                table = next(iter(schemas.keys()))
                return {
                    "sql": f"SELECT * FROM {table} LIMIT 10",
                    "raw": None,
                    "metadata": {
                        "provider": _provider_name(self._provider),
                        "provider_error": str(e),
                        "fallback": True,
                    },
                }

            # ------------------------------------------------------------------
            # Normalize output → SQL
            # ------------------------------------------------------------------
            if isinstance(raw, str):
                text = raw
            elif isinstance(raw, dict):
                text = raw.get("text") or raw.get("output") or ""
            else:
                text = str(raw)

            text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

            return {
                "sql": text,
                "raw": raw,
                "metadata": {
                    "provider": _provider_name(self._provider),
                },
            }

        except Exception as e:
            LOG.exception("GenerateNode.run failed")
            raise CustomException(e, sys)
