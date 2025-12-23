# D:\text_to_sql_bot\app\graph\nodes\generate_node.py
from __future__ import annotations

import sys
import json
import logging
import re
from typing import Dict, Any, Optional, List

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

logger = get_logger("generate_node")
LOG = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================
def _provider_name(provider: object) -> str:
    """
    Safely extract provider class name for logging / metadata.
    """
    try:
        return provider.__class__.__name__
    except Exception:
        return str(type(provider))


# ============================================================
# Generate Node
# ============================================================
class GenerateNode:
    """
    GenerateNode
    ============

    Responsibility
    --------------
    Convert a natural-language user query into a **SAFE, READ-ONLY SQL SELECT**
    statement that is **semantically correct** and **robust to dirty data**.

    This node is the **ONLY place** where:
    - Natural language → SQL translation happens
    - Numeric normalization rules are defined
    - Type-safe aggregation / ordering logic is enforced

    HARD ARCHITECTURAL CONTRACT
    ---------------------------
    ✔ Generates SELECT-only SQL
    ✔ NEVER executes SQL
    ✔ NEVER validates schema existence or data presence
    ✔ NEVER mutates shared state
    ✔ NEVER crashes the graph
    ✔ MUST tolerate missing context / retrievals

    INPUT (from GraphBuilder)
    -------------------------
    - user_query : str                     (REQUIRED)
    - schemas    : Dict[str, Any]           (REQUIRED)
    - context    : Dict[str, Any]           (OPTIONAL)
    - retrieved  : List[Dict[str, Any]]     (OPTIONAL, RAG)

    OUTPUT
    ------
    {
        "sql": str,            # Generated SQL (empty on failure)
        "raw": Any,            # Raw provider output
        "metadata": Dict       # Provider + diagnostics
    }

    FAILURE BEHAVIOR
    ----------------
    - NEVER raise user-facing exceptions
    - Provider failure → deterministic fallback SQL
    - Missing inputs → structured error metadata
    """

    def __init__(
        self,
        tools: Optional[Tools] = None,
        provider_client: Optional[Any] = None,
    ):
        try:
            self.tools = tools or Tools()

            if provider_client is not None:
                self.provider = provider_client
            elif hasattr(self.tools, "get_provider_client"):
                self.provider = self.tools.get_provider_client()
            else:
                self.provider = None

            LOG.info(
                "GenerateNode initialized | provider=%s",
                _provider_name(self.provider) if self.provider else None,
            )

        except Exception as e:
            logger.exception("GenerateNode initialization failed")
            raise CustomException(e, sys)

    # ============================================================
    # Prompt construction
    # ============================================================
    def _build_prompt(
        self,
        *,
        user_query: str,
        schemas: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        retrieved: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build a deterministic, safety-first prompt.

        Prompt guarantees:
        - SELECT-only SQL
        - No schema hallucination
        - Correct numeric typing (DECIMAL vs INTEGER)
        - No unsafe CAST operations
        """

        parts: List[str] = [
            "You are a production-grade Text-to-SQL system.",
            "",
            "NON-NEGOTIABLE RULES:",
            "- Generate EXACTLY ONE SQL SELECT query",
            "- NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE",
            "- Use ONLY tables and columns explicitly listed in the schema",
            "- DO NOT invent tables or columns",
            "",
            "CRITICAL DATA SAFETY RULES:",
            "- Numeric-looking columns may contain commas, '+', spaces, or text",
            "",
            "COLUMN TYPE RULES (MANDATORY):",
            "- Ratings / scores / averages → DECIMAL values → CAST to DOUBLE",
            "- Counts / installs / totals → INTEGER values → CAST to BIGINT",
            "",
            "NUMERIC NORMALIZATION:",
            "- For INTEGER-like columns:",
            "    cleaned := regexp_replace(column, '[^0-9]', '', 'g')",
            "    CAST(NULLIF(cleaned, '') AS BIGINT)",
            "",
            "- For DECIMAL-like columns (e.g. Rating):",
            "    cleaned := regexp_replace(column, '[^0-9.]', '', 'g')",
            "    CAST(NULLIF(cleaned, '') AS DOUBLE)",
            "",
            "AGGREGATION / ORDERING:",
            "- Always aggregate or order using the CAST expression",
            "- NEVER CAST raw column values directly",
            "",
            "INCORRECT (DO NOT USE):",
            "- CAST(column AS BIGINT)",
            "- regexp_replace(column, '[^0-9]', '', 'g') for ratings",
            "",
            "User Question:",
            user_query,
            "",
        ]

        # ---------------- Context injection
        if context:
            last_sql = context.get("last_successful_sql")
            history = context.get("conversation_history", [])

            if last_sql:
                parts.extend(
                    [
                        "Previous successful SQL (reference only):",
                        last_sql,
                        "",
                    ]
                )

            if history:
                parts.append("Recent Conversation History:")
                for h in history[-3:]:
                    q = h.get("user_query")
                    s = h.get("sql")
                    if q and s:
                        parts.append(f"- Q: {q}")
                        parts.append(f"  SQL: {s}")
                parts.append("")

        # ---------------- Schema injection (authoritative)
        parts.extend(
            [
                "Available Database Schemas (authoritative):",
                json.dumps(schemas, indent=2),
                "",
            ]
        )

        # ---------------- Retrieved context (RAG)
        if retrieved:
            parts.append("Relevant Retrieved Context:")
            for r in retrieved:
                txt = r.get("text")
                if txt:
                    parts.append(f"- {txt}")
            parts.append("")

        parts.append("Return ONLY the SQL query. No explanation. No markdown.")

        return "\n".join(parts)

    # ============================================================
    # Graph entrypoint
    # ============================================================
    def run(
        self,
        *,
        user_query: Optional[str] = None,
        schemas: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        retrieved: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Graph-safe execution entrypoint.

        Guarantees:
        - Side-effect free
        - Always returns a structured dict
        - Never crashes the graph
        """

        try:
            # ---------------- Input validation
            if not user_query or not schemas:
                LOG.warning(
                    "GenerateNode.run missing input | user_query=%s | schemas=%s",
                    bool(user_query),
                    bool(schemas),
                )
                return {
                    "sql": "",
                    "raw": None,
                    "metadata": {"error": "missing_input"},
                }

            # ---------------- No provider → deterministic fallback
            if not self.provider:
                table = next(iter(schemas.keys()))
                fallback_sql = f"SELECT * FROM {table} LIMIT 10"

                LOG.warning(
                    "No LLM provider configured | fallback_sql=%s",
                    fallback_sql,
                )

                return {
                    "sql": fallback_sql,
                    "raw": None,
                    "metadata": {
                        "warning": "no_provider_fallback",
                        "table": table,
                    },
                }

            # ---------------- Build prompt
            prompt = self._build_prompt(
                user_query=user_query,
                schemas=schemas,
                context=context,
                retrieved=retrieved,
            )

            LOG.debug(
                "GenerateNode prompt built | chars=%d | has_context=%s",
                len(prompt),
                bool(context),
            )

            # ---------------- Provider invocation
            try:
                if callable(self.provider):
                    raw = self.provider(prompt)
                elif hasattr(self.provider, "generate"):
                    raw = self.provider.generate(prompt)
                elif hasattr(self.provider, "run"):
                    raw = self.provider.run(prompt)
                else:
                    raise RuntimeError("Unsupported provider interface")

            except Exception as e:
                LOG.exception("LLM provider failed; using fallback SQL")
                table = next(iter(schemas.keys()))
                return {
                    "sql": f"SELECT * FROM {table} LIMIT 10",
                    "raw": None,
                    "metadata": {
                        "provider": _provider_name(self.provider),
                        "provider_error": str(e),
                        "fallback": True,
                    },
                }

            # ---------------- Normalize provider output
            if isinstance(raw, str):
                text = raw
            elif isinstance(raw, dict):
                text = raw.get("text") or raw.get("output") or ""
            else:
                text = str(raw)

            # Strip markdown fences
            text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

            LOG.info(
                "SQL generated | length=%d | provider=%s",
                len(text),
                _provider_name(self.provider),
            )

            return {
                "sql": text,
                "raw": raw,
                "metadata": {
                    "provider": _provider_name(self.provider),
                },
            }

        except Exception as e:
            LOG.exception("GenerateNode.run failed")
            raise CustomException(e, sys)
