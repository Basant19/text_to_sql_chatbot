from __future__ import annotations

"""
ContextNode
===========

Purpose
-------
ContextNode is the **memory boundary** of the Text-to-SQL system.

It is the ONLY place where:
- Conversation history is loaded
- Conversational context is assembled
- Follow-up SQL continuity is handled

ARCHITECTURAL GUARANTEES
-----------------------
- UI NEVER passes history into graph.run()
- GraphBuilder remains stateless
- HistoryStore remains the source of truth
- ContextNode is read-only during run()

This node does NOT:
- generate SQL
- validate SQL
- execute SQL

This node DOES:
- Load bounded conversation history
- Expose prompt-safe context to downstream nodes
- Track last successful SQL for follow-up questions
"""

import sys
import logging
from typing import Dict, Any, Optional, List

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools
from app.history_sql import HistoryStore

logger = get_logger("context_node")
LOG = logging.getLogger(__name__)


class ContextNode:
    """
    ContextNode
    -----------

    Lifecycle:
        UI → HistoryStore → ContextNode → GenerateNode → ...

    Responsibilities:
    - Load persisted conversation history
    - Provide bounded conversational memory
    - Track last successful SQL (ephemeral, per-session)
    """

    def __init__(
        self,
        tools: Optional[Tools] = None,
        *,
        max_history: int = 5,
        history_store: Optional[HistoryStore] = None,
    ):
        try:
            self.tools = tools
            self.max_history = max_history

            # Stateful store (injected here, NEVER via UI)
            self.history_store = history_store or HistoryStore()

            # Ephemeral execution memory (NOT persisted)
            self._last_successful_sql: Optional[str] = None
            self._last_tables: Optional[List[str]] = None

            LOG.info("ContextNode initialized | max_history=%d", max_history)

        except Exception as e:
            logger.exception("ContextNode initialization failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Graph entrypoint (READ-ONLY)
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        conversation_id: Optional[str] = None,
        csv_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Graph entrypoint.

        Rules:
        - MUST be side-effect free
        - MUST NOT mutate internal state
        - MUST NOT assume UI involvement
        """
        try:
            history: List[Dict[str, Any]] = []

            if conversation_id:
                try:
                    conv = self.history_store.get_conversation(conversation_id)
                    messages = conv.get("messages", [])
                    history = messages[-self.max_history :]
                except Exception:
                    LOG.warning(
                        "Conversation not found | id=%s",
                        conversation_id,
                    )

            context: Dict[str, Any] = {
                # Structural context
                "available_tables": csv_names or [],

                # Conversational memory (prompt-safe)
                "conversation_history": history,

                # Follow-up continuity (ephemeral)
                "last_successful_sql": self._last_successful_sql,
                "last_tables": self._last_tables,
            }

            LOG.debug(
                "ContextNode.run | conv=%s | history=%d | has_last_sql=%s",
                conversation_id,
                len(history),
                bool(self._last_successful_sql),
            )

            return context

        except Exception as e:
            logger.exception("ContextNode.run failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Explicit mutation AFTER execution
    # ------------------------------------------------------------------
    def update(
        self,
        *,
        user_query: str,
        sql: Optional[str],
        valid: bool,
        tables_used: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Persist successful execution context.

        Called ONLY after:
            ValidateNode → ExecuteNode succeed

        This mutation is:
        - Explicit
        - Ephemeral
        - Non-persistent
        """
        try:
            if valid and sql:
                self._last_successful_sql = sql
                self._last_tables = tables_used or []

                LOG.info(
                    "Context updated | sql_len=%d | tables=%s",
                    len(sql),
                    self._last_tables,
                )
            else:
                LOG.info("Context update skipped (invalid execution)")

        except Exception:
            LOG.exception("ContextNode.update failed")

    # ------------------------------------------------------------------
    # Follow-up SQL resolution helper
    # ------------------------------------------------------------------
    def resolve_base_sql(self, proposed_sql: Optional[str]) -> Optional[str]:
        """
        Resolve SQL base for follow-up questions.

        Priority:
        1. Newly proposed SQL
        2. Last successful SQL
        3. None
        """
        if proposed_sql:
            return proposed_sql

        if self._last_successful_sql:
            LOG.info("Reusing last successful SQL for follow-up")
            return self._last_successful_sql

        LOG.warning("No base SQL available for follow-up")
        return None
