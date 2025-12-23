# File: app/graph/agent.py
from __future__ import annotations

"""
LangGraphAgent
==============

Thin execution wrapper over GraphBuilder.

This module provides a **stable, programmatic entrypoint**
to the Text-to-SQL pipeline for:

- APIs (FastAPI / Flask)
- CLI tools
- Batch jobs
- LangSmith tracing & observability

IMPORTANT ARCHITECTURAL GUARANTEES
----------------------------------
- ❌ NEVER passes conversation history
- ❌ NEVER reads HistoryStore
- ❌ NEVER mutates ContextNode behavior
- ❌ NEVER stores state between runs

This agent is a **pure executor**.
All intelligence and orchestration live inside GraphBuilder.
"""

import sys
import time
from typing import Any, Dict, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.graph.builder import GraphBuilder
from app.tools import Tools

logger = get_logger("graph_agent")


class LangGraphAgent:
    """
    LangGraphAgent
    --------------

    Stateless execution facade over GraphBuilder.

    Responsibilities
    ----------------
    - Tool injection (exactly once)
    - Graph lifecycle management
    - Execution timing & logging
    - Error surfacing (no swallowing)

    Non-Responsibilities
    --------------------
    - ❌ No schema normalization logic
    - ❌ No retrieval / generation logic
    - ❌ No SQL validation or execution logic
    - ❌ No persistence or caching

    Think of this class as a **clean boundary**
    between your application layer and the graph engine.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        tools: Optional[Tools] = None,
    ) -> None:
        """
        Initialize LangGraphAgent.

        Parameters
        ----------
        tools : Optional[Tools]
            Explicit Tools registry.

        Notes
        -----
        🔒 This is the ONLY place where tools are injected
        into the global GraphBuilder registry.

        If tools are not provided, a default Tools()
        instance is created.
        """
        try:
            # ----------------------------------------------
            # Resolve tools explicitly (no hidden globals)
            # ----------------------------------------------
            self.tools: Tools = tools or Tools()

            # ----------------------------------------------
            # Register tools globally for GraphBuilder
            # ----------------------------------------------
            GraphBuilder.set_global_tools(self.tools)

            logger.info(
                "LangGraphAgent initialized | tools=%s",
                self.tools.__class__.__name__,
            )

        except Exception as e:
            logger.exception("LangGraphAgent initialization failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Execution entrypoint
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        user_query: str,
        schemas: Dict[str, Dict[str, Any]],
        run_query: bool = True,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute the Text-to-SQL graph.

        STRICT CONTRACT
        ---------------
        - Mirrors GraphBuilder.run() EXACTLY
        - No additional parameters
        - No implicit defaults beyond GraphBuilder
        - No state carried between runs

        Parameters
        ----------
        user_query : str
            Natural-language query from the user.

        schemas : Dict[str, Dict[str, Any]]
            Canonical schema map:
            {
                "table_name": {
                    "columns": {...},
                    "types": {...},
                    ...
                }
            }

        run_query : bool, default=True
            Whether SQL execution is allowed.
            If False, generation + validation only.

        top_k : int, default=5
            Retriever depth.

        Returns
        -------
        Dict[str, Any]
            Standard GraphBuilder result object:
            {
                "sql": str | None,
                "rows": list,
                "columns": list,
                "rowcount": int,
                "valid": bool,
                "error": str | None,
                "timings": dict,
                ...
            }
        """
        start = time.time()

        try:
            logger.info(
                "LangGraphAgent.run started | run_query=%s | top_k=%s",
                run_query,
                top_k,
            )

            # ----------------------------------------------
            # Build a fresh, stateless graph
            # ----------------------------------------------
            graph = GraphBuilder()

            # ----------------------------------------------
            # Execute graph
            # ----------------------------------------------
            result = graph.run(
                user_query=user_query,
                schemas=schemas,
                run_query=run_query,
                top_k=top_k,
            )

            elapsed = time.time() - start

            logger.info(
                "LangGraphAgent.run completed | valid=%s | sql_len=%s | rows=%s | time=%.3fs",
                result.get("valid"),
                len(result.get("sql") or ""),
                result.get("rowcount"),
                elapsed,
            )

            return result

        except Exception as e:
            logger.exception("LangGraphAgent.run failed")
            raise CustomException(e, sys)
