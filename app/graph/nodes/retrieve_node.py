# File: app/graph/nodes/retrieve_node.py
from __future__ import annotations

"""
RetrieveNode
============

Optional retrieval / grounding node for the Text-to-SQL pipeline.

This node enriches SQL generation with:
- schema-aware context
- documentation chunks
- example SQL snippets
- column/table descriptions

CRITICAL GUARANTEES
------------------
- ❌ NEVER mutates schemas
- ❌ NEVER blocks graph execution
- ❌ NEVER raises on retrieval failure
- ❌ NEVER stores state

If retrieval is unavailable or fails, the pipeline continues safely.
"""

import sys
import logging
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

logger = get_logger("retrieve_node")
LOG = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _normalize_result(item: Any) -> Dict[str, Any]:
    """
    Normalize a single vector-search hit into a canonical dict.

    Supported input shapes:
    - dict-based vector DB results
    - tuple/list results
    - lightweight objects with attributes

    Output schema (guaranteed):
    {
        "id": Any | None,
        "score": float | None,
        "text": str,
        "meta": dict
    }
    """
    try:
        # ------------------------------
        # Dict-like result
        # ------------------------------
        if isinstance(item, dict):
            return {
                "id": item.get("id") or item.get("key") or item.get("_id"),
                "score": item.get("score"),
                "text": item.get("text") or item.get("content") or "",
                "meta": item.get("meta") or item.get("metadata") or {},
            }

        # ------------------------------
        # Tuple / list result
        # ------------------------------
        if isinstance(item, (list, tuple)):
            if len(item) >= 4:
                return {
                    "id": item[0],
                    "score": item[1],
                    "text": item[2],
                    "meta": item[3],
                }
            if len(item) == 3:
                return {
                    "id": item[0],
                    "score": item[1],
                    "text": item[2],
                    "meta": {},
                }

        # ------------------------------
        # Object-like result
        # ------------------------------
        return {
            "id": getattr(item, "id", None),
            "score": getattr(item, "score", None),
            "text": getattr(item, "text", "") or "",
            "meta": getattr(item, "meta", {}) or {},
        }

    except Exception:
        LOG.debug("Failed to normalize vector result", exc_info=True)
        return {
            "id": None,
            "score": None,
            "text": str(item),
            "meta": {},
        }


# ------------------------------------------------------------------
# Retrieve Node
# ------------------------------------------------------------------
class RetrieveNode:
    """
    RetrieveNode
    ------------

    Purpose
    -------
    Perform *optional* retrieval to ground SQL generation.

    Responsibilities
    ----------------
    - Query vector / semantic search backends
    - Normalize heterogeneous retrieval outputs
    - Provide ranked context chunks to GenerateNode

    Non-Responsibilities
    --------------------
    - ❌ Schema mutation
    - ❌ Query rewriting
    - ❌ Validation logic
    - ❌ Error propagation
    """

    def __init__(
        self,
        tools: Optional[Tools] = None,
        vector_client: Optional[Any] = None,
    ) -> None:
        """
        Initialize RetrieveNode.

        Parameters
        ----------
        tools : Optional[Tools]
            Shared tools registry.
        vector_client : Optional[Any]
            Explicit vector search client (overrides tools).

        Notes
        -----
        Retrieval is *best-effort*. If no client exists,
        the node becomes a safe no-op.
        """
        try:
            self._tools = tools or Tools()

            # Prefer explicit client, fallback to Tools-provided one
            self._vector_client = (
                vector_client
                or getattr(self._tools, "_vector_search", None)
            )

            logger.info(
                "RetrieveNode initialized | vector_enabled=%s",
                bool(self._vector_client),
            )

        except Exception as e:
            logger.exception("RetrieveNode initialization failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        query: Optional[str] = None,
        schemas: Optional[Dict[str, Dict[str, Any]]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Execute retrieval.

        Keyword-only to guarantee compatibility with GraphBuilder.

        Parameters
        ----------
        query : Optional[str]
            User natural-language query.
        schemas : Optional[Dict[str, Dict[str, Any]]]
            Canonical schema map (read-only).
        top_k : int, default=5
            Maximum number of retrieved chunks.

        Returns
        -------
        List[Dict[str, Any]]
            Normalized retrieval results.

        Guarantees
        ----------
        - Always returns a list
        - Never raises on failure
        - Empty list means "no enrichment"
        """

        # ------------------------------
        # Defensive validation
        # ------------------------------
        if not query:
            LOG.warning("RetrieveNode.run called without query; skipping retrieval")
            return []

        if not schemas or not isinstance(schemas, dict):
            LOG.warning(
                "RetrieveNode received invalid schemas (%s); skipping retrieval",
                type(schemas),
            )
            return []

        # ------------------------------
        # Vector search disabled
        # ------------------------------
        client = self._vector_client
        if not client:
            LOG.info("RetrieveNode: vector search disabled; returning empty context")
            return []

        # ------------------------------
        # Sanitize top_k
        # ------------------------------
        try:
            top_k = int(top_k)
            top_k = min(max(top_k, 1), 50)
        except Exception:
            top_k = 5

        # ------------------------------
        # Execute vector search
        # ------------------------------
        try:
            if hasattr(client, "search"):
                raw_results = client.search(query, top_k=top_k)
            elif hasattr(client, "search_vectors"):
                raw_results = client.search_vectors(query, top_k=top_k)
            elif hasattr(self._tools, "search_vectors"):
                raw_results = self._tools.search_vectors(query, top_k=top_k)
            else:
                LOG.warning("RetrieveNode: no compatible vector search API found")
                return []

        except Exception as e:
            LOG.exception("RetrieveNode vector search failed: %s", e)
            return []

        # ------------------------------
        # Normalize results
        # ------------------------------
        results = list(raw_results)[:top_k]
        normalized = [_normalize_result(item) for item in results]

        LOG.info(
            "RetrieveNode retrieved %d chunks | top_k=%d",
            len(normalized),
            top_k,
        )

        return normalized
