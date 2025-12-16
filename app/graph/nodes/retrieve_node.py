# File: app/graph/nodes/retrieve_node.py
from __future__ import annotations

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
    """Normalize a single vector search hit into a consistent dict."""
    try:
        if isinstance(item, dict):
            return {
                "id": item.get("id") or item.get("key") or item.get("_id"),
                "score": item.get("score"),
                "text": item.get("text") or item.get("content") or "",
                "meta": item.get("meta") or item.get("metadata") or {},
            }

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
    Responsibilities:
      - Perform optional vector search for schema/context grounding
      - NEVER mutate or drop schema metadata
      - Fully keyword-callable for GraphBuilder / LangGraph
    """

    def __init__(self, tools: Optional[Tools] = None, vector_client: Optional[Any] = None):
        try:
            self._tools = tools or Tools()
            # Prefer explicit vector client, otherwise use Tools-provided one
            self._vector_client = vector_client or getattr(self._tools, "_vector_search", None)
        except Exception as e:
            logger.exception("RetrieveNode initialization failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        query: Optional[str] = None,
        schemas: Optional[Dict[str, Dict[str, Any]]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Keyword-only execution method.

        Parameters
        ----------
        query : str
            User natural-language query
        schemas : dict
            Canonical schema map (must not be lost)
        top_k : int
            Number of vector search results

        Returns
        -------
        List[Dict[str, Any]]
            List of normalized context chunks
        """

        # ------------------------------
        # Defensive validation
        # ------------------------------
        if not query:
            LOG.warning("RetrieveNode called without query; skipping retrieval")
            return []

        if not schemas or not isinstance(schemas, dict):
            LOG.warning(
                "RetrieveNode received invalid schemas (%s); skipping enrichment",
                type(schemas),
            )
            return []

        # --------------------------------------------------------------
        # Vector search disabled → do NOT break the pipeline
        # --------------------------------------------------------------
        client = self._vector_client
        if not client:
            LOG.info("RetrieveNode: vector search disabled; no context retrieved")
            return []

        # --------------------------------------------------------------
        # Sanitize top_k
        # --------------------------------------------------------------
        try:
            top_k = int(top_k)
            top_k = min(max(top_k, 1), 50)
        except Exception:
            top_k = 5

        # --------------------------------------------------------------
        # Execute vector search
        # --------------------------------------------------------------
        results: List[Any] = []
        try:
            if hasattr(client, "search"):
                results = client.search(query, top_k=top_k)
            elif hasattr(client, "search_vectors"):
                results = client.search_vectors(query, top_k=top_k)
            elif hasattr(self._tools, "search_vectors"):
                results = self._tools.search_vectors(query, top_k=top_k)
            else:
                LOG.warning("RetrieveNode: no usable vector search API found")
                return []
        except Exception as e:
            LOG.exception("RetrieveNode vector search failed: %s", e)
            return []

        # --------------------------------------------------------------
        # Normalize results
        # --------------------------------------------------------------
        normalized: List[Dict[str, Any]] = [_normalize_result(it) for it in list(results)[:top_k]]

        LOG.info(
            "RetrieveNode: retrieved %d context chunks (top_k=%d)",
            len(normalized),
            top_k,
        )

        return normalized
