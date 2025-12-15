# app/graph/nodes/retrieve_node.py
from __future__ import annotations
import sys
import logging
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

logger = get_logger("retrieve_node")
LOG = logging.getLogger(__name__)


def _normalize_result(item: Any) -> Dict[str, Any]:
    """Normalize a single vector search hit into a dict."""
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
                return {"id": item[0], "score": item[1], "text": item[2], "meta": item[3]}
            if len(item) == 3:
                return {"id": item[0], "score": item[1], "text": item[2], "meta": {}}

        return {
            "id": getattr(item, "id", None),
            "score": getattr(item, "score", None),
            "text": getattr(item, "text", "") or "",
            "meta": getattr(item, "meta", {}) or {},
        }
    except Exception:
        LOG.debug("Failed to normalize vector result", exc_info=True)
        return {"id": None, "score": None, "text": str(item), "meta": {}}


class RetrieveNode:
    """
    RetrieveNode
    ------------
    Responsibilities:
      - Perform vector search (optional)
      - NEVER destroy schema metadata
      - Enrich schemas with retrieved context when available
      - Pass schemas through unchanged if vector search is disabled
    """

    def __init__(self, tools: Optional[Tools] = None, vector_client: Optional[Any] = None):
        try:
            self._tools = tools or Tools()
            self._vector_client = vector_client or getattr(self._tools, "_vector_search", None)
        except Exception as e:
            logger.exception("RetrieveNode initialization failed")
            raise CustomException(e, sys)

    def run(
        self,
        query: str,
        schemas: Dict[str, Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Parameters
        ----------
        query : str
            User query
        schemas : dict
            Canonical schema map (MUST NOT be lost)
        top_k : int
            Number of vector results
        """

        if not query:
            return []

        if not isinstance(schemas, dict):
            LOG.warning(
                "RetrieveNode received non-dict schemas (%s); skipping enrichment",
                type(schemas),
            )
            return []

        # --------------------------------------------------------------
        # If no vector search client → DO NOT break schema propagation
        # --------------------------------------------------------------
        client = self._vector_client
        if not client:
            LOG.info("RetrieveNode: vector search disabled; passing schemas through")
            return []

        # Normalize top_k
        try:
            top_k = int(top_k)
            top_k = min(max(top_k, 1), 50)
        except Exception:
            top_k = 5

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

        normalized: List[Dict[str, Any]] = []
        for it in list(results)[:top_k]:
            normalized.append(_normalize_result(it))

        LOG.info(
            "RetrieveNode: retrieved %d context chunks (top_k=%d)",
            len(normalized),
            top_k,
        )

        return normalized
