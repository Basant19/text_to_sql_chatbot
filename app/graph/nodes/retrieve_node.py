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


class RetrieveNode:
    """
    RetrieveNode:
      - Performs a vector search for a query using the provided Tools.vector_search (or provided client).
      - Returns a list of top-k results with id, score, text, meta.
      - If vector backend missing, returns empty list (but does not crash).
    """

    def __init__(self, tools: Optional[Tools] = None, vector_client: Optional[Any] = None):
        try:
            self._tools = tools or Tools()
            # allow explicit client override for testing
            self._vector_client = vector_client or getattr(self._tools, "_vector_search", None)
        except Exception as e:
            logger.exception("Failed to initialize RetrieveNode")
            raise CustomException(e, sys)

    def run(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            if not query:
                return []

            client = self._vector_client or getattr(self._tools, "_vector_search", None)
            if not client:
                LOG.warning("RetrieveNode: no vector search client configured")
                return []

            # prefer standard API: search(query, top_k)
            try:
                if hasattr(client, "search"):
                    return client.search(query, top_k=top_k)
                # some clients use search_vectors or similar
                if hasattr(client, "search_vectors"):
                    return client.search_vectors(query, top_k=top_k)
                # fallback: if Tools has search_vectors
                if hasattr(self._tools, "search_vectors"):
                    return self._tools.search_vectors(query, top_k=top_k)
            except Exception:
                LOG.exception("Vector search failed for query: %s", query)

            return []
        except Exception as e:
            logger.exception("RetrieveNode.run failed")
            raise CustomException(e, sys)
