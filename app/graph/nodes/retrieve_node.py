#D:\text_to_sql_bot\app\graph\nodes\retrieve_node.py
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
    """Normalize a single vector search hit into a dict with keys: id, score, text, meta."""
    try:
        # If it's already a dict-like object
        if isinstance(item, dict):
            return {
                "id": item.get("id") or item.get("key") or item.get("_id"),
                "score": item.get("score"),
                "text": item.get("text") or item.get("content") or item.get("output") or "",
                "meta": item.get("meta") or item.get("metadata") or {},
            }
        # If it's a tuple/list like (id, score, text, meta)
        if isinstance(item, (list, tuple)):
            # try to map common shapes
            if len(item) >= 4:
                return {"id": item[0], "score": item[1], "text": item[2], "meta": item[3]}
            if len(item) == 3:
                return {"id": item[0], "score": item[1], "text": item[2], "meta": {}}
            if len(item) == 2:
                return {"id": None, "score": item[0], "text": item[1], "meta": {}}
        # Generic object with attributes
        id_ = getattr(item, "id", None) or getattr(item, "key", None) or getattr(item, "_id", None)
        score = getattr(item, "score", None) or getattr(item, "distance", None)
        text = getattr(item, "text", None) or getattr(item, "content", None) or getattr(item, "output", None)
        meta = getattr(item, "meta", None) or getattr(item, "metadata", None) or getattr(item, "meta_info", None)
        return {"id": id_, "score": score, "text": text or "", "meta": meta or {}}
    except Exception:
        LOG.debug("_normalize_result failed for item: %s", str(item), exc_info=True)
        return {"id": None, "score": None, "text": str(item), "meta": {}}


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

            # cap top_k to reasonable limits
            try:
                top_k = int(top_k)
                if top_k <= 0:
                    top_k = 5
                if top_k > 100:
                    top_k = 100
            except Exception:
                top_k = 5

            results: List[Any] = []

            # Attempt several common search method signatures
            try:
                # Some clients: search(query, top_k=...)
                if hasattr(client, "search"):
                    try:
                        results = client.search(query, top_k=top_k)
                    except TypeError:
                        # maybe signature is search(query, k)
                        try:
                            results = client.search(query, k=top_k)
                        except Exception:
                            results = client.search(query, top_k)

                # Some: search_vectors(query, top_k=...)
                elif hasattr(client, "search_vectors"):
                    try:
                        results = client.search_vectors(query, top_k=top_k)
                    except TypeError:
                        results = client.search_vectors(query, top_k)

                # Tools wrapper fallback
                elif hasattr(self._tools, "search_vectors"):
                    results = self._tools.search_vectors(query, top_k=top_k)

                else:
                    # last-resort: try a generic 'query' method
                    if hasattr(client, "query"):
                        results = client.query(query, top_k=top_k)
                    else:
                        LOG.warning("RetrieveNode: vector client has no recognized search method: %s", type(client))
                        return []

            except Exception as e:
                LOG.exception("Vector search execution failed: %s", e)
                return []

            # Normalize results into expected dicts
            normalized: List[Dict[str, Any]] = []
            try:
                # Some clients return dict with 'matches' or 'results'
                if isinstance(results, dict):
                    for key in ("matches", "results", "hits", "items"):
                        if key in results and isinstance(results[key], (list, tuple)):
                            results_list = results[key]
                            for it in results_list[:top_k]:
                                normalized.append(_normalize_result(it))
                            break
                    else:
                        # if dict-like but not containing known keys, try to normalize directly
                        normalized.append(_normalize_result(results))
                elif isinstance(results, (list, tuple)):
                    for it in list(results)[:top_k]:
                        normalized.append(_normalize_result(it))
                else:
                    # single object
                    normalized.append(_normalize_result(results))
            except Exception:
                LOG.exception("Failed to normalize vector search results")

            LOG.info("RetrieveNode: returned %d results for query (top_k=%d)", len(normalized), top_k)
            return normalized

        except Exception as e:
            logger.exception("RetrieveNode.run failed")
            raise CustomException(e, sys)
