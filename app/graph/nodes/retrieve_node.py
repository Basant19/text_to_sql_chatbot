# app/graph/nodes/retrieve_node.py
import sys
from typing import List, Dict, Any, Optional, Callable

from app.logger import get_logger
from app.exception import CustomException
from app import vector_search as vector_search_module

logger = get_logger("retrieve_node")


class RetrieveNode:
    """
    Node that performs RAG retrieval using VectorSearch.

    Parameters
    ----------
    vs_instance : Optional[vector_search_module.VectorSearch]
        If provided, the node will use this instance (useful for tests). Otherwise
        a default VectorSearch() will be created on first run.
    """

    def __init__(self, vs_instance: Optional[vector_search_module.VectorSearch] = None):
        try:
            self._vs = vs_instance
            self._vs_factory: Callable[[], vector_search_module.VectorSearch] = (
                lambda: vector_search_module.VectorSearch()
            )
        except Exception as e:
            logger.exception("Failed to initialize RetrieveNode")
            raise CustomException(e, sys)

    def _get_vs(self) -> vector_search_module.VectorSearch:
        if self._vs is None:
            try:
                self._vs = self._vs_factory()
            except Exception as e:
                logger.exception("Failed to create VectorSearch instance")
                raise CustomException(e, sys)
        return self._vs

    def run(self, query: str, csv_names: Optional[List[str]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Run retrieval.

        - query: user query to embed and search.
        - csv_names: optional list of csv/table names to filter results to; if omitted, no filtering.
        - top_k: number of top documents to request from the index.

        Returns a list of documents in the form returned by VectorSearch.search:
          [{"id": ..., "score": ..., "text": ..., "meta": {...}}, ...]
        """
        try:
            vs = self._get_vs()
            results = vs.search(query, top_k=top_k) or []
            if not results:
                return []

            if not csv_names:
                return results

            # filter by meta.path, meta.table_name, or meta.source heuristics
            filtered = []
            for d in results:
                meta = d.get("meta", {}) or {}
                path = str(meta.get("path", "") or meta.get("source", "") or meta.get("table_name", ""))
                # normalize to lower-case for matching
                low_path = path.lower()
                matched = False
                for name in csv_names:
                    if not name:
                        continue
                    if name.lower() in low_path or name.lower() == str(meta.get("table_name", "")).lower():
                        matched = True
                        break
                if matched:
                    filtered.append(d)

            # If filtering produced no results, fallback to original results (useful when metadata is sparse)
            return filtered if filtered else results
        except CustomException:
            raise
        except Exception as e:
            logger.exception("RetrieveNode.run failed")
            raise CustomException(e, sys)
