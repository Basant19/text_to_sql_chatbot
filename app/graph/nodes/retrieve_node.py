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
        """Return the VectorSearch instance, creating it if necessary."""
        if self._vs is None:
            try:
                self._vs = self._vs_factory()
            except Exception as e:
                logger.exception("Failed to create VectorSearch instance")
                raise CustomException(e, sys)
        return self._vs

    def run(
        self,
        query: str,
        csv_names: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform retrieval for a query, optionally filtering by CSV/table names.

        Parameters
        ----------
        query : str
            User query to embed and search.
        csv_names : Optional[List[str]]
            Optional list of CSV or table names to filter results; case-insensitive.
        top_k : int
            Number of top documents to return.

        Returns
        -------
        List[Dict[str, Any]]
            List of documents in the form returned by VectorSearch.search:
            [{"id": ..., "score": ..., "text": ..., "meta": {...}}, ...]
        """
        try:
            vs = self._get_vs()
            results = vs.search(query, top_k=top_k) or []
            if not results:
                return []

            # No filtering requested
            if not csv_names:
                return results

            # Filter results based on CSV/table name or metadata source/path
            filtered = []
            csv_names_lower = [name.lower() for name in csv_names if name]

            for doc in results:
                meta = doc.get("meta") or {}
                meta_identifiers = [
                    str(meta.get("path", "") or ""),
                    str(meta.get("source", "") or ""),
                    str(meta.get("table_name", "") or "")
                ]
                meta_identifiers_lower = [m.lower() for m in meta_identifiers]

                if any(csv_name in m or csv_name == meta.get("table_name", "").lower() 
                       for csv_name in csv_names_lower for m in meta_identifiers_lower):
                    filtered.append(doc)

            # Fallback to original results if filtering yields nothing
            return filtered if filtered else results

        except CustomException:
            raise
        except Exception as e:
            logger.exception("RetrieveNode.run failed")
            raise CustomException(e, sys)
