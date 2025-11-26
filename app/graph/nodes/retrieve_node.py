# app/graph/nodes/retrieve_node.py

import sys
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

logger = get_logger("retrieve_node")


class RetrieveNode:
    """
    Node that performs RAG retrieval using VectorSearch.

    Accepts an optional Tools instance for dependency injection:
        - vector search (Google embeddings + FAISS)
    """

    def __init__(self, tools: Optional[Tools] = None):
        try:
            self._tools = tools or Tools()
            if not hasattr(self._tools, "_vector_search") or self._tools._vector_search is None:
                logger.warning("RetrieveNode: Tools instance has no VectorSearch; retrieval may fail.")
        except Exception as e:
            logger.exception("Failed to initialize RetrieveNode")
            raise CustomException(e, sys)

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
            if not hasattr(self._tools, "search_vectors"):
                raise CustomException("Tools instance does not implement search_vectors", sys)

            results = self._tools.search_vectors(query, top_k=top_k) or []
            if not results:
                logger.debug("RetrieveNode.run: No results found for query.")
                return []

            # No filtering requested
            if not csv_names:
                return results

            # Filter results based on CSV/table name or metadata source/path
            csv_names_lower = [name.lower() for name in csv_names if name]
            filtered = []

            for doc in results:
                meta = doc.get("meta", {}) or {}
                meta_identifiers = [
                    str(meta.get("path", "") or ""),
                    str(meta.get("source", "") or ""),
                    str(meta.get("table_name", "") or "")
                ]
                meta_identifiers_lower = [m.lower() for m in meta_identifiers]

                if any(csv_name in m for csv_name in csv_names_lower for m in meta_identifiers_lower):
                    filtered.append(doc)

            # Fallback to original results if filtering yields nothing
            if not filtered:
                logger.debug("RetrieveNode.run: Filtering yielded no matches; returning all results.")
                return results

            return filtered

        except CustomException:
            raise
        except Exception as e:
            logger.exception("RetrieveNode.run failed")
            raise CustomException(e, sys)
