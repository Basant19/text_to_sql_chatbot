# app/graph/nodes/context_node.py

import sys
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

logger = get_logger("context_node")


class ContextNode:
    """
    Node responsible for collecting schema information for a list of CSV names.

    Returns a mapping:
        csv_name -> {"columns": [...], "sample_rows": [...]}

    Can accept a Tools instance for dependency injection (SchemaStore, DB, etc.).
    """

    def __init__(self, tools: Optional[Tools] = None):
        try:
            # If Tools not provided, create default with SchemaStore
            self._tools = tools or Tools()
            if not hasattr(self._tools, "_schema_store") or self._tools._schema_store is None:
                raise CustomException("Tools instance must have a SchemaStore", sys)
        except Exception as e:
            logger.exception("Failed to initialize ContextNode (Tools/SchemaStore creation)")
            raise CustomException(e, sys)

    def run(self, csv_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Collect schemas for csv_names. For missing schemas, include an entry with
        empty columns/sample_rows so downstream nodes know the CSV was requested.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Mapping from CSV/table name to schema info:
            {
                "columns": List[str],
                "sample_rows": List[Dict[str, Any]]
            }
        """
        try:
            out: Dict[str, Dict[str, Any]] = {}
            for name in csv_names:
                try:
                    cols = self._tools.get_schema(name)
                    samples = self._tools.get_sample_rows(name)
                except Exception:
                    # If SchemaStore threw while fetching, convert to warning and continue
                    logger.exception(f"SchemaStore access failed for '{name}'; returning empty schema info")
                    cols = None
                    samples = None

                out[name] = {
                    "columns": cols or [],
                    "sample_rows": samples or []
                }

                if not cols:
                    logger.warning(f"Schema not found for '{name}'; returning empty schema info")

            logger.debug(f"ContextNode.run completed for CSVs: {csv_names}")
            return out

        except Exception as e:
            logger.exception("ContextNode.run failed unexpectedly")
            raise CustomException(e, sys)
