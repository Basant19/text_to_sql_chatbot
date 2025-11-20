
import sys
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import schema_store as schema_store_module

logger = get_logger("context_node")


class ContextNode:
    """
    Node responsible for collecting schema information for a list of CSV names.
    Returns a mapping: csv_name -> {"columns": [...], "sample_rows": [...]}
    """

    def __init__(self, store: Optional[schema_store_module.SchemaStore] = None):
        try:
            # allow injection of a SchemaStore for testing
            self._store = store or schema_store_module.SchemaStore()
        except Exception as e:
            logger.exception("Failed to initialize ContextNode (SchemaStore creation)")
            raise CustomException(e, sys)

    def run(self, csv_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Collect schemas for csv_names. For missing schemas include an entry with
        empty columns/sample_rows (but still present so downstream nodes know it was requested).
        """
        try:
            out: Dict[str, Dict[str, Any]] = {}
            for name in csv_names:
                try:
                    cols = self._store.get_schema(name)
                    samples = self._store.get_sample_rows(name)
                except Exception:
                    # If SchemaStore threw while fetching, convert to warning and continue
                    logger.exception(f"SchemaStore access failed for {name}; using empty schema")
                    cols = None
                    samples = None

                if cols is None:
                    logger.warning(f"Schema not found for {name}; returning empty schema info")
                    out[name] = {"columns": [], "sample_rows": []}
                else:
                    out[name] = {"columns": cols, "sample_rows": samples or []}
            return out
        except Exception as e:
            logger.exception("ContextNode.run failed")
            raise CustomException(e, sys)
