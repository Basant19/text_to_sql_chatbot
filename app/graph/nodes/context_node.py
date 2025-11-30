# app/graph/nodes/context_node.py
import sys
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools
from app.schema_store import _canonical_name  # use canonical helper for display

logger = get_logger("context_node")


class ContextNode:
    """
    Node responsible for collecting schema information for a list of CSV names/paths.

    Behaviour:
      - Accepts csv_names which may be file paths, sanitized filenames, or canonical names.
      - Resolves each input to the canonical schema key (filename without extension, lowercased)
        using the injected Tools (or falling back to accessing Tools._schema_store).
      - Returns mapping: canonical_table_name -> {"columns": [...], "sample_rows": [...]}
        where sample_rows is limited to a small number (1-3) for prompt brevity.
      - If schema not found, returns empty lists and logs a warning.
    """

    def __init__(self, tools: Optional[Tools] = None, sample_limit: int = 3):
        try:
            self._tools = tools or Tools()
            # try to ensure we can reach a SchemaStore by a few fallback access patterns
            if not (
                hasattr(self._tools, "get_schema")
                or getattr(self._tools, "_schema_store", None) is not None
            ):
                # Tools may still create schema store lazily; leave to runtime checks
                logger.warning("Tools provided to ContextNode does not expose schema helpers directly.")
            self._sample_limit = sample_limit
        except Exception as e:
            logger.exception("Failed to initialize ContextNode (Tools/SchemaStore creation)")
            raise CustomException(e, sys)

    def _get_schema_via_tools(self, name: str) -> Optional[List[str]]:
        """Try a few ways to get schema columns from Tools/SchemaStore."""
        # 1) Tools wrapper method
        try:
            if hasattr(self._tools, "get_schema"):
                cols = self._tools.get_schema(name)
                if cols is not None:
                    return cols
        except Exception:
            logger.debug("Tools.get_schema failed for %s", name, exc_info=True)

        # 2) Direct access to underlying SchemaStore if present
        try:
            ss = getattr(self._tools, "_schema_store", None)
            if ss is not None:
                # SchemaStore exposes get_schema and tolerant matching
                cols = ss.get_schema(name)
                if cols is not None:
                    return cols
                # try to find by canonical key if get_schema returns None
                # attempt _find_matching_key if available
                if hasattr(ss, "_find_matching_key"):
                    match = ss._find_matching_key(name)
                    if match:
                        return ss.get_schema(match)
        except Exception:
            logger.debug("Direct schema_store access failed for %s", name, exc_info=True)

        return None

    def _get_sample_rows_via_tools(self, name: str) -> Optional[List[Dict[str, Any]]]:
        """Try a few ways to get sample rows from Tools/SchemaStore."""
        try:
            if hasattr(self._tools, "get_sample_rows"):
                rows = self._tools.get_sample_rows(name)
                if rows is not None:
                    return rows[: self._sample_limit]
        except Exception:
            logger.debug("Tools.get_sample_rows failed for %s", name, exc_info=True)

        try:
            ss = getattr(self._tools, "_schema_store", None)
            if ss is not None:
                rows = ss.get_sample_rows(name)
                if rows is not None:
                    return rows[: self._sample_limit]
                if hasattr(ss, "_find_matching_key"):
                    match = ss._find_matching_key(name)
                    if match:
                        rows = ss.get_sample_rows(match)
                        if rows is not None:
                            return rows[: self._sample_limit]
        except Exception:
            logger.debug("Direct schema_store sample access failed for %s", name, exc_info=True)

        return None

    def run(self, csv_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Collect schemas for csv_names. For missing schemas, include an entry with
        empty columns/sample_rows so downstream nodes know the CSV was requested.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Mapping from canonical_table_name -> schema info:
            {
                "columns": List[str],
                "sample_rows": List[Dict[str, Any]]
            }
        """
        try:
            out: Dict[str, Dict[str, Any]] = {}

            # Normalize inputs: csv_names may be paths, filenames, or canonical names
            for input_name in csv_names or []:
                try:
                    # derive a candidate canonical name from the input (basename, no ext, lower)
                    canonical_candidate = _canonical_name(input_name)

                    # try to find actual schema columns using Tools (tolerant matching inside)
                    cols = self._get_schema_via_tools(input_name)
                    samples = self._get_sample_rows_via_tools(input_name)

                    # If that failed, try with canonical_candidate explicitly
                    if cols is None:
                        cols = self._get_schema_via_tools(canonical_candidate)
                        samples = samples or self._get_sample_rows_via_tools(canonical_candidate)

                    # determine the key we will expose to the prompt: prefer canonical_candidate
                    table_key = canonical_candidate

                    # final safety: if Tools/SchemaStore can return the actual matching key (private)
                    # try to prefer that key so the prompt shows the exact key stored
                    try:
                        ss = getattr(self._tools, "_schema_store", None)
                        if ss is not None and hasattr(ss, "_find_matching_key"):
                            matched_key = ss._find_matching_key(input_name)
                            if matched_key:
                                # use matched_key's canonical form as table_key
                                table_key = _canonical_name(matched_key)
                                # prefer explicit columns/sample from matched_key if available
                                cols = cols or ss.get_schema(matched_key)
                                samples = samples or ss.get_sample_rows(matched_key)
                    except Exception:
                        # ignore any private-access failures
                        pass

                    out[table_key] = {
                        "columns": cols or [],
                        "sample_rows": (samples or [])[: self._sample_limit],
                    }

                    if not cols:
                        logger.warning("Schema not found for '%s' (mapped->%s); returning empty schema info", input_name, table_key)

                except Exception:
                    logger.exception("Failed to collect schema for %s; continuing", input_name)
                    out[_canonical_name(input_name)] = {"columns": [], "sample_rows": []}

            logger.debug("ContextNode.run completed for CSVs: %s", csv_names)
            return out

        except Exception as e:
            logger.exception("ContextNode.run failed unexpectedly")
            raise CustomException(e, sys)
