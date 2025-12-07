# app/graph/nodes/context_node.py
from __future__ import annotations
import sys
import os
import logging
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

logger = get_logger("context_node")
LOG = logging.getLogger(__name__)


def _canonical_name_from_input(inp: Optional[str]) -> str:
    """
    Derive a stable canonical-like token from an arbitrary input.
    Accepts file paths, filenames, canonical names. Lowercases and strips ext.
    """
    if not inp:
        return ""
    name = str(inp).strip()
    # if looks like a path, take basename
    name = os.path.basename(name)
    # remove extension
    name = os.path.splitext(name)[0]
    # simple normalization: lowercase and replace spaces with underscore
    name = name.strip().lower().replace(" ", "_")
    return name


class ContextNode:
    """
    Collect schema context for a list of CSV names.

    Returns mapping:
      canonical_table_key -> {"columns": [...], "sample_rows": [...]}

    Behavior:
      - Uses Tools.get_schema / Tools.get_sample_rows when available.
      - Falls back to Tools._schema_store if present.
      - Normalizes inputs to attempt tolerant lookups.
      - Does not raise if a schema is missing; returns empty lists for that table.
    """

    def __init__(self, tools: Optional[Tools] = None, sample_limit: int = 3):
        try:
            self._tools = tools or Tools()
            self._sample_limit = sample_limit
        except Exception as e:
            logger.exception("Failed to initialize ContextNode")
            raise CustomException(e, sys)

    def _try_get_schema(self, name: str) -> Optional[List[str]]:
        # 1) Tools wrapper
        try:
            if hasattr(self._tools, "get_schema"):
                cols = self._tools.get_schema(name)
                if cols is not None:
                    return cols
        except Exception:
            LOG.debug("Tools.get_schema failed for: %s", name, exc_info=True)

        # 2) direct SchemaStore
        try:
            ss = getattr(self._tools, "_schema_store", None)
            if ss is not None:
                # try canonical lookup from store first
                try:
                    can = ss.get_table_canonical(name)
                    if can:
                        cols = ss.get_schema(can)
                        if cols is not None:
                            return cols
                except Exception:
                    pass
                # try direct name
                try:
                    cols = ss.get_schema(name)
                    if cols is not None:
                        return cols
                except Exception:
                    pass
        except Exception:
            LOG.debug("Direct schema_store access failed for %s", name, exc_info=True)

        return None

    def _try_get_samples(self, name: str) -> Optional[List[Dict[str, Any]]]:
        try:
            if hasattr(self._tools, "get_sample_rows"):
                rows = self._tools.get_sample_rows(name)
                if rows is not None:
                    return rows[: self._sample_limit]
        except Exception:
            LOG.debug("Tools.get_sample_rows failed for: %s", name, exc_info=True)

        try:
            ss = getattr(self._tools, "_schema_store", None)
            if ss is not None:
                can = ss.get_table_canonical(name)
                if can:
                    rows = ss.get_sample_rows(can)
                    if rows is not None:
                        return rows[: self._sample_limit]
                rows = ss.get_sample_rows(name)
                if rows is not None:
                    return rows[: self._sample_limit]
        except Exception:
            LOG.debug("Direct schema_store sample access failed for %s", name, exc_info=True)

        return None

    def run(self, csv_names: List[str]) -> Dict[str, Dict[str, Any]]:
        try:
            out: Dict[str, Dict[str, Any]] = {}
            if not csv_names:
                return out

            for raw in csv_names:
                try:
                    canonical_candidate = _canonical_name_from_input(raw)
                    cols = self._try_get_schema(raw)
                    samples = self._try_get_samples(raw)

                    # try the canonical candidate as a fallback
                    if cols is None:
                        cols = self._try_get_schema(canonical_candidate)
                        samples = samples or self._try_get_samples(canonical_candidate)

                    # final key exposed to downstream: prefer canonical_candidate
                    table_key = canonical_candidate or str(raw)

                    # if SchemaStore knows a different canonical key, prefer that for clarity
                    try:
                        ss = getattr(self._tools, "_schema_store", None)
                        if ss is not None:
                            store_key = ss.get_table_canonical(raw) or ss.get_table_canonical(canonical_candidate)
                            if store_key:
                                table_key = store_key
                                cols = cols or ss.get_schema(store_key)
                                samples = samples or ss.get_sample_rows(store_key)
                    except Exception:
                        LOG.debug("Could not prefer store canonical key for %s", raw, exc_info=True)

                    out[table_key] = {
                        "columns": cols or [],
                        "sample_rows": (samples or [])[: self._sample_limit],
                    }

                    if not cols:
                        LOG.warning("ContextNode: schema not found for '%s' (mapped->%s)", raw, table_key)

                except Exception:
                    LOG.exception("ContextNode: failed for input %s", raw)
                    out[_canonical_name_from_input(raw)] = {"columns": [], "sample_rows": []}

            LOG.debug("ContextNode.run completed for: %s", csv_names)
            return out
        except Exception as e:
            logger.exception("ContextNode.run failed unexpectedly")
            raise CustomException(e, sys)
