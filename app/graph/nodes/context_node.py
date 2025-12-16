# File: app/graph/nodes/context_node.py
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
    Accepts file paths, filenames, canonical names. Lowercases and strips extension.
    """
    if not inp:
        return ""
    try:
        name = str(inp).strip()
        name = os.path.basename(name)
        name = os.path.splitext(name)[0]
        name = "_".join(name.split()).lower()
        return name
    except Exception:
        return str(inp).strip().lower()


class ContextNode:
    """
    Collect schema context for a list of CSV names.

    Returns mapping:
      canonical_table_key -> {
          "columns": [...],
          "sample_rows": [...],
          "path": <abs path or None>,
          "canonical": <key>
      }

    Behavior:
      - Uses Tools.get_schema / Tools.get_sample_rows when available.
      - Falls back to Tools._schema_store if present.
      - Normalizes inputs to attempt tolerant lookups.
      - Does not raise if a schema is missing; returns empty lists for that table.
    """

    def __init__(self, tools: Optional[Tools] = None, sample_limit: int = 3):
        try:
            self._tools = tools or Tools()
            self._sample_limit = int(sample_limit)
        except Exception as e:
            logger.exception("Failed to initialize ContextNode")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # SchemaStore helper
    # ------------------------------------------------------------------
    def _schema_store_keys(self) -> List[str]:
        try:
            ss = getattr(self._tools, "_schema_store", None)
            if ss is None:
                return []
            for fn in ("list_keys", "list_tables", "get_all_tables", "keys"):
                if hasattr(ss, fn):
                    try:
                        method = getattr(ss, fn)
                        keys = method() if callable(method) else list(method)
                        if isinstance(keys, (list, tuple)):
                            return list(keys)
                    except Exception:
                        continue
            if hasattr(ss, "schemas"):
                try:
                    return list(ss.schemas.keys())
                except Exception:
                    pass
        except Exception:
            LOG.debug("schema_store_keys retrieval failed", exc_info=True)
        return []

    # ------------------------------------------------------------------
    # Fetch schema columns
    # ------------------------------------------------------------------
    def _try_get_schema(self, name: str) -> Optional[List[str]]:
        try:
            if hasattr(self._tools, "get_schema"):
                cols = self._tools.get_schema(name)
                if cols is not None:
                    return list(cols)
        except Exception:
            LOG.debug("Tools.get_schema failed for: %s", name, exc_info=True)

        try:
            ss = getattr(self._tools, "_schema_store", None)
            if ss is not None:
                if hasattr(ss, "get_table_canonical"):
                    can = ss.get_table_canonical(name)
                    if can:
                        cols = ss.get_schema(can)
                        if cols:
                            return list(cols)
                cols = ss.get_schema(name)
                if cols:
                    return list(cols)
        except Exception:
            LOG.debug("Direct schema_store access failed for %s", name, exc_info=True)

        return None

    # ------------------------------------------------------------------
    # Fetch sample rows
    # ------------------------------------------------------------------
    def _try_get_samples(self, name: str) -> Optional[List[Dict[str, Any]]]:
        try:
            if hasattr(self._tools, "get_sample_rows"):
                rows = self._tools.get_sample_rows(name)
                if rows is not None:
                    return list(rows)[: self._sample_limit]
        except Exception:
            LOG.debug("Tools.get_sample_rows failed for: %s", name, exc_info=True)

        try:
            ss = getattr(self._tools, "_schema_store", None)
            if ss is not None:
                if hasattr(ss, "get_table_canonical"):
                    can = ss.get_table_canonical(name)
                    if can:
                        rows = ss.get_sample_rows(can)
                        if rows:
                            return list(rows)[: self._sample_limit]
                rows = ss.get_sample_rows(name)
                if rows:
                    return list(rows)[: self._sample_limit]
        except Exception:
            LOG.debug("Direct schema_store sample access failed for %s", name, exc_info=True)

        return None

    # ------------------------------------------------------------------
    # Fetch CSV path
    # ------------------------------------------------------------------
    def _try_get_path_from_store(self, name: str) -> Optional[str]:
        try:
            ss = getattr(self._tools, "_schema_store", None)
            if ss is None:
                return None

            # Try common getter names
            for fn in ("get_path", "get_csv_path", "get_file", "get_file_path"):
                if hasattr(ss, fn):
                    try:
                        p = getattr(ss, fn)(name)
                        if p:
                            return os.path.abspath(p)
                    except Exception:
                        continue

            # Try generic metadata getters
            for fn in ("get_entry", "get", "get_metadata", "get_schema_entry"):
                if hasattr(ss, fn):
                    try:
                        entry = getattr(ss, fn)(name)
                        if isinstance(entry, dict):
                            for key in ("path", "csv_path", "file", "source"):
                                if entry.get(key):
                                    return os.path.abspath(entry.get(key))
                    except Exception:
                        continue

            # Try dict-like access
            if hasattr(ss, "schemas"):
                try:
                    s = getattr(ss, "schemas")
                    if isinstance(s, dict) and name in s:
                        entry = s.get(name)
                        if isinstance(entry, dict):
                            for key in ("path", "csv_path", "file", "source"):
                                if entry.get(key):
                                    return os.path.abspath(entry.get(key))
                except Exception:
                    pass

            # Try canonical mapping
            try:
                if hasattr(ss, "get_table_canonical"):
                    can = ss.get_table_canonical(name)
                    if can and can != name:
                        return self._try_get_path_from_store(can)
            except Exception:
                pass

        except Exception:
            LOG.debug("_try_get_path_from_store failed for %s", name, exc_info=True)
        return None

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
    def run(self, csv_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Build schema context mapping for multiple CSVs / tables.
        Returns:
            canonical_table_key -> {"columns": [...], "sample_rows": [...], "path": <abs path>, "canonical": <key>}
        """
        try:
            out: Dict[str, Dict[str, Any]] = {}
            if not csv_names:
                return out

            normalized_inputs: List[str] = []
            for raw in csv_names:
                try:
                    if isinstance(raw, dict):
                        candidate = raw.get("canonical") or raw.get("name") or raw.get("path") or ""
                    else:
                        candidate = raw
                    normalized_inputs.append(str(candidate))
                except Exception:
                    normalized_inputs.append(str(raw))

            known_keys = set(self._schema_store_keys())

            for raw in normalized_inputs:
                try:
                    if not raw:
                        continue
                    canonical_candidate = _canonical_name_from_input(raw)

                    # Direct lookups
                    cols = self._try_get_schema(raw)
                    samples = self._try_get_samples(raw)

                    # Try canonical version
                    if cols is None and canonical_candidate and canonical_candidate != raw:
                        cols = self._try_get_schema(canonical_candidate)
                        samples = samples or self._try_get_samples(canonical_candidate)

                    # Fuzzy/partial match against known keys
                    if cols is None and known_keys:
                        lower_raw = canonical_candidate.lower()
                        for k in known_keys:
                            if k.lower() == lower_raw:
                                cols = self._try_get_schema(k)
                                samples = samples or self._try_get_samples(k)
                                if cols:
                                    canonical_candidate = k
                                    break
                        if cols is None:
                            for k in known_keys:
                                if lower_raw in k.lower() or k.lower() in lower_raw:
                                    cols = self._try_get_schema(k)
                                    samples = samples or self._try_get_samples(k)
                                    if cols:
                                        canonical_candidate = k
                                        break

                    table_key = canonical_candidate or str(raw)

                    # Prefer store canonical key
                    try:
                        ss = getattr(self._tools, "_schema_store", None)
                        if ss and hasattr(ss, "get_table_canonical"):
                            store_key = ss.get_table_canonical(raw) or ss.get_table_canonical(canonical_candidate)
                            if store_key:
                                table_key = store_key
                                cols = cols or ss.get_schema(store_key) or []
                                samples = samples or ss.get_sample_rows(store_key) or []
                    except Exception:
                        LOG.debug("Could not prefer store canonical key for %s", raw, exc_info=True)

                    # CSV path
                    path = None
                    try:
                        ss = getattr(self._tools, "_schema_store", None)
                        if ss:
                            path = self._try_get_path_from_store(table_key)
                            if not path:
                                path = self._try_get_path_from_store(raw) or self._try_get_path_from_store(canonical_candidate)
                    except Exception:
                        path = None

                    out[table_key] = {
                        "columns": cols or [],
                        "sample_rows": (samples or [])[: self._sample_limit],
                        "path": os.path.abspath(path) if path else None,
                        "canonical": table_key,
                    }

                    if not cols:
                        LOG.warning(
                            "ContextNode: schema not found for '%s' (mapped->%s). Known keys: %s",
                            raw,
                            table_key,
                            list(sorted(known_keys))[:30],
                        )

                except Exception:
                    LOG.exception("ContextNode: failed for input %s", raw)
                    key = _canonical_name_from_input(raw) or str(raw)
                    out[key] = {"columns": [], "sample_rows": [], "path": None, "canonical": key}

            LOG.debug("ContextNode.run completed for: %s -> keys=%s", csv_names, list(out.keys()))
            return out

        except Exception as e:
            logger.exception("ContextNode.run failed unexpectedly")
            raise CustomException(e, sys)
