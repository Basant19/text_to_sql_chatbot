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
    Accepts file paths, filenames, canonical names. Lowercases and strips ext.
    """
    if not inp:
        return ""
    try:
        name = str(inp).strip()
        # if looks like a path, take basename
        name = os.path.basename(name)
        # remove extension
        name = os.path.splitext(name)[0]
        # simple normalization: lowercase and replace whitespace with underscore
        name = "_".join(name.split()).lower()
        return name
    except Exception:
        return str(inp).strip().lower()


class ContextNode:
    """
    Collect schema context for a list of CSV names.

    Returns mapping:
      canonical_table_key -> {"columns": [...], "sample_rows": [...], "path": <abs path or None>, "canonical": <key>}

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

    def _schema_store_keys(self) -> List[str]:
        """Return a list of known schema keys from the backing SchemaStore if present."""
        try:
            ss = getattr(self._tools, "_schema_store", None)
            if ss is None:
                return []
            # try common listing methods
            for fn in ("list_keys", "list_tables", "get_all_tables", "keys"):
                if hasattr(ss, fn):
                    try:
                        method = getattr(ss, fn)
                        keys = method() if callable(method) else list(method)
                        if isinstance(keys, (list, tuple)):
                            return list(keys)
                    except Exception:
                        continue
            # fallback: try iterating attributes
            if hasattr(ss, "schemas"):
                try:
                    return list(ss.schemas.keys())
                except Exception:
                    pass
        except Exception:
            LOG.debug("schema_store_keys retrieval failed", exc_info=True)
        return []

    def _try_get_schema(self, name: str) -> Optional[List[str]]:
        # 1) Tools wrapper
        try:
            if hasattr(self._tools, "get_schema"):
                cols = self._tools.get_schema(name)
                if cols is not None:
                    return list(cols)
        except Exception:
            LOG.debug("Tools.get_schema failed for: %s", name, exc_info=True)

        # 2) direct SchemaStore
        try:
            ss = getattr(self._tools, "_schema_store", None)
            if ss is not None:
                # try canonical lookup from store first
                try:
                    if hasattr(ss, "get_table_canonical"):
                        can = ss.get_table_canonical(name)
                        if can:
                            cols = ss.get_schema(can)
                            if cols is not None:
                                return list(cols)
                except Exception:
                    pass
                # try direct name
                try:
                    cols = ss.get_schema(name)
                    if cols is not None:
                        return list(cols)
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
                    return list(rows)[: self._sample_limit]
        except Exception:
            LOG.debug("Tools.get_sample_rows failed for: %s", name, exc_info=True)

        try:
            ss = getattr(self._tools, "_schema_store", None)
            if ss is not None:
                try:
                    if hasattr(ss, "get_table_canonical"):
                        can = ss.get_table_canonical(name)
                        if can:
                            rows = ss.get_sample_rows(can)
                            if rows is not None:
                                return list(rows)[: self._sample_limit]
                except Exception:
                    pass
                try:
                    rows = ss.get_sample_rows(name)
                    if rows is not None:
                        return list(rows)[: self._sample_limit]
                except Exception:
                    pass
        except Exception:
            LOG.debug("Direct schema_store sample access failed for %s", name, exc_info=True)

        return None

    def _try_get_path_from_store(self, name: str) -> Optional[str]:
        """
        Attempt to extract a CSV path for a table name from the SchemaStore via common methods.
        """
        try:
            ss = getattr(self._tools, "_schema_store", None)
            if ss is None:
                return None

            # 1) try common getter names
            for fn in ("get_path", "get_csv_path", "get_file", "get_file_path"):
                if hasattr(ss, fn):
                    try:
                        p = getattr(ss, fn)(name)
                        if p:
                            return os.path.abspath(p)
                    except Exception:
                        continue

            # 2) try generic get / get_entry / get_metadata
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

            # 3) try looking in ss.schemas dict-like structure
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

            # 4) try canonical mapping then re-run lookups
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

    def run(self, csv_names: List[str]) -> Dict[str, Dict[str, Any]]:
        try:
            out: Dict[str, Dict[str, Any]] = {}
            if not csv_names:
                return out

            # normalize requested names to strings (allow dicts with 'name' or 'canonical')
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

            # pre-cache known schema keys for helpful logging
            known_keys = set(self._schema_store_keys())

            for raw in normalized_inputs:
                try:
                    if not raw:
                        continue
                    canonical_candidate = _canonical_name_from_input(raw)

                    # Try multiple lookup strategies (original, canonical, known store keys)
                    cols = self._try_get_schema(raw)
                    samples = self._try_get_samples(raw)

                    if cols is None and canonical_candidate and canonical_candidate != raw:
                        cols = self._try_get_schema(canonical_candidate)
                        samples = samples or self._try_get_samples(canonical_candidate)

                    # attempt fuzzy/partial match against known store keys
                    if cols is None and known_keys:
                        # exact match ignoring extension and case
                        lower_raw = canonical_candidate.lower()
                        for k in known_keys:
                            if k.lower() == lower_raw:
                                cols = self._try_get_schema(k)
                                samples = samples or self._try_get_samples(k)
                                if cols:
                                    # prefer store's canonical key
                                    canonical_candidate = k
                                    break
                        # try substring match (last resort)
                        if cols is None:
                            for k in known_keys:
                                if lower_raw in k.lower() or k.lower() in lower_raw:
                                    cols = self._try_get_schema(k)
                                    samples = samples or self._try_get_samples(k)
                                    if cols:
                                        canonical_candidate = k
                                        break

                    # final key exposed to downstream: prefer store's canonical if available
                    table_key = canonical_candidate or str(raw)

                    try:
                        ss = getattr(self._tools, "_schema_store", None)
                        if ss is not None and hasattr(ss, "get_table_canonical"):
                            store_key = ss.get_table_canonical(raw) or ss.get_table_canonical(canonical_candidate)
                            if store_key:
                                table_key = store_key
                                cols = cols or ss.get_schema(store_key) or []
                                samples = samples or ss.get_sample_rows(store_key) or []
                    except Exception:
                        LOG.debug("Could not prefer store canonical key for %s", raw, exc_info=True)

                    # attempt to find a csv path in the store for this key
                    path = None
                    try:
                        # prefer store-backed key lookup
                        ss = getattr(self._tools, "_schema_store", None)
                        if ss is not None:
                            # try store_key first
                            lookup_key = table_key
                            path = self._try_get_path_from_store(lookup_key)
                            # if not found, try the original raw/canonical forms
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
                    out[_canonical_name_from_input(raw) or str(raw)] = {"columns": [], "sample_rows": [], "path": None, "canonical": _canonical_name_from_input(raw)}

            LOG.debug("ContextNode.run completed for: %s -> keys=%s", csv_names, list(out.keys()))
            return out
        except Exception as e:
            logger.exception("ContextNode.run failed unexpectedly")
            raise CustomException(e, sys)
