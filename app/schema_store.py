# File: app/schema_store.py
"""
SchemaStore: Centralized registry for CSV schemas with persistent JSON storage.
...
(kept same header docs)
"""
from __future__ import annotations
import os
import json
import threading
import uuid
import re
import sys
from typing import List, Dict, Any, Optional, Tuple

from app.logger import get_logger
from app.exception import CustomException
from app.csv_loader import load_csv_metadata

logger = get_logger("schema_store")


# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    """Ensure directory exists before file write operations."""
    if path:
        os.makedirs(path, exist_ok=True)


def _atomic_write(path: str, data: Any) -> None:
    """
    Atomic JSON write:
    - Write to <file>.tmp
    - fsync()
    - Safely replace original
    """
    dirpath = os.path.dirname(path) or "."
    _ensure_dir(dirpath)

    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

    os.replace(tmp, path)


def _normalize(name: Optional[str]) -> str:
    """Normalize names: lowercase, keep only alnum+underscore, compress repeats."""
    if not name:
        return ""
    s = re.sub(r"[^\w]", "_", str(name))
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or str(name).lower()


def _short_uuid() -> str:
    """8-char UUID suffix used to avoid canonical collisions."""
    return uuid.uuid4().hex[:8]


# ----------------------------------------------------------------------
# SchemaStore Class
# ----------------------------------------------------------------------


class SchemaStore:
    """
    Manages a persistent set of CSV schemas.
    Stored structure and behavior as described in header.
    """

    def __init__(self, store_path: Optional[str] = None):
        try:
            default_dir = os.path.join(os.getcwd(), "data")
            self.store_path = store_path or os.path.join(default_dir, "schema_store.json")

            _ensure_dir(os.path.dirname(self.store_path))
            self._lock = threading.RLock()

            self._schemas: Dict[str, Dict[str, Any]] = {}
            self._alias_map: Dict[str, str] = {}

            self._load()
        except Exception as e:
            logger.exception("SchemaStore initialization failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load schema_store.json into memory."""
        with self._lock:
            if not os.path.exists(self.store_path):
                return
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._schemas = data.get("schemas", {}) or {}
                self._alias_map = data.get("alias_map", {}) or {}
            except Exception as e:
                logger.exception(
                    "Failed to load SchemaStore (%s). Resetting. Error: %s", self.store_path, e
                )
                self._schemas = {}
                self._alias_map = {}

    def _save(self) -> None:
        """Persist all schemas to disk."""
        with self._lock:
            _atomic_write(
                self.store_path,
                {
                    "schemas": self._schemas,
                    "alias_map": self._alias_map,
                },
            )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_from_csv(self, csv_path: str) -> str:
        """
        Extract metadata using load_csv_metadata() and register as schema.
        Returns canonical name.
        """
        try:
            meta = load_csv_metadata(csv_path)
        except Exception as e:
            logger.exception("CSV metadata load failed: %s", csv_path)
            raise CustomException(e, sys)

        suggested = meta.get("canonical_name") or _normalize(
            os.path.splitext(os.path.basename(csv_path))[0]
        )
        return self.register(suggested, meta)

    def register(self, canonical: str, meta: Dict[str, Any]) -> str:
        """
        Register schema metadata.
        Auto-normalizes canonical name.
        Avoids conflicts using UUID suffix if paths differ.
        """
        with self._lock:
            try:
                c = _normalize(canonical)

                # Validate metadata fields
                columns_raw: List[str] = meta.get("columns") or []
                columns_norm: List[str] = [
                    _normalize(col) for col in meta.get("columns_normalized", columns_raw)
                ]

                aliases: List[str] = []
                for a in meta.get("aliases", []) or []:
                    if a:
                        aliases.append(str(a))
                if meta.get("original_name"):
                    aliases.append(str(meta["original_name"]))
                if meta.get("table_name"):
                    aliases.append(str(meta["table_name"]))

                # de-dup and normalize aliases for storage
                normalized_aliases = []
                for a in aliases:
                    n = _normalize(a)
                    if n and n not in normalized_aliases:
                        normalized_aliases.append(n)

                key = c
                if key in self._schemas:  # collision check
                    old_path = self._schemas[key].get("path")
                    new_path = meta.get("path")
                    if old_path and new_path and os.path.abspath(old_path) != os.path.abspath(new_path):
                        key = f"{c}_{_short_uuid()}"

                schema_entry = {
                    "canonical": key,
                    "aliases": normalized_aliases,
                    "path": os.path.abspath(meta.get("path")) if meta.get("path") else meta.get("path"),
                    "columns": columns_raw,
                    "columns_normalized": columns_norm,
                    "meta": meta,
                }

                self._schemas[key] = schema_entry

                # Update alias map
                for a in schema_entry["aliases"]:
                    self._alias_map[a] = key

                # Canonical also acts as alias
                self._alias_map[_normalize(key)] = key

                self._save()
                logger.info("Schema registered: %s (cols=%d)", key, len(columns_norm))
                return key
            except Exception as e:
                logger.exception("SchemaStore.register failed")
                raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Unregister
    # ------------------------------------------------------------------

    def unregister(self, name_or_alias: str) -> bool:
        """Remove schema entry and associated aliases."""
        with self._lock:
            can = self.get_table_canonical(name_or_alias)
            if not can:
                return False

            self._schemas.pop(can, None)

            for alias, target in list(self._alias_map.items()):
                if target == can:
                    self._alias_map.pop(alias, None)

            self._save()
            logger.info("Schema unregistered: %s", can)
            return True

    # ------------------------------------------------------------------
    # Lookup Methods
    # ------------------------------------------------------------------

    def has_table(self, name_or_alias: str) -> bool:
        return bool(self.get_table_canonical(name_or_alias))

    def get_table_canonical(self, name_or_alias: Optional[str]) -> Optional[str]:
        """Normalize alias and resolve canonical key."""
        if not name_or_alias:
            return None
        return self._alias_map.get(_normalize(name_or_alias))

    def get_schema_entry(self, name_or_alias: str) -> Optional[Dict[str, Any]]:
        """Return full metadata dict (entry) for canonical or alias."""
        can = self.get_table_canonical(name_or_alias)
        return self._schemas.get(can) if can else None

    # Backwards-compatible 'get' / 'get_entry' methods used by other modules:
    def get_entry(self, name_or_alias: str) -> Optional[Dict[str, Any]]:
        return self.get_schema_entry(name_or_alias)

    def get(self, name_or_alias: str) -> Optional[Dict[str, Any]]:
        return self.get_schema_entry(name_or_alias)

    def get_metadata(self, name_or_alias: str) -> Optional[Dict[str, Any]]:
        return self.get_schema_entry(name_or_alias)

    def get_path(self, name_or_alias: str) -> Optional[str]:
        """Return absolute CSV path for the table if present."""
        entry = self.get_schema_entry(name_or_alias)
        if not entry:
            return None
        p = entry.get("path") or entry.get("meta", {}).get("path")
        if p:
            try:
                return os.path.abspath(p)
            except Exception:
                return p
        return None

    def get_csv_path(self, name_or_alias: str) -> Optional[str]:
        return self.get_path(name_or_alias)

    def get_file_path(self, name_or_alias: str) -> Optional[str]:
        return self.get_path(name_or_alias)

    def get_schema(self, name_or_alias: str) -> List[str]:
        """Return normalized column names only."""
        entry = self.get_schema_entry(name_or_alias)
        return entry.get("columns_normalized", []) if entry else []

    def get_sample_rows(self, name_or_alias: str) -> List[Dict[str, Any]]:
        """Return sample rows for UI preview."""
        entry = self.get_schema_entry(name_or_alias)
        return entry.get("meta", {}).get("sample_rows", []) if entry else []

    def list_tables(self) -> List[str]:
        """List canonical table names."""
        with self._lock:
            return list(self._schemas.keys())

    def get_columns(self, name_or_alias: str) -> List[str]:
        return self.get_schema(name_or_alias)

    def has_column(self, table_name_or_alias: str, column_name_or_alias: str) -> bool:
        cols = set(self.get_columns(table_name_or_alias))
        return _normalize(column_name_or_alias) in cols

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_table_and_columns(
        self, table_name: str, columns: Optional[List[str]]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate existence of:
        - table
        - columns
        Returns:
            (ok, missing_tables, missing_columns)
        """
        missing_tables: List[str] = []
        missing_columns: List[str] = []

        can = self.get_table_canonical(table_name)
        if not can:
            return False, [table_name], columns or []

        if not columns:
            return True, [], []

        defined = set(self.get_columns(can))
        for col in columns:
            if _normalize(col) not in defined:
                missing_columns.append(col)

        return len(missing_columns) == 0, [], missing_columns

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def dump(self) -> Dict[str, Any]:
        """Return full JSON-serializable dataset."""
        with self._lock:
            return {"schemas": self._schemas, "alias_map": self._alias_map}

    def clear(self) -> None:
        """Erase everything (dev only)."""
        with self._lock:
            self._schemas.clear()
            self._alias_map.clear()
            try:
                if os.path.exists(self.store_path):
                    os.remove(self.store_path)
            except Exception:
                pass
            logger.info("SchemaStore cleared")

    # -------------------------
    # Backwards-compatible helpers used by app.py
    # -------------------------
    def add_csv(self, path: str, csv_name: Optional[str] = None, aliases: Optional[List[str]] = None) -> str:
        """
        Backwards-compatible wrapper used by app.py.
        """
        try:
            meta = load_csv_metadata(path)
        except Exception as e:
            logger.exception("add_csv: failed to load metadata for %s", path)
            raise CustomException(e, sys)

        # allow provided overrides
        if csv_name:
            meta["canonical_name"] = csv_name
        if aliases:
            meta_aliases = list(meta.get("aliases") or [])
            for a in aliases:
                if a and a not in meta_aliases:
                    meta_aliases.append(a)
            meta["aliases"] = meta_aliases

        meta["path"] = path

        key = self.register(
            meta.get("canonical_name")
            or csv_name
            or os.path.splitext(os.path.basename(path))[0],
            meta,
        )
        return key

    def list_csvs_meta(self) -> List[Dict[str, Any]]:
        """
        Return a list of schema metadata entries suitable for UI consumption.
        """
        out: List[Dict[str, Any]] = []
        with self._lock:
            for key, entry in self._schemas.items():
                try:
                    friendly = entry.get("meta", {}).get("friendly") or entry.get("canonical") or key
                    obj = {
                        "key": key,
                        "canonical": entry.get("canonical", key),
                        "path": entry.get("path"),
                        "aliases": entry.get("aliases", []) or [],
                        "columns": entry.get("columns", []) or [],
                        "columns_normalized": entry.get("columns_normalized", []) or [],
                        "sample_rows": entry.get("meta", {}).get("sample_rows", []) or [],
                        "friendly": friendly,
                    }
                    out.append(obj)
                except Exception:
                    logger.exception("list_csvs_meta: failed to format schema entry %s", key)
        return out
