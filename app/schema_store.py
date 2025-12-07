"""
SchemaStore: manage CSV schema metadata persisted to JSON.

Behavior notes:
- SchemaStore persists a mapping of canonical schema entries and alias-to-canonical map.
- get_schema(name) returns a list of normalized column names (convenient for LLM prompt builders).
- get_schema_entry(name) returns the full stored schema dict if callers need the complete metadata.
"""
from __future__ import annotations
import os
import json
import threading
import uuid
import re
from typing import List, Dict, Any, Optional, Tuple

from app.logger import get_logger
from app.exception import CustomException
from app.csv_loader import load_csv_metadata

logger = get_logger("schema_store")


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _atomic_write(path: str, data: Any) -> None:
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
    """Normalize names to lowercase with underscores (deterministic)."""
    if not name:
        return ""
    s = str(name)
    # keep alnum and underscore; replace others with underscore
    s = re.sub(r"[^\w]", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_").lower()
    return s or name.lower()


def _short_uuid() -> str:
    return uuid.uuid4().hex[:8]


class SchemaStore:
    """
    Thread-safe schema store. Persists schemas to disk.

    Stored structure:
    {
      "schemas": {
         "<canonical>": {
             "canonical": "<canonical>",
             "aliases": ["orig_filename", "orig_table_name", ...],
             "path": "<csv_path>",
             "columns": ["Raw Header 1", ...],
             "columns_normalized": ["raw_header_1", ...],
             "meta": { ... full metadata from load_csv_metadata ... }
         }, ...
      },
      "alias_map": {
         "<alias_normalized>": "<canonical>",
         ...
      }
    }
    """

    def __init__(self, store_path: Optional[str] = None):
        default_dir = os.path.join(os.getcwd(), "data")
        self.store_path = store_path or os.path.join(default_dir, "schema_store.json")
        _ensure_dir(os.path.dirname(self.store_path) or ".")
        self._lock = threading.RLock()
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._alias_map: Dict[str, str] = {}
        self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        with self._lock:
            if os.path.exists(self.store_path):
                try:
                    with open(self.store_path, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    self._schemas = d.get("schemas", {}) or {}
                    self._alias_map = d.get("alias_map", {}) or {}
                except Exception as e:
                    logger.exception("Failed to load SchemaStore (%s), starting empty: %s", self.store_path, e)
                    self._schemas = {}
                    self._alias_map = {}

    def _save(self) -> None:
        with self._lock:
            _atomic_write(self.store_path, {"schemas": self._schemas, "alias_map": self._alias_map})

    # ---------- registration ----------
    def register_from_csv(self, csv_path: str) -> str:
        """
        Load metadata using load_csv_metadata and register the schema.
        Returns canonical name.
        """
        try:
            meta = load_csv_metadata(csv_path)
        except Exception as e:
            logger.exception("Failed to load CSV metadata for %s", csv_path)
            raise CustomException(e)

        # prefer canonical_name if provided by loader else derive from filename
        suggested = meta.get("canonical_name") or _normalize(os.path.splitext(os.path.basename(csv_path))[0])
        return self.register(suggested, meta)

    def register(self, canonical: str, meta: Dict[str, Any]) -> str:
        """
        Register a schema dict (meta should include 'columns' and 'columns_normalized' optional).
        Returns canonical key used (may be suffixed to avoid collision).
        """
        with self._lock:
            c = _normalize(canonical)
            # extract columns (raw) and normalized
            columns_raw: List[str] = meta.get("columns", []) or []
            if meta.get("columns_normalized"):
                columns_norm = [ _normalize(x) for x in meta["columns_normalized"] ]
            else:
                columns_norm = [ _normalize(x) for x in columns_raw ]

            aliases: List[str] = []
            # include provided aliases, original name, table_name if present
            for a in meta.get("aliases", []) or []:
                if a:
                    aliases.append(str(a))
            if meta.get("original_name"):
                aliases.append(str(meta["original_name"]))
            if meta.get("table_name"):
                aliases.append(str(meta["table_name"]))

            # ensure uniqueness of canonical key: if existing but different path, append short uuid
            key = c
            if key in self._schemas:
                existing_path = self._schemas[key].get("path")
                new_path = meta.get("path")
                if existing_path and new_path and os.path.abspath(existing_path) != os.path.abspath(new_path):
                    key = f"{c}_{_short_uuid()}"

            schema_entry = {
                "canonical": key,
                "aliases": list(dict.fromkeys([_normalize(a) for a in aliases if a])),  # normalized aliases
                "path": meta.get("path"),
                "columns": columns_raw,
                "columns_normalized": columns_norm,
                "meta": meta
            }

            self._schemas[key] = schema_entry

            # update alias map
            for a in schema_entry["aliases"]:
                self._alias_map[a] = key
            # also map canonical itself
            self._alias_map[_normalize(key)] = key

            self._save()
            logger.info("SchemaStore: registered %s (cols=%d)", key, len(columns_norm))
            return key

    def unregister(self, name_or_alias: str) -> bool:
        """
        Remove a registered schema by canonical or alias.
        Returns True if removed.
        """
        with self._lock:
            can = self.get_table_canonical(name_or_alias)
            if not can:
                return False
            # remove schema
            if can in self._schemas:
                del self._schemas[can]
            # remove alias map entries pointing to this canonical
            to_rm = [k for k, v in list(self._alias_map.items()) if v == can]
            for k in to_rm:
                del self._alias_map[k]
            self._save()
            logger.info("SchemaStore: unregistered %s", can)
            return True

    # ---------- lookup & helpers ----------
    def has_table(self, name_or_alias: str) -> bool:
        return bool(self.get_table_canonical(name_or_alias))

    def get_table_canonical(self, name_or_alias: Optional[str]) -> Optional[str]:
        if not name_or_alias:
            return None
        n = _normalize(name_or_alias)
        with self._lock:
            return self._alias_map.get(n)

    def get_schema_entry(self, name_or_alias: str) -> Optional[Dict[str, Any]]:
        """
        Return the full stored schema entry (dict) for the canonical name or alias.
        """
        can = self.get_table_canonical(name_or_alias)
        if not can:
            return None
        return self._schemas.get(can)

    def get_schema(self, name_or_alias: str) -> List[str]:
        """
        Return the list of normalized column names for the given table (canonical or alias).
        This is what LLM prompt builders and other call-sites expect.
        """
        entry = self.get_schema_entry(name_or_alias)
        if not entry:
            return []
        return entry.get("columns_normalized", []) or []

    def get_sample_rows(self, name_or_alias: str) -> List[Dict[str, Any]]:
        """
        Return the sample rows extracted from the CSV metadata (if available).
        """
        entry = self.get_schema_entry(name_or_alias)
        if not entry:
            return []
        meta = entry.get("meta", {}) or {}
        return meta.get("sample_rows", []) or []

    def list_tables(self) -> List[str]:
        with self._lock:
            return list(self._schemas.keys())

    def get_columns(self, name_or_alias: str) -> List[str]:
        """
        Alias for get_schema (keeps compatibility with older callers).
        """
        return self.get_schema(name_or_alias)

    def has_column(self, table_name_or_alias: str, column_name_or_alias: str) -> bool:
        cols = set(self.get_columns(table_name_or_alias))
        return _normalize(column_name_or_alias) in cols

    def validate_table_and_columns(self, table_name: str, columns: Optional[List[str]]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate existence of table and columns. Returns:
          (ok, missing_tables, missing_columns)
        - missing_tables: list with table_name if it doesn't exist, else []
        - missing_columns: list of columns not found (using normalized names)
        """
        missing_tables: List[str] = []
        missing_columns: List[str] = []

        can = self.get_table_canonical(table_name)
        if not can:
            missing_tables.append(table_name)
            return False, missing_tables, columns or []

        if not columns:
            return True, [], []

        cols_norm = set(self.get_columns(can))
        for col in columns:
            if _normalize(col) not in cols_norm:
                missing_columns.append(col)

        ok = len(missing_columns) == 0
        return ok, missing_tables, missing_columns

    def dump(self) -> Dict[str, Any]:
        with self._lock:
            return {"schemas": self._schemas, "alias_map": self._alias_map}

    def clear(self) -> None:
        with self._lock:
            self._schemas = {}
            self._alias_map = {}
            try:
                if os.path.exists(self.store_path):
                    os.remove(self.store_path)
            except Exception:
                pass
            logger.info("SchemaStore cleared")
