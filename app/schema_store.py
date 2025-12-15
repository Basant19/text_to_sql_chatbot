# File: app/schema_store.py
"""
SchemaStore: Centralized registry for CSV schemas with persistent JSON storage.
"""
from __future__ import annotations

import os
import json
import threading
import uuid
import re
import sys
import hashlib
from typing import List, Dict, Any, Optional, Tuple

from app.logger import get_logger
from app.exception import CustomException
from app.csv_loader import load_csv_metadata

logger = get_logger("schema_store")

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _atomic_write(path: str, data: Any) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
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
    if not name:
        return ""
    s = re.sub(r"[^\w]", "_", str(name))
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s


def _short_uuid() -> str:
    return uuid.uuid4().hex[:8]


def _file_sha256(path: str, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# SchemaStore
# ----------------------------------------------------------------------

class SchemaStore:
    """
    Persistent registry of CSV schemas.
    """

    def __init__(self, store_path: Optional[str] = None):
        try:
            base = os.path.join(os.getcwd(), "data")
            self.store_path = store_path or os.path.join(base, "schema_store.json")

            _ensure_dir(os.path.dirname(self.store_path))
            self._lock = threading.RLock()

            self._schemas: Dict[str, Dict[str, Any]] = {}
            self._alias_map: Dict[str, str] = {}

            self._load()
        except Exception as e:
            logger.exception("SchemaStore init failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with self._lock:
            if not os.path.exists(self.store_path):
                return
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._schemas = data.get("schemas", {}) or {}
                self._alias_map = data.get("alias_map", {}) or {}
            except Exception as e:
                logger.exception("SchemaStore load failed, resetting: %s", e)
                self._schemas = {}
                self._alias_map = {}

    def _save(self) -> None:
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
        try:
            meta = load_csv_metadata(csv_path)
        except Exception as e:
            raise CustomException(e, sys)

        canonical = meta.get("canonical_name") or _normalize(
            os.path.splitext(os.path.basename(csv_path))[0]
        )
        return self.register(canonical, meta)

    def register(self, canonical: str, meta: Dict[str, Any]) -> str:
        with self._lock:
            try:
                base = _normalize(canonical)

                columns_raw = meta.get("columns") or []
                columns_norm = (
                    [_normalize(c) for c in meta.get("columns_normalized", [])]
                    if meta.get("columns_normalized")
                    else [_normalize(c) for c in columns_raw]
                )

                aliases = []
                for a in meta.get("aliases", []) or []:
                    na = _normalize(a)
                    if na and na not in aliases:
                        aliases.append(na)

                for extra in (meta.get("original_name"), meta.get("table_name")):
                    na = _normalize(extra)
                    if na and na not in aliases:
                        aliases.append(na)

                key = base
                if key in self._schemas:
                    old = self._schemas[key].get("path")
                    new = meta.get("path")
                    if old and new and os.path.abspath(old) != os.path.abspath(new):
                        key = f"{base}_{_short_uuid()}"

                entry = {
                    "canonical": key,
                    "aliases": aliases,
                    "path": os.path.abspath(meta.get("path")) if meta.get("path") else None,
                    "columns": columns_raw,
                    "columns_normalized": columns_norm,
                    "meta": meta,
                }

                meta.setdefault("columns_normalized", columns_norm)

                self._schemas[key] = entry

                for a in aliases:
                    self._alias_map[a] = key
                self._alias_map[_normalize(key)] = key

                self._save()
                logger.info("Schema registered: %s", key)
                return key
            except Exception as e:
                logger.exception("SchemaStore.register failed")
                raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def unregister(self, name_or_alias: str) -> bool:
        with self._lock:
            can = self.get_table_canonical(name_or_alias)
            if not can:
                return False

            self._schemas.pop(can, None)
            for a in list(self._alias_map):
                if self._alias_map.get(a) == can:
                    self._alias_map.pop(a, None)

            self._save()
            logger.info("Schema unregistered: %s", can)
            return True

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_table_canonical(self, name_or_alias: Optional[str]) -> Optional[str]:
        if not name_or_alias:
            return None
        return self._alias_map.get(_normalize(name_or_alias))

    def get_schema_entry(self, name_or_alias: str) -> Optional[Dict[str, Any]]:
        can = self.get_table_canonical(name_or_alias)
        return self._schemas.get(can) if can else None

    # Backwards-compatible aliases
    get = get_entry = get_schema_entry
    get_metadata = get_schema_entry

    def get_path(self, name_or_alias: str) -> Optional[str]:
        entry = self.get_schema_entry(name_or_alias)
        return entry.get("path") if entry else None

    get_csv_path = get_file_path = get_path

    def get_schema(self, name_or_alias: str) -> List[str]:
        entry = self.get_schema_entry(name_or_alias)
        return entry.get("columns_normalized", []) if entry else []

    def list_tables(self) -> List[str]:
        with self._lock:
            return list(self._schemas.keys())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_table_and_columns(
        self, table: str, columns: Optional[List[str]]
    ) -> Tuple[bool, List[str], List[str]]:

        can = self.get_table_canonical(table)
        if not can:
            return False, [table], columns or []

        if not columns:
            return True, [], []

        defined = set(self.get_schema(can))
        missing = [c for c in columns if _normalize(c) not in defined]

        return not missing, [], missing

    # ------------------------------------------------------------------
    # Backwards-compatible helpers (used by app.py)
    # ------------------------------------------------------------------

    def add_csv(
        self,
        path: str,
        csv_name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> str:
        meta = load_csv_metadata(path)

        file_hash = None
        if os.path.exists(path):
            try:
                file_hash = _file_sha256(path)
                meta["file_hash"] = file_hash
            except Exception:
                pass

        abs_path = os.path.abspath(path)

        with self._lock:
            for k, entry in self._schemas.items():
                if entry.get("path") == abs_path:
                    return k
                if file_hash and entry.get("meta", {}).get("file_hash") == file_hash:
                    entry["path"] = abs_path
                    self._save()
                    return k

        if csv_name:
            meta["canonical_name"] = csv_name
        if aliases:
            meta["aliases"] = list(set(meta.get("aliases", []) + aliases))

        meta["path"] = abs_path
        return self.register(meta.get("canonical_name"), meta)

    def list_csvs_meta(self) -> List[Dict[str, Any]]:
        out = []
        with self._lock:
            for k, e in self._schemas.items():
                out.append(
                    {
                        "key": k,
                        "canonical": e.get("canonical"),
                        "path": e.get("path"),
                        "aliases": e.get("aliases", []),
                        "columns": e.get("columns", []),
                        "columns_normalized": e.get("columns_normalized", []),
                        "sample_rows": e.get("meta", {}).get("sample_rows", []),
                    }
                )
        return out
