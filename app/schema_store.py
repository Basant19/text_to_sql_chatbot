# D:\text_to_sql_bot\app\schema_store.py
"""
SchemaStore: Centralized registry for CSV schemas with persistent JSON storage.

KEY DESIGN GUARANTEES (FIXED):
1. Exactly ONE canonical table name per CSV
2. Canonical name ALWAYS:
   - human readable
   - starts with a letter
   - SQL-safe for LLMs
3. Hashes / filenames are INTERNAL ONLY (never exposed as tables)
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
    s = re.sub(r"[^a-zA-Z0-9_]", "_", str(name))
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s


def _safe_canonical(name: str) -> str:
    """
    Enforce SQL + LLM safe table name:
    - normalized
    - must start with a letter
    """
    base = _normalize(name)
    if not base:
        base = f"table_{uuid.uuid4().hex[:6]}"
    if not base[0].isalpha():
        base = f"t_{base}"
    return base


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
    SINGLETON-AWARE (required by GraphBuilder & LangGraph).
    """

    _instance: Optional["SchemaStore"] = None
    _instance_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton helpers
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "SchemaStore":
        with cls._instance_lock:
            if cls._instance is None:
                try:
                    base = os.path.join(os.getcwd(), "data")
                    store_path = os.path.join(base, "schema_store.json")
                    cls._instance = cls(store_path=store_path)
                    logger.info("SchemaStore singleton created at %s", store_path)
                except Exception as e:
                    logger.exception("Failed to create SchemaStore singleton")
                    raise CustomException(e, sys)
            return cls._instance

    @classmethod
    def set_instance(cls, instance: "SchemaStore") -> None:
        with cls._instance_lock:
            cls._instance = instance
            logger.info("SchemaStore singleton injected externally")

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

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
            except Exception:
                logger.exception("SchemaStore load failed, resetting")
                self._schemas = {}
                self._alias_map = {}

    def _save(self) -> None:
        with self._lock:
            _atomic_write(
                self.store_path,
                {"schemas": self._schemas, "alias_map": self._alias_map},
            )

    # ------------------------------------------------------------------
    # Registration (🔥 CORE FIX)
    # ------------------------------------------------------------------

    def register_from_csv(self, csv_path: str) -> str:
        meta = load_csv_metadata(csv_path)

        filename = os.path.splitext(os.path.basename(csv_path))[0]
        canonical = _safe_canonical(filename)

        meta["path"] = os.path.abspath(csv_path)
        meta["canonical_name"] = canonical
        meta["aliases"] = []  # aliases are NOT tables

        return self.register(canonical, meta)

    def register(self, canonical: str, meta: Dict[str, Any]) -> str:
        with self._lock:
            key = _safe_canonical(canonical)

            # deduplicate same file
            new_path = meta.get("path")
            new_hash = meta.get("file_hash")
            for k, entry in self._schemas.items():
                if new_path and entry.get("path") == new_path:
                    return k
                if new_hash and entry.get("meta", {}).get("file_hash") == new_hash:
                    entry["path"] = new_path
                    self._save()
                    return k

            # handle name collision
            if key in self._schemas:
                key = f"{key}_{_short_uuid()}"

            columns_raw = meta.get("columns") or []
            columns_norm = [_normalize(c) for c in columns_raw]
            meta["columns_normalized"] = columns_norm

            entry = {
                "canonical": key,
                "aliases": [],
                "path": new_path,
                "columns": columns_raw,
                "columns_normalized": columns_norm,
                "meta": meta,
            }

            self._schemas[key] = entry
            self._alias_map[key] = key  # ONLY canonical is valid

            self._save()
            logger.info("Schema registered: %s", key)
            return key

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_table_canonical(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        return self._alias_map.get(_normalize(name))

    def get_schema_entry(self, name: str) -> Optional[Dict[str, Any]]:
        can = self.get_table_canonical(name)
        return self._schemas.get(can) if can else None

    # Backward compatibility
    get = get_entry = get_schema_entry
    get_metadata = get_schema_entry

    def get_path(self, name: str) -> Optional[str]:
        entry = self.get_schema_entry(name)
        return entry.get("path") if entry else None

    get_csv_path = get_file_path = get_path

    def get_schema(self, name: str) -> List[str]:
        entry = self.get_schema_entry(name)
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
    # CSV helpers (used by app.py)
    # ------------------------------------------------------------------

    def add_csv(
        self,
        path: str,
        csv_name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> str:
        meta = load_csv_metadata(path)

        abs_path = os.path.abspath(path)
        meta["path"] = abs_path

        try:
            meta["file_hash"] = _file_sha256(abs_path)
        except Exception:
            pass

        canonical = _safe_canonical(csv_name or os.path.splitext(os.path.basename(path))[0])
        meta["canonical_name"] = canonical
        meta["aliases"] = []  # aliases intentionally ignored

        return self.register(canonical, meta)

    def list_csvs_meta(self) -> List[Dict[str, Any]]:
        """
        UI-safe listing:
        ONE CSV → ONE schema option
        """
        out = []
        with self._lock:
            for k, e in self._schemas.items():
                out.append(
                    {
                        "key": k,
                        "canonical": k,
                        "path": e.get("path"),
                        "columns": e.get("columns", []),
                        "columns_normalized": e.get("columns_normalized", []),
                        "sample_rows": e.get("meta", {}).get("sample_rows", []),
                    }
                )
        return out
