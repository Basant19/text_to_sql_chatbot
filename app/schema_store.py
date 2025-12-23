# D:\text_to_sql_bot\app\schema_store.py
"""
SchemaStore
===========

Centralized, persistent registry for CSV-backed schemas used by the
Text-to-SQL system.

This module is the SINGLE SOURCE OF TRUTH for all dataset identities
across UI, LLM reasoning, and SQL execution.

----------------------------------------------------------------------
NAMING MODEL (CRITICAL — DO NOT VIOLATE)
----------------------------------------------------------------------

Each dataset is represented by THREE distinct identifiers:

1. display_name
   - Human-readable
   - UI-facing only
   - Stable across restarts
   - Example: "sales_data"

2. canonical_name
   - Internal logical identifier
   - SQL-safe, deterministic
   - Used ONLY inside the LangGraph / LLM reasoning layer
   - NEVER executed in DuckDB
   - NEVER exposed to UI

3. runtime_table
   - Physical DuckDB table/view name
   - SQL-safe
   - Used ONLY at execution time

----------------------------------------------------------------------
HARD GUARANTEES
----------------------------------------------------------------------

- Exactly ONE runtime_table per CSV
- Canonical names NEVER reach DuckDB
- UI NEVER sees canonical or runtime names
- DuckDB NEVER sees display or canonical names
- SchemaStore is thread-safe
- SchemaStore is singleton-aware (explicit, no magic)
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


def _safe_sql_name(name: str) -> str:
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


def _display_name_from_filename(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    name = re.sub(r"^[a-f0-9]{8,}_", "", name)
    return _normalize(name) or f"dataset_{_short_uuid()}"


# ----------------------------------------------------------------------
# SchemaStore
# ----------------------------------------------------------------------

class SchemaStore:
    """
    Persistent registry of CSV schemas.

    Characteristics
    ----------------
    - Thread-safe
    - Explicit singleton lifecycle
    - Disk-backed (JSON)
    - UI-safe API surface
    """

    _instance: Optional["SchemaStore"] = None
    _instance_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton helpers
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "SchemaStore":
        """
        Return the global SchemaStore singleton.

        This is the DEFAULT and RECOMMENDED access path.
        """
        with cls._instance_lock:
            if cls._instance is None:
                base = os.path.join(os.getcwd(), "data")
                store_path = os.path.join(base, "schema_store.json")
                cls._instance = cls(store_path=store_path)
                logger.info("SchemaStore singleton created at %s", store_path)
            return cls._instance

    @classmethod
    def set_instance(cls, instance: "SchemaStore") -> None:
        """
        Explicitly inject a SchemaStore singleton.

        Intended for:
        - UI bootstrap
        - Tests
        - Controlled dependency injection

        WARNING:
        --------
        Do NOT call this repeatedly in production code.
        """
        if not isinstance(instance, SchemaStore):
            raise TypeError("set_instance expects a SchemaStore instance")

        with cls._instance_lock:
            cls._instance = instance
            logger.info("SchemaStore singleton explicitly set")

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, store_path: Optional[str] = None):
        try:
            base = os.path.join(os.getcwd(), "data")
            self.store_path = store_path or os.path.join(base, "schema_store.json")
            _ensure_dir(os.path.dirname(self.store_path))

            self._lock = threading.RLock()

            # canonical_name -> schema entry
            self._schemas: Dict[str, Dict[str, Any]] = {}

            # display_name -> canonical_name
            self._display_map: Dict[str, str] = {}

            self._load()
            self._patch_missing_runtime_tables()

        except Exception as e:
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
                self._display_map = data.get("display_map", {}) or {}
            except Exception:
                logger.exception("SchemaStore load failed, resetting")
                self._schemas = {}
                self._display_map = {}

    def _save(self) -> None:
        with self._lock:
            _atomic_write(
                self.store_path,
                {
                    "schemas": self._schemas,
                    "display_map": self._display_map,
                },
            )

    # ------------------------------------------------------------------
    # Backward-compatibility patch
    # ------------------------------------------------------------------

    def _patch_missing_runtime_tables(self) -> None:
        """
        Patch older schema entries missing `runtime_table`.
        """
        patched = False

        for canonical, entry in self._schemas.items():
            if "runtime_table" not in entry:
                display = entry.get("display") or canonical
                runtime = _safe_sql_name(display)
                entry["runtime_table"] = runtime
                patched = True
                logger.warning(
                    "Patched missing runtime_table | canonical=%s runtime=%s",
                    canonical,
                    runtime,
                )

        if patched:
            self._save()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_csv(self, path: str) -> str:
        """
        Register a CSV file.

        Returns
        -------
        str
            Canonical name of the dataset.
        """
        meta = load_csv_metadata(path)

        abs_path = os.path.abspath(path)
        meta["path"] = abs_path

        try:
            meta["file_hash"] = _file_sha256(abs_path)
        except Exception:
            pass

        display = _display_name_from_filename(abs_path)
        canonical = _safe_sql_name(f"{display}_{_short_uuid()}")
        runtime = _safe_sql_name(display)

        with self._lock:
            if display in self._display_map:
                return self._display_map[display]

            columns_raw = meta.get("columns") or []
            columns_norm = [_normalize(c) for c in columns_raw]

            entry = {
                "canonical": canonical,
                "display": display,
                "runtime_table": runtime,
                "path": abs_path,
                "columns": columns_raw,
                "columns_normalized": columns_norm,
                "meta": meta,
            }

            self._schemas[canonical] = entry
            self._display_map[display] = canonical

            self._save()
            logger.info(
                "Schema registered | display=%s canonical=%s runtime=%s",
                display,
                canonical,
                runtime,
            )

            return canonical

    # ------------------------------------------------------------------
    # Lookup (CORE API)
    # ------------------------------------------------------------------

    def resolve_table(self, name: str) -> Optional[str]:
        if not name:
            return None
        if name in self._schemas:
            return name
        return self._display_map.get(_normalize(name))

    def get_schema_entry(self, name: str) -> Optional[Dict[str, Any]]:
        can = self.resolve_table(name)
        return self._schemas.get(can) if can else None

    get = get_entry = get_metadata = get_schema_entry

    def get_runtime_table(self, name: str) -> Optional[str]:
        entry = self.get_schema_entry(name)
        return entry.get("runtime_table") if entry else None

    def get_csv_path(self, name: str) -> Optional[str]:
        entry = self.get_schema_entry(name)
        return entry.get("path") if entry else None

    def get_schema(self, name: str) -> List[str]:
        entry = self.get_schema_entry(name)
        return entry.get("columns_normalized", []) if entry else []

    # ------------------------------------------------------------------
    # UI-safe methods
    # ------------------------------------------------------------------

    def list_csvs(self) -> List[str]:
        with self._lock:
            return sorted(self._display_map.keys())

    def list_csvs_meta(self) -> List[Dict[str, Any]]:
        out = []
        with self._lock:
            for display, canonical in self._display_map.items():
                e = self._schemas.get(canonical, {})
                out.append(
                    {
                        "display": display,
                        "canonical": canonical,
                        "runtime_table": e.get("runtime_table"),
                        "path": e.get("path"),
                        "columns": e.get("columns", []),
                    }
                )
        return out

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def validate_table_and_columns(
        self, table: str, columns: Optional[List[str]]
    ) -> Tuple[bool, List[str], List[str]]:

        can = self.resolve_table(table)
        if not can:
            return False, [table], columns or []

        if not columns:
            return True, [], []

        defined = set(self.get_schema(can))
        missing = [c for c in columns if _normalize(c) not in defined]

        return not missing, [], missing
