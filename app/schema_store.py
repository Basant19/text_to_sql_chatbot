
#D:\text_to_sql_bot\app\schema_store.py

"""
SchemaStore: manage CSV schema metadata persisted to JSON.

Features:
 - Atomic saves (.tmp -> replace)
 - Canonical name normalization
 - Fuzzy and alias matching for lookups
 - Thread-safe load/save
 - CSV header/sample extraction with safe fallbacks
 - Utilities: add_csv, update_schema, remove_schema, get_schema, list_csvs_meta
"""
from __future__ import annotations
import os
import sys
import json
import csv
import uuid
import threading
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("schema_store")


def _ensure_dir(path: str) -> None:
    if not path:
        return
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
            # not critical on all platforms
            pass
    os.replace(tmp, path)


def _canonical_name(name: Optional[str]) -> str:
    """
    Produce a stable canonical token for matching:
      - use basename
      - strip extension
      - lowercase
      - replace non-word sequences with underscore
      - strip
    """
    if not name:
        return ""
    base = os.path.basename(str(name))
    no_ext = os.path.splitext(base)[0]
    # normalize separators, collapse non-alnum to underscore
    import re

    token = re.sub(r"[^\w]+", "_", no_ext)
    return token.lower().strip()


def _read_csv_head_and_samples(path: str, sample_limit: int = 5) -> (List[str], List[Dict[str, Any]]):
    """
    Read header and up to sample_limit rows safely.
    Tries to handle files with or without header using csv.Sniffer.
    Returns (columns, samples)
    """
    columns: List[str] = []
    samples: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(8192)
            f.seek(0)
            sniffer = csv.Sniffer()
            has_header = False
            try:
                has_header = sniffer.has_header(text)
            except Exception:
                has_header = True  # optimistic default

            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                return [], []

            if has_header:
                header = rows[0]
                columns = [h.strip() for h in header]
                data_rows = rows[1: 1 + sample_limit]
            else:
                # create synthetic column names: col_0, col_1...
                first = rows[0]
                columns = [f"col_{i}" for i in range(len(first))]
                data_rows = rows[:sample_limit]

            # convert data_rows -> list of dicts matching columns length
            for r in data_rows:
                rowdict: Dict[str, Any] = {}
                for i, col in enumerate(columns):
                    try:
                        rowdict[col] = r[i] if i < len(r) else ""
                    except Exception:
                        rowdict[col] = ""
                samples.append(rowdict)
            return columns, samples
    except Exception as e:
        logger.warning("Failed to read CSV header/samples for %s: %s", path, e)
        # best-effort fallback: empty
        return [], []


def uuid4_hex() -> str:
    return uuid.uuid4().hex[:8]


class SchemaStore:
    """
    Persistent store for table schemas (CSV-based).

    Stored JSON structure (self.store_path):
      {
        "<store_key>": {
           "path": "<path>",
           "columns": [...],
           "columns_normalized": [...],    # lowercased/canonical tokens for matching
           "sample_rows": [...],
           "canonical": "<canonical>",
           "friendly": "<friendly>",
           "aliases": [...]
        },
        ...
      }
    """

    def __init__(self, store_path: Optional[str] = None, sample_limit: int = 5):
        try:
            data_dir = getattr(config, "DATA_DIR", "./data")
            self.store_path = store_path or os.path.join(data_dir, "schema_store.json")
            _ensure_dir(os.path.dirname(self.store_path) or ".")
            self.sample_limit = int(sample_limit or 5)
            self._lock = threading.RLock()
            self._store: Dict[str, Dict[str, Any]] = {}
            self._load_store()
            logger.info("SchemaStore initialized at %s", self.store_path)
        except Exception as e:
            logger.exception("Failed to initialize SchemaStore")
            raise CustomException(e, sys)

    # -------------------------
    # Persistence
    # -------------------------
    def _load_store(self) -> None:
        with self._lock:
            try:
                if os.path.exists(self.store_path):
                    with open(self.store_path, "r", encoding="utf-8") as f:
                        data = json.load(f) or {}
                        if not isinstance(data, dict):
                            logger.warning("SchemaStore file has unexpected shape; resetting store.")
                            self._store = {}
                            return
                        # normalize entries to expected shape
                        normalized: Dict[str, Dict[str, Any]] = {}
                        for k, v in data.items():
                            if not isinstance(v, dict):
                                continue
                            # ensure canonical and aliases exist
                            canonical = v.get("canonical") or _canonical_name(k)
                            friendly = v.get("friendly") or canonical
                            aliases = v.get("aliases") or []
                            # normalize alias tokens
                            norm_aliases = []
                            for a in aliases:
                                if a:
                                    norm_aliases.append(_canonical_name(a))
                            # ensure canonical token present
                            if canonical and _canonical_name(canonical) not in norm_aliases:
                                norm_aliases.append(_canonical_name(canonical))

                            # columns and normalized columns (backwards-compatible)
                            raw_columns = v.get("columns") or []
                            cols_norm = v.get("columns_normalized")
                            if not cols_norm:
                                # compute normalized tokens from raw columns if not present
                                cols_norm = [ _canonical_name(str(c)) for c in (raw_columns or []) ]

                            rec = {
                                "path": v.get("path"),
                                "columns": raw_columns,
                                "columns_normalized": cols_norm,
                                "sample_rows": v.get("sample_rows", [])[: self.sample_limit],
                                "canonical": canonical,
                                "friendly": friendly,
                                "aliases": sorted(list(dict.fromkeys(norm_aliases))),
                            }
                            normalized[k] = rec
                        self._store = normalized
                else:
                    self._store = {}
            except Exception as e:
                logger.exception("Failed to load schema store")
                raise CustomException(e, sys)

    def _save_store(self) -> None:
        with self._lock:
            try:
                _atomic_write(self.store_path, self._store)
            except Exception as e:
                logger.exception("Failed to save schema store")
                raise CustomException(e, sys)

    # -------------------------
    # Core operations
    # -------------------------
    def add_csv(self, csv_path: str, csv_name: Optional[str] = None, aliases: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add or register a CSV schema.

        Returns: store_key used to store the schema (may be the canonical name or a unique key on conflict)
        """
        with self._lock:
            try:
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"CSV file not found: {csv_path}")

                provided_key = csv_name or os.path.basename(csv_path)
                canonical = _canonical_name(provided_key)

                # If metadata provided (e.g., CSVLoader result), prefer it
                if metadata and isinstance(metadata, dict):
                    columns = metadata.get("columns", []) or []
                    sample_rows = metadata.get("sample_rows", [])[: self.sample_limit]
                    canonical = metadata.get("canonical") or canonical
                    friendly = metadata.get("table_name") or metadata.get("friendly") or canonical
                    aliases_list = metadata.get("aliases", []) or []
                else:
                    cols, samples = _read_csv_head_and_samples(csv_path, sample_limit=self.sample_limit)
                    columns = cols
                    sample_rows = samples
                    friendly = csv_name or canonical
                    aliases_list = aliases or []

                # normalized columns for reliable matching (lowercased canonical tokens)
                columns_normalized = [ _canonical_name(c) for c in (columns or []) ]

                # Build alias set (normalized tokens)
                alias_set = set()
                for a in aliases_list:
                    if a:
                        alias_set.add(_canonical_name(a))
                alias_set.add(_canonical_name(provided_key))
                alias_set.add(_canonical_name(os.path.splitext(os.path.basename(csv_path))[0]))
                alias_set.add(_canonical_name(canonical))
                alias_list = sorted([a for a in alias_set if a])

                meta = {
                    "path": csv_path,
                    "columns": columns,
                    "columns_normalized": columns_normalized,
                    "sample_rows": sample_rows,
                    "canonical": canonical,
                    "friendly": friendly,
                    "aliases": alias_list,
                }

                # Choose store key: prefer canonical if unused or same path; otherwise provided_key
                store_key = canonical if canonical not in self._store else provided_key

                # Conflict detection: if canonical exists but with different path, generate unique key
                existing = None
                for k, v in self._store.items():
                    try:
                        if _canonical_name(k) == _canonical_name(store_key):
                            existing = (k, v)
                            break
                    except Exception:
                        continue

                if existing:
                    # If same path, update; otherwise create a unique key to avoid overwriting
                    existing_key, existing_val = existing
                    if existing_val.get("path") == csv_path:
                        store_key = existing_key
                        self._store[store_key] = meta
                    else:
                        unique_store_key = f"{store_key}_{uuid4_hex()}"
                        self._store[unique_store_key] = meta
                        store_key = unique_store_key
                        logger.warning("SchemaStore: canonical conflict for '%s'. Stored under '%s' instead.", canonical, store_key)
                else:
                    self._store[store_key] = meta
                    # Also ensure canonical entry exists pointing to same meta (if not clashing)
                    try:
                        if canonical not in self._store:
                            self._store[canonical] = meta
                    except Exception:
                        pass

                self._save_store()
                logger.info("CSV schema stored for %s (key=%s canonical=%s)", provided_key, store_key, canonical)
                return store_key
            except CustomException:
                raise
            except Exception as e:
                logger.exception("Failed to add CSV: %s", csv_path)
                raise CustomException(e, sys)

    def update_schema(self, store_key: str, updates: Dict[str, Any]) -> bool:
        """
        Update fields for an existing schema entry. Returns True if updated.
        """
        with self._lock:
            try:
                if store_key not in self._store:
                    # try canonical/alias lookup
                    key = self._find_matching_key(store_key)
                    if not key:
                        return False
                    store_key = key
                item = self._store.get(store_key, {})
                item.update(updates)
                # normalize aliases if provided
                if "aliases" in updates and isinstance(updates.get("aliases"), list):
                    item["aliases"] = sorted(list({_canonical_name(a) for a in updates.get("aliases", []) if a}))
                # if columns updated, compute normalized columns too
                if "columns" in updates and isinstance(updates.get("columns"), (list, tuple)):
                    item["columns_normalized"] = [ _canonical_name(str(c)) for c in updates.get("columns", []) ]
                self._store[store_key] = item
                self._save_store()
                return True
            except Exception as e:
                logger.exception("Failed to update schema %s: %s", store_key, e)
                raise CustomException(e, sys)

    def remove_schema(self, store_key: str) -> bool:
        """
        Remove a schema entry by key/canonical/alias. Returns True if removed.
        """
        with self._lock:
            try:
                key = self._find_matching_key(store_key)
                if not key:
                    return False
                # remove the matching key only; do not automatically remove aliases that point to other keys
                try:
                    del self._store[key]
                except KeyError:
                    return False
                self._save_store()
                return True
            except Exception as e:
                logger.exception("Failed to remove schema %s: %s", store_key, e)
                raise CustomException(e, sys)

    # -------------------------
    # Lookup helpers (fuzzy)
    # -------------------------
    def _find_matching_key(self, csv_name: Optional[str]) -> Optional[str]:
        if not csv_name:
            return None
        # exact key
        if csv_name in self._store:
            return csv_name

        target = _canonical_name(csv_name)
        # exact canonical match or alias match
        for k, v in self._store.items():
            try:
                if _canonical_name(k) == target:
                    return k
                if v.get("canonical") and _canonical_name(v.get("canonical")) == target:
                    return k
                aliases = v.get("aliases") or []
                if any(_canonical_name(a) == target for a in aliases):
                    return k
            except Exception:
                continue

        # substring/contain checks
        for k, v in self._store.items():
            try:
                k_can = v.get("canonical") or _canonical_name(k)
                if target == k_can or target in k_can or k_can in target:
                    return k
            except Exception:
                continue

        # fuzzy matching via difflib
        try:
            import difflib

            candidates = []
            for k, v in self._store.items():
                try:
                    candidates.append(v.get("canonical") or _canonical_name(k))
                except Exception:
                    candidates.append(_canonical_name(k))
            match = difflib.get_close_matches(target, candidates, n=1, cutoff=0.6)
            if match:
                chosen = match[0]
                # find the store key that has that canonical
                for k, v in self._store.items():
                    if (v.get("canonical") and _canonical_name(v.get("canonical")) == _canonical_name(chosen)) or _canonical_name(k) == _canonical_name(chosen):
                        return k
        except Exception:
            pass

        return None

    # -------------------------
    # Query API
    # -------------------------
    def get_schema(self, csv_name: str) -> Optional[List[str]]:
        key = self._find_matching_key(csv_name)
        if key:
            return self._store.get(key, {}).get("columns")
        return None

    def get_sample_rows(self, csv_name: str) -> Optional[List[Dict[str, Any]]]:
        key = self._find_matching_key(csv_name)
        if key:
            return self._store.get(key, {}).get("sample_rows")
        return None

    def get_internal_key(self, csv_name: str) -> Optional[str]:
        return self._find_matching_key(csv_name)

    def find_canonical_for(self, name: str) -> Optional[str]:
        key = self._find_matching_key(name)
        if not key:
            return None
        rec = self._store.get(key)
        if not rec:
            return None
        return rec.get("canonical") or _canonical_name(key)

    def list_csvs(self) -> List[str]:
        return list(self._store.keys())

    def list_csvs_meta(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for k, v in self._store.items():
            try:
                out.append(
                    {
                        "key": k,
                        "canonical": v.get("canonical") or _canonical_name(k),
                        "friendly": v.get("friendly"),
                        "path": v.get("path"),
                        "columns": v.get("columns", []),
                        "columns_normalized": v.get("columns_normalized", []),
                        "sample_rows": v.get("sample_rows", []),
                        "aliases": v.get("aliases", []),
                    }
                )
            except Exception:
                out.append({"key": k})
        return out

    def clear(self) -> None:
        with self._lock:
            try:
                self._store = {}
                if os.path.exists(self.store_path):
                    os.remove(self.store_path)
                logger.info("Schema store cleared")
            except Exception as e:
                logger.exception("Failed to clear schema store")
                raise CustomException(e, sys)
