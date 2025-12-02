# app/schema_store.py
import os
import sys
import json
from typing import List, Dict, Any, Optional
import csv
import uuid

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("schema_store")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _canonical_name(name: str) -> str:
    base = os.path.basename(name)
    no_ext = os.path.splitext(base)[0]
    return no_ext.lower().strip()


def uuid4_hex() -> str:
    return uuid.uuid4().hex[:8]


class SchemaStore:
    def __init__(self, store_path: Optional[str] = None, sample_limit: int = 5):
        try:
            data_dir = getattr(config, "DATA_DIR", "./data")
            self.store_path = store_path or os.path.join(data_dir, "schema_store.json")
            _ensure_dir(os.path.dirname(self.store_path))
            self.sample_limit = sample_limit
            self._store: Dict[str, Dict[str, Any]] = {}
            self._load_store()
            logger.info("SchemaStore initialized at %s", self.store_path)
        except Exception as e:
            logger.exception("Failed to initialize SchemaStore")
            raise CustomException(e, sys)

    def _load_store(self) -> None:
        try:
            if os.path.exists(self.store_path):
                with open(self.store_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    if isinstance(items, dict):
                        normalized: Dict[str, Dict[str, Any]] = {}
                        for k, v in items.items():
                            if not isinstance(v, dict):
                                continue
                            # if already new shape
                            if "canonical" in v or "aliases" in v:
                                normalized[k] = v
                            else:
                                canonical = _canonical_name(k)
                                friendly = canonical
                                aliases = [k, canonical]
                                rec = {
                                    "path": v.get("path"),
                                    "columns": v.get("columns", []),
                                    "sample_rows": v.get("sample_rows", []),
                                    "canonical": canonical,
                                    "friendly": friendly,
                                    "aliases": aliases,
                                }
                                normalized[k] = rec
                        self._store = normalized
                    else:
                        logger.warning("SchemaStore file has unexpected shape; resetting store.")
                        self._store = {}
            else:
                self._store = {}
        except Exception as e:
            logger.exception("Failed to load schema store")
            raise CustomException(e, sys)

    def _save_store(self) -> None:
        try:
            dirpath = os.path.dirname(self.store_path) or "."
            _ensure_dir(dirpath)
            tmp = f"{self.store_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._store, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.store_path)
        except Exception as e:
            logger.exception("Failed to save schema store")
            raise CustomException(e, sys)

    def add_csv(self, csv_path: str, csv_name: Optional[str] = None, aliases: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a CSV schema to the store.

        Accepts:
          - csv_path: path to file
          - csv_name: optional friendly/key name
          - aliases: optional aliases list
          - metadata: optional metadata dict (as returned by CSVLoader.load_and_extract)
            If metadata is provided, it is used directly to populate columns/aliases/sample_rows.
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            provided_key = csv_name or os.path.basename(csv_path)

            if metadata and isinstance(metadata, dict):
                # Prefer metadata fields when provided (compat with CSVLoader output)
                columns = metadata.get("columns", [])
                sample_rows = metadata.get("sample_rows", [])[: self.sample_limit]
                canonical = metadata.get("canonical_name") or metadata.get("canonical") or _canonical_name(provided_key)
                friendly = metadata.get("table_name") or canonical
                aliases_list = metadata.get("aliases") or []
            else:
                # read the CSV to get header & samples
                columns: List[str] = []
                sample_rows: List[Dict[str, Any]] = []
                with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    columns = reader.fieldnames or []
                    for i, row in enumerate(reader):
                        if i >= self.sample_limit:
                            break
                        sample_rows.append(row)
                canonical = _canonical_name(provided_key)
                friendly = csv_name or canonical
                aliases_list = aliases or []

            # build alias set (normalize)
            alias_set = set()
            for a in aliases_list:
                if a:
                    alias_set.add(_canonical_name(a))
            alias_set.add(_canonical_name(provided_key))
            alias_set.add(_canonical_name(os.path.splitext(os.path.basename(csv_path))[0]))
            alias_set.add(canonical)
            alias_list = [a for a in alias_set if a]

            meta = {
                "path": csv_path,
                "columns": columns,
                "sample_rows": sample_rows,
                "canonical": canonical,
                "friendly": friendly,
                "aliases": alias_list,
            }

            store_key = provided_key
            if _canonical_name(store_key) == canonical:
                store_key = canonical

            # conflict detection: if canonical already exists but different path, avoid overwrite
            conflict_key = None
            for k, v in list(self._store.items()):
                try:
                    if _canonical_name(k) == canonical and v.get("path") != csv_path:
                        conflict_key = k
                        break
                except Exception:
                    continue

            if conflict_key:
                unique_store_key = f"{store_key}_{uuid4_hex()}"
                self._store[unique_store_key] = meta
                logger.warning("SchemaStore: canonical conflict detected for '%s'. Stored under '%s' instead.", canonical, unique_store_key)
            else:
                self._store[store_key] = meta
                # ensure canonical key exists and points to same meta
                try:
                    existing = self._store.get(canonical)
                    if not existing:
                        self._store[canonical] = meta
                    else:
                        # prefer identical path; otherwise preserve existing
                        if existing.get("path") == csv_path:
                            self._store[canonical] = meta
                except Exception:
                    pass

            self._save_store()
            logger.info("CSV schema stored for %s (key=%s canonical=%s)", provided_key, store_key, canonical)
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Failed to add CSV: %s", csv_path)
            raise CustomException(e, sys)

    def _find_matching_key(self, csv_name: str) -> Optional[str]:
        if not csv_name:
            return None
        if csv_name in self._store:
            return csv_name

        target = _canonical_name(csv_name)

        for k, v in self._store.items():
            if v.get("canonical") == target or _canonical_name(k) == target:
                return k

        for k, v in self._store.items():
            aliases = v.get("aliases") or []
            if any(_canonical_name(a) == target for a in aliases):
                return k

        for k, v in self._store.items():
            k_can = v.get("canonical") or _canonical_name(k)
            if k_can.startswith(target) or target.startswith(k_can) or target in k_can or k_can in target:
                return k

        for k in self._store.keys():
            if target in _canonical_name(k) or _canonical_name(k) in target:
                return k

        return None

    def get_schema(self, csv_name: str) -> Optional[List[str]]:
        match = self._find_matching_key(csv_name)
        if match:
            return self._store.get(match, {}).get("columns")
        return None

    def get_sample_rows(self, csv_name: str) -> Optional[List[Dict[str, Any]]]:
        match = self._find_matching_key(csv_name)
        if match:
            return self._store.get(match, {}).get("sample_rows")
        return None

    def get_internal_key(self, csv_name: str) -> Optional[str]:
        return self._find_matching_key(csv_name)

    def find_canonical_for(self, name: str) -> Optional[str]:
        if not name:
            return None
        match = self._find_matching_key(name)
        if not match:
            return None
        rec = self._store.get(match)
        if not rec:
            return None
        return rec.get("canonical") or _canonical_name(match)

    def list_csvs(self) -> List[str]:
        return list(self._store.keys())

    def list_csvs_meta(self) -> List[Dict[str, Any]]:
        out = []
        for k, v in self._store.items():
            try:
                out.append({
                    "key": k,
                    "canonical": v.get("canonical") or _canonical_name(k),
                    "friendly": v.get("friendly"),
                    "path": v.get("path"),
                    "columns": v.get("columns", []),
                    "sample_rows": v.get("sample_rows", []),
                    "aliases": v.get("aliases", []),
                })
            except Exception:
                out.append({"key": k})
        return out

    def clear(self) -> None:
        try:
            self._store = {}
            if os.path.exists(self.store_path):
                os.remove(self.store_path)
            logger.info("Schema store cleared")
        except Exception as e:
            logger.exception("Failed to clear schema store")
            raise CustomException(e, sys)
