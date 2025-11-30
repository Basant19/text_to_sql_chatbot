#D:\text_to_sql_bot\app\schema_store.py
import os
import sys
import json
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("schema_store")


def _ensure_dir(path: str) -> None:
    """Ensure the given directory exists."""
    os.makedirs(path, exist_ok=True)


def _canonical_name(name: str) -> str:
    """
    Produce a normalized canonical form for comparing CSV/table names.
    - strip directory components
    - strip file extension
    - lower-case
    """
    base = os.path.basename(name)
    no_ext = os.path.splitext(base)[0]
    return no_ext.lower().strip()


class SchemaStore:
    """
    Manages CSV metadata: schemas, sample rows, and column info.
    Stores metadata persistently in JSON files.

    Backwards-compatible public API:
      - add_csv(csv_path, csv_name=None)
      - get_schema(csv_name) -> Optional[List[str]]
      - get_sample_rows(csv_name) -> Optional[List[Dict[str, Any]]]
      - list_csvs() -> List[str]  (list of store keys)
      - clear()

    Added utilities:
      - get_internal_key(csv_name) -> Optional[str]
      - list_csvs_meta() -> List[Dict[str, Any]] (useful for UI)
    """

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

    # ---------------------------
    # Persistence
    # ---------------------------
    def _load_store(self) -> None:
        """Load schema metadata from JSON file if exists."""
        try:
            if os.path.exists(self.store_path):
                with open(self.store_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    # Basic validation: expect dict
                    if isinstance(items, dict):
                        self._store = items
                    else:
                        logger.warning("SchemaStore file has unexpected shape; resetting store.")
                        self._store = {}
            else:
                self._store = {}
        except Exception as e:
            logger.exception("Failed to load schema store")
            raise CustomException(e, sys)

    def _save_store(self) -> None:
        """Persist schema metadata to JSON file (atomic replace)."""
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

    # ---------------------------
    # Public API
    # ---------------------------
    def add_csv(self, csv_path: str, csv_name: Optional[str] = None) -> None:
        """
        Read CSV file, store column names and sample rows.
        Updates internal store and persists to disk.

        csv_name: the key under which to save this CSV's metadata. If None, we use the filename.
        In addition to saving under csv_name, we attempt to save a canonical key
        (filename without extension, lower-cased) unless it would conflict with an existing
        canonical mapping that points to a different path.
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            import csv

            csv_name = csv_name or os.path.basename(csv_path)
            columns: List[str] = []
            samples: List[Dict[str, Any]] = []

            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                for i, row in enumerate(reader):
                    if i >= self.sample_limit:
                        break
                    samples.append(row)

            meta = {
                "path": csv_path,
                "columns": columns,
                "sample_rows": samples,
            }

            # Store under provided key (explicit)
            self._store[csv_name] = meta

            # Attempt to create canonical alias
            canonical = _canonical_name(csv_name)
            # If canonical equals the literal key we just added, it's already present.
            if canonical != csv_name:
                existing_key_for_canonical = None
                # check if there already exists a key whose canonical form equals this canonical
                for k, v in list(self._store.items()):
                    if _canonical_name(k) == canonical and k != csv_name:
                        existing_key_for_canonical = k
                        break

                if existing_key_for_canonical is None:
                    # no existing canonical mapping -> create alias
                    if canonical not in self._store:
                        self._store[canonical] = meta
                        logger.debug("SchemaStore: created canonical alias '%s' -> %s", canonical, csv_name)
                else:
                    # existing key present; if it points to same path, ok; otherwise do not overwrite
                    existing_path = self._store.get(existing_key_for_canonical, {}).get("path")
                    if existing_path and os.path.samefile(existing_path, csv_path) if os.path.exists(existing_path) and os.path.exists(csv_path) else existing_path == csv_path:
                        # same file under different names: we can safely create canonical pointing to same meta
                        if canonical not in self._store:
                            self._store[canonical] = meta
                    else:
                        # conflict: different files share same canonical name - avoid overwrite
                        logger.warning(
                            "SchemaStore canonical name conflict for '%s' (existing key '%s' -> %s, new path=%s). "
                            "Skipping canonical alias creation to avoid overwrite.",
                            canonical, existing_key_for_canonical, existing_path, csv_path
                        )

            # persist
            self._save_store()
            logger.info("CSV schema stored for %s (canonical=%s)", csv_name, canonical)
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Failed to add CSV: %s", csv_path)
            raise CustomException(e, sys)

    def _find_matching_key(self, csv_name: str) -> Optional[str]:
        """
        Try to locate a store key that corresponds to csv_name.
        Strategies (in order):
          1) exact key match
          2) canonicalized exact match (strip path + extension, lower)
          3) prefix / contains matches (deterministic)
          4) substring match

        Returns the *store key* if found, otherwise None.
        """
        if not csv_name:
            return None

        # 1) exact key
        if csv_name in self._store:
            return csv_name

        target = _canonical_name(csv_name)

        # 2) canonical exact
        for k in self._store.keys():
            if _canonical_name(k) == target:
                return k

        # 3) prefix / contains (deterministic pass over keys)
        for k in self._store.keys():
            k_can = _canonical_name(k)
            # prefer keys where canonical startswith target (e.g. googleplaystore_v2 vs googleplaystore)
            if k_can.startswith(target) or target.startswith(k_can) or target in k_can or k_can in target:
                return k

        # 4) fallback substring match
        for k in self._store.keys():
            if target in _canonical_name(k) or _canonical_name(k) in target:
                return k

        return None

    def get_schema(self, csv_name: str) -> Optional[List[str]]:
        """Return column names for a given CSV/table name. Tries tolerant matching."""
        match = self._find_matching_key(csv_name)
        if match:
            return self._store.get(match, {}).get("columns")
        return None

    def get_sample_rows(self, csv_name: str) -> Optional[List[Dict[str, Any]]]:
        """Return sample rows for a given CSV. Tries tolerant matching."""
        match = self._find_matching_key(csv_name)
        if match:
            return self._store.get(match, {}).get("sample_rows")
        return None

    def get_internal_key(self, csv_name: str) -> Optional[str]:
        """
        Return the actual internal store key that matches csv_name (or None).
        Useful for debugging/inspecting which key was selected by tolerant matching.
        """
        return self._find_matching_key(csv_name)

    def list_csvs(self) -> List[str]:
        """Return a list of all CSV store keys."""
        return list(self._store.keys())

    def list_csvs_meta(self) -> List[Dict[str, Any]]:
        """
        Return list of metadata entries for UI consumption.
        Each element is: {"key": key, "canonical": canonical_name, "path": path, "columns": [...]}.
        """
        out = []
        for k, v in self._store.items():
            try:
                out.append({
                    "key": k,
                    "canonical": _canonical_name(k),
                    "path": v.get("path"),
                    "columns": v.get("columns", []),
                    "sample_rows": v.get("sample_rows", []),
                })
            except Exception:
                out.append({"key": k})
        return out

    def clear(self) -> None:
        """Clear in-memory store and remove persisted JSON file."""
        try:
            self._store = {}
            if os.path.exists(self.store_path):
                os.remove(self.store_path)
            logger.info("Schema store cleared")
        except Exception as e:
            logger.exception("Failed to clear schema store")
            raise CustomException(e, sys)
