# schema_store.py
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
    return no_ext.lower()


class SchemaStore:
    """
    Manages CSV metadata: schemas, sample rows, and column info.
    Stores metadata persistently in JSON files.
    Designed for compatibility with CSVLoader and vectorization pipelines.
    """

    def __init__(self, store_path: Optional[str] = None, sample_limit: int = 5):
        try:
            data_dir = getattr(config, "DATA_DIR", "./data")
            self.store_path = store_path or os.path.join(data_dir, "schema_store.json")
            _ensure_dir(os.path.dirname(self.store_path))
            self.sample_limit = sample_limit
            self._store: Dict[str, Dict[str, Any]] = {}
            self._load_store()
            logger.info(f"SchemaStore initialized at {self.store_path}")
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
                    self._store = json.load(f)
            else:
                self._store = {}
        except Exception as e:
            logger.exception("Failed to load schema store")
            raise CustomException(e, sys)

    def _save_store(self) -> None:
        """Persist schema metadata to JSON file."""
        try:
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(self._store, f, indent=2, ensure_ascii=False)
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

            # store under the provided name (often filename with suffix)
            self._store[csv_name] = {
                "path": csv_path,
                "columns": columns,
                "sample_rows": samples,
            }

            # also save an entry under the canonical (no-extension) form if not already present
            canonical = _canonical_name(csv_name)
            if canonical not in { _canonical_name(k) for k in self._store.keys() if k != csv_name }:
                # ensure canonical key exists (store path and columns still point to the same file)
                # but avoid overwriting a different explicit key
                if canonical not in self._store:
                    self._store[canonical] = self._store[csv_name]

            self._save_store()
            logger.info(f"CSV schema stored for {csv_name}")
        except Exception as e:
            logger.exception(f"Failed to add CSV: {csv_path}")
            raise CustomException(e, sys)

    def _find_matching_key(self, csv_name: str) -> Optional[str]:
        """
        Try to locate a store key that corresponds to csv_name.
        Strategies:
          1) exact key match
          2) canonicalized match (strip path + extension, lower)
          3) prefix match (key startswith canonical or vice-versa)
          4) substring match
        Returns the matching key or None.
        """
        if not csv_name:
            return None

        # exact
        if csv_name in self._store:
            return csv_name

        target = _canonical_name(csv_name)

        # canonical exact
        for k in self._store.keys():
            if _canonical_name(k) == target:
                return k

        # prefix / contains matches (try to be permissive but deterministic)
        for k in self._store.keys():
            k_can = _canonical_name(k)
            if k_can.startswith(target) or target.startswith(k_can) or target in k_can or k_can in target:
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

    def list_csvs(self) -> List[str]:
        """Return a list of all CSVs stored."""
        return list(self._store.keys())

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
