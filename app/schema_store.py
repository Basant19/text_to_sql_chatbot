# app/schema_store.py
import os
import sys
import csv
import json
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("schema_store")


def _ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


class SchemaStore:
    """
    Manages CSV metadata: schemas, sample rows, and column info.
    Stores metadata in JSON files for persistent access.
    Compatible with CSVLoader and vectorization pipelines.
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
        """Load schema metadata from JSON file."""
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
        """Save schema metadata to JSON file."""
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
        Read CSV file, store columns and sample rows.
        Updates internal store and persists to disk.
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

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

            self._store[csv_name] = {
                "path": csv_path,
                "columns": columns,
                "sample_rows": samples,
            }
            self._save_store()
            logger.info(f"CSV schema stored for {csv_name}")
        except Exception as e:
            logger.exception(f"Failed to add CSV: {csv_path}")
            raise CustomException(e, sys)

    def get_schema(self, csv_name: str) -> Optional[List[str]]:
        """Return list of column names for a CSV."""
        return self._store.get(csv_name, {}).get("columns")

    def get_sample_rows(self, csv_name: str) -> Optional[List[Dict[str, Any]]]:
        """Return sample rows for a CSV."""
        return self._store.get(csv_name, {}).get("sample_rows")

    def list_csvs(self) -> List[str]:
        """List all CSVs currently stored."""
        return list(self._store.keys())

    def clear(self) -> None:
        """Clear store in memory and remove persisted JSON."""
        try:
            self._store = {}
            if os.path.exists(self.store_path):
                os.remove(self.store_path)
            logger.info("Schema store cleared")
        except Exception as e:
            logger.exception("Failed to clear schema store")
            raise CustomException(e, sys)
