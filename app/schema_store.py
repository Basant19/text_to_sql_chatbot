import os
import json
import tempfile
import shutil
import threading
from typing import Dict, List, Optional, Any

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("schema_store")
_lock = threading.Lock()


def _default_store_path() -> str:
    """
    Default path for schema store JSON.
    Uses the same directory as DATABASE_PATH (config), fallback to ./data/schemas.json
    """
    try:
        db_path = getattr(config, "DATABASE_PATH", None)
        if db_path:
            base_dir = os.path.dirname(db_path) or "."
        else:
            base_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, "schemas.json")
    except Exception as e:
        logger.exception("Failed to determine default schema store path")
        raise CustomException(e, __import__("sys"))


class SchemaStore:
    """
    Simple JSON-backed schema store.
    Each schema is stored under its table_name key.
    """

    def __init__(self, store_path: Optional[str] = None):
        try:
            self.store_path = store_path or getattr(config, "SCHEMA_STORE_PATH", None) or _default_store_path()
            # ensure dir exists
            os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
            # initialize file if missing
            if not os.path.exists(self.store_path):
                with open(self.store_path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
            logger.info(f"SchemaStore initialized at {self.store_path}")
        except Exception as e:
            logger.exception("Failed to initialize SchemaStore")
            raise CustomException(e, __import__("sys"))

    def _read_all(self) -> Dict[str, Any]:
        with _lock:
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                # corrupted file — treat as empty and overwrite on next write
                logger.warning("Schema store JSON corrupted; treating as empty")
                return {}
            except Exception as e:
                logger.exception("Failed to read schema store")
                raise CustomException(e, __import__("sys"))

    def _write_all(self, data: Dict[str, Any]) -> None:
        with _lock:
            try:
                # atomic write: write to temp file then replace
                dirpath = os.path.dirname(self.store_path)
                fd, tmp_path = tempfile.mkstemp(dir=dirpath)
                with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
                    json.dump(data, tmpf, indent=2, ensure_ascii=False)
                    tmpf.flush()
                    os.fsync(tmpf.fileno())
                shutil.move(tmp_path, self.store_path)
            except Exception as e:
                logger.exception("Failed to write schema store atomically")
                raise CustomException(e, __import__("sys"))

    def add_schema(self, metadata: Dict[str, Any]) -> None:
        """
        Add or update a schema metadata dict.
        metadata must include 'table_name' key.
        """
        try:
            if "table_name" not in metadata:
                raise ValueError("metadata must include 'table_name'")

            data = self._read_all()
            data[metadata["table_name"]] = metadata
            self._write_all(data)
            logger.info(f"Added/updated schema: {metadata['table_name']}")
        except Exception as e:
            logger.exception("Failed to add schema")
            raise CustomException(e, __import__("sys"))

    def get_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Return metadata dict for the given table_name, or None if missing."""
        try:
            data = self._read_all()
            return data.get(table_name)
        except Exception as e:
            logger.exception("Failed to get schema")
            raise CustomException(e, __import__("sys"))

    def list_schemas(self) -> List[Dict[str, Any]]:
        """Return a list of all schema metadata dicts."""
        try:
            data = self._read_all()
            return list(data.values())
        except Exception as e:
            logger.exception("Failed to list schemas")
            raise CustomException(e, __import__("sys"))

    def remove_schema(self, table_name: str) -> bool:
        """
        Remove schema for table_name.
        Returns True if removed, False if it did not exist.
        """
        try:
            data = self._read_all()
            if table_name in data:
                del data[table_name]
                self._write_all(data)
                logger.info(f"Removed schema: {table_name}")
                return True
            logger.info(f"Schema not found for removal: {table_name}")
            return False
        except Exception as e:
            logger.exception("Failed to remove schema")
            raise CustomException(e, __import__("sys"))

    def clear(self) -> None:
        """Remove all schemas."""
        try:
            self._write_all({})
            logger.info("Cleared all schemas from store")
        except Exception as e:
            logger.exception("Failed to clear schema store")
            raise CustomException(e, __import__("sys"))
