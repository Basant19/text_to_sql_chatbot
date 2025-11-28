# app/history_sql.py
import sys
import os
import json
import tempfile
import uuid
import time
import sqlite3
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from app.logger import get_logger
from app.exception import CustomException
import app.config as config_module

logger = get_logger("history_store")


def _make_json_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable forms.
    Special-case LangChain message objects when available, datetimes, bytes,
    sets, objects with to_dict(), dataclasses / __dict__ and finally fall back to str().
    """
    # 1) LangChain message objects (AIMessage/HumanMessage/SystemMessage)
    try:
        # lazy import: don't require langchain to be installed
        from langchain.schema import AIMessage, HumanMessage, SystemMessage  # type: ignore
        if isinstance(obj, (AIMessage, HumanMessage, SystemMessage)):
            return {
                "message_type": getattr(obj, "type", obj.__class__.__name__),
                "content": getattr(obj, "content", str(obj))
            }
    except Exception:
        # not installed or different API — ignore
        pass

    # Basic types
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # datetime/date
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # bytes
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except Exception:
            return str(obj)

    # set -> list
    if isinstance(obj, set):
        return list(obj)

    # objects with to_dict()
    try:
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            try:
                return obj.to_dict()
            except Exception:
                pass
    except Exception:
        pass

    # objects with __dict__
    try:
        if hasattr(obj, "__dict__"):
            d = {}
            for k, v in vars(obj).items():
                try:
                    d[k] = _make_json_serializable(v)
                except Exception:
                    d[k] = str(v)
            return d
    except Exception:
        pass

    # fallback
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def _write_json_file_atomic(path: str, items: Any) -> None:
    """
    Write `items` to `path` atomically; use _make_json_serializable as json.default
    to tolerate non-serializable values (LLM message objects, etc).
    """
    tmp_path = None
    try:
        dirpath = os.path.dirname(path) or "."
        os.makedirs(dirpath, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_history_", dir=dirpath, text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2, default=_make_json_serializable)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        # cleanup tmp file if present
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise CustomException(e, sys)


class HistoryStore:
    """
    Simple history persistence with pluggable backends: "json" (default) or "sqlite".

    Methods:
      - add_entry(name, query, result) -> entry dict (with id)
      - update_entry(entry_id, **fields) -> updated entry dict or None
      - delete_entry(entry_id)
      - export_json(path=None) -> path of exported JSON
      - list_entries() -> list of entries
    """

    def __init__(self, backend: str = "json", store_path: Optional[str] = None):
        try:
            self.backend = backend or "json"
            data_dir = getattr(config_module, "DATA_DIR", "./data")
            os.makedirs(data_dir, exist_ok=True)
            # JSON file path
            self.json_path = store_path or os.path.join(data_dir, "history.json")
            # In-memory fallback
            self._mem: List[Dict[str, Any]] = []
            # sqlite connection if requested
            self._conn: Optional[sqlite3.Connection] = None

            if self.backend == "sqlite":
                self._init_sqlite(os.path.join(data_dir, "history.db"))
            else:
                # ensure file exists
                if not os.path.exists(self.json_path):
                    _write_json_file_atomic(self.json_path, [])
                self._load_from_json()
            logger.info("HistoryStore initialized (backend=%s) at %s", self.backend, self.json_path)
        except Exception as e:
            logger.exception("HistoryStore initialization failed")
            raise CustomException(e, sys)

    # --------------------------
    # SQLite backend helpers
    # --------------------------
    def _init_sqlite(self, path: str) -> None:
        try:
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    query TEXT,
                    result_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.commit()
            self._conn = conn
        except Exception as e:
            logger.exception("Failed to initialize sqlite backend; falling back to json")
            self._conn = None
            self.backend = "json"
            # create json file
            if not os.path.exists(self.json_path):
                _write_json_file_atomic(self.json_path, [])
            self._load_from_json()

    # --------------------------
    # JSON persistence helpers
    # --------------------------
    def _load_from_json(self) -> None:
        try:
            if os.path.exists(self.json_path):
                with open(self.json_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    if isinstance(items, list):
                        self._mem = items
                    else:
                        # unexpected shape; reset
                        self._mem = []
            else:
                self._mem = []
        except Exception as e:
            logger.exception("Failed to load history JSON; using empty in-memory store")
            self._mem = []

    def _persist_json(self) -> None:
        try:
            _write_json_file_atomic(self.json_path, self._mem)
        except Exception as e:
            logger.exception("Failed to persist history JSON")
            # surface exception to caller as CustomException
            raise

    # --------------------------
    # CRUD operations
    # --------------------------
    def add_entry(self, name: str, query: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add an entry and persist. Returns the stored entry.
        """
        try:
            now = datetime.utcnow().isoformat() + "Z"
            entry_id = uuid.uuid4().hex
            entry = {
                "id": entry_id,
                "name": name,
                "query": query,
                "result": result,
                "created_at": now,
                "updated_at": now,
            }

            if self.backend == "sqlite" and self._conn:
                # store result as JSON string (use our serializable helper)
                result_json = json.dumps(entry["result"], default=_make_json_serializable, ensure_ascii=False)
                cur = self._conn.cursor()
                cur.execute(
                    "INSERT INTO history (id, name, query, result_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (entry_id, name, query, result_json, entry["created_at"], entry["updated_at"]),
                )
                self._conn.commit()
                return entry
            else:
                # JSON backend
                self._mem.append(entry)
                try:
                    self._persist_json()
                except Exception:
                    logger.exception("Persisting to JSON failed; entry remains in memory")
                return entry
        except Exception as e:
            logger.exception("Failed to add history entry")
            raise CustomException(e, sys)

    def update_entry(self, entry_id: str, **fields) -> Optional[Dict[str, Any]]:
        """
        Update an existing entry (name/query/result). Returns updated entry or None.
        """
        try:
            if self.backend == "sqlite" and self._conn:
                cur = self._conn.cursor()
                # load existing
                cur.execute("SELECT name, query, result_json, created_at FROM history WHERE id = ?", (entry_id,))
                row = cur.fetchone()
                if not row:
                    return None
                name, query, result_json, created_at = row
                result_obj = json.loads(result_json)
                # apply fields
                name = fields.get("name", name)
                query = fields.get("query", query)
                if "result" in fields:
                    result_obj = fields["result"]
                updated_at = datetime.utcnow().isoformat() + "Z"
                cur.execute(
                    "UPDATE history SET name=?, query=?, result_json=?, updated_at=? WHERE id=?",
                    (name, query, json.dumps(result_obj, default=_make_json_serializable, ensure_ascii=False), updated_at, entry_id),
                )
                self._conn.commit()
                return {
                    "id": entry_id,
                    "name": name,
                    "query": query,
                    "result": result_obj,
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            else:
                for e in self._mem:
                    if e.get("id") == entry_id:
                        for k, v in fields.items():
                            if k == "result":
                                e["result"] = v
                            elif k in ("name", "query"):
                                e[k] = v
                        e["updated_at"] = datetime.utcnow().isoformat() + "Z"
                        try:
                            self._persist_json()
                        except Exception:
                            logger.exception("Failed to persist history update to JSON")
                        return e
                return None
        except Exception as e:
            logger.exception("Failed to update history entry")
            raise CustomException(e, sys)

    def delete_entry(self, entry_id: str) -> bool:
        """
        Delete the entry; return True if deleted, False if not found.
        """
        try:
            if self.backend == "sqlite" and self._conn:
                cur = self._conn.cursor()
                cur.execute("DELETE FROM history WHERE id = ?", (entry_id,))
                self._conn.commit()
                return cur.rowcount > 0
            else:
                for i, e in enumerate(self._mem):
                    if e.get("id") == entry_id:
                        self._mem.pop(i)
                        try:
                            self._persist_json()
                        except Exception:
                            logger.exception("Failed to persist history deletion to JSON")
                        return True
                return False
        except Exception as e:
            logger.exception("Failed to delete history entry")
            raise CustomException(e, sys)

    def list_entries(self) -> List[Dict[str, Any]]:
        """
        Return all entries (JSON-deserialized objects).
        """
        try:
            if self.backend == "sqlite" and self._conn:
                cur = self._conn.cursor()
                cur.execute("SELECT id, name, query, result_json, created_at, updated_at FROM history ORDER BY created_at ASC")
                rows = cur.fetchall()
                result = []
                for r in rows:
                    id_, name, query, result_json, created_at, updated_at = r
                    try:
                        result_obj = json.loads(result_json)
                    except Exception:
                        result_obj = result_json
                    result.append({
                        "id": id_,
                        "name": name,
                        "query": query,
                        "result": result_obj,
                        "created_at": created_at,
                        "updated_at": updated_at,
                    })
                return result
            else:
                # ensure in-memory has been loaded
                if self._mem is None:
                    self._load_from_json()
                return list(self._mem)
        except Exception as e:
            logger.exception("Failed to list history entries")
            raise CustomException(e, sys)

    def export_json(self, path: Optional[str] = None) -> str:
        """
        Export history to a JSON file. If path is None, export to data/history_export_<ts>.json
        Returns the path written.
        """
        try:
            ts = int(time.time())
            data_dir = getattr(config_module, "DATA_DIR", "./data")
            os.makedirs(data_dir, exist_ok=True)
            out_path = path or os.path.join(data_dir, f"history_export_{ts}.json")
            entries = self.list_entries()
            _write_json_file_atomic(out_path, entries)
            return out_path
        except Exception as e:
            logger.exception("Failed to export history to JSON")
            raise CustomException(e, sys)

    # Backwards-compatibility alias: older callers may call add_entry_and_return(...)
    # Keep add_entry as the canonical method and alias the older name to it.
    add_entry_and_return = add_entry
