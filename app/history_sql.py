# app/history_sql.py
import os
import sys
import json
import sqlite3
import uuid
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("history_store")


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _safe_json_dumps(obj: Any) -> str:
    """Safely dump object to JSON string; fallback to str() if not serializable."""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception as e:
        logger.warning("Fallback json.dumps -> str() for object: %s", e)
        return json.dumps(str(obj), ensure_ascii=False)


class HistoryStore:
    """
    History storage with JSON or SQLite backend.
    """

    def __init__(self, backend: str = "json", path: Optional[str] = None):
        try:
            self.backend = (backend or "json").lower()
            default_base = getattr(config, "DATA_DIR", os.path.join(os.getcwd(), "data"))
            path = os.path.abspath(path) if path else None

            if self.backend == "json":
                if path and (os.path.isdir(path) or path.endswith(os.sep)):
                    base_dir = path
                    self.file_path = os.path.join(base_dir, "history.json")
                elif path:
                    base_dir = os.path.dirname(path) or "."
                    self.file_path = path
                else:
                    base_dir = os.path.abspath(default_base)
                    self.file_path = os.path.join(base_dir, "history.json")

                _ensure_dir(base_dir)
                if not os.path.exists(self.file_path):
                    with open(self.file_path, "w", encoding="utf-8") as f:
                        json.dump([], f)

            elif self.backend == "sqlite":
                if path and (os.path.isdir(path) or path.endswith(os.sep)):
                    base_dir = path
                    self.db_path = os.path.join(base_dir, "history.db")
                elif path:
                    base_dir = os.path.dirname(path) or "."
                    self.db_path = path
                else:
                    base_dir = os.path.abspath(default_base)
                    self.db_path = os.path.join(base_dir, "history.db")

                _ensure_dir(base_dir)
                self._init_sqlite()
            else:
                raise ValueError("Unsupported backend: choose 'json' or 'sqlite'")

            logger.info("HistoryStore initialized (backend=%s) at %s", self.backend, path or default_base)

        except Exception as e:
            logger.exception("Failed to initialize HistoryStore")
            raise CustomException(e, sys)

    # -------------------------
    # JSON backend
    # -------------------------
    def _read_json_file(self) -> List[Dict[str, Any]]:
        try:
            if not os.path.exists(getattr(self, "file_path", "")):
                return []
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f) or []
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.exception("Failed to read JSON history")
            raise CustomException(e, sys)

    def _write_json_file_atomic(self, items: List[Dict[str, Any]]) -> None:
        try:
            dirpath = os.path.dirname(self.file_path) or "."
            _ensure_dir(dirpath)
            fd, tmp_path = tempfile.mkstemp(dir=dirpath, prefix="history_", suffix=".tmp")
            os.close(fd)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            shutil.move(tmp_path, self.file_path)
        except Exception as e:
            logger.exception("Failed to write JSON atomically")
            raise CustomException(e, sys)

    # -------------------------
    # SQLite backend
    # -------------------------
    def _init_sqlite(self) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    query TEXT,
                    result_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.exception("Failed to initialize SQLite history DB")
            raise CustomException(e, sys)

    def _get_sqlite_conn(self):
        try:
            return sqlite3.connect(self.db_path, check_same_thread=False)
        except Exception as e:
            logger.exception("Failed to open SQLite connection")
            raise CustomException(e, sys)

    # -------------------------
    # Public API
    # -------------------------
    def add_entry(self, name: str, query: str, result: Any) -> Dict[str, Any]:
        try:
            entry_id = str(uuid.uuid4())
            created_at = datetime.utcnow().isoformat() + "Z"
            entry = {
                "id": entry_id,
                "name": name,
                "query": query,
                "result": result,
                "created_at": created_at,
                "updated_at": created_at,
            }

            if self.backend == "json":
                items = self._read_json_file()
                items.append(entry)
                self._write_json_file_atomic(items)
            else:
                conn = self._get_sqlite_conn()
                conn.execute(
                    "INSERT INTO history (id, name, query, result_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (entry_id, name, query, _safe_json_dumps(result), created_at, created_at)
                )
                conn.commit()
                conn.close()

            logger.info("History entry added: %s", entry_id)
            return entry
        except Exception as e:
            logger.exception("Failed to add history entry")
            raise CustomException(e, sys)

    def update_entry(self, entry_id: str, name: Optional[str] = None, query: Optional[str] = None, result: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        try:
            updated_at = datetime.utcnow().isoformat() + "Z"

            if self.backend == "json":
                items = self._read_json_file()
                for it in items:
                    if it.get("id") == entry_id:
                        if name is not None: it["name"] = name
                        if query is not None: it["query"] = query
                        if result is not None: it["result"] = result
                        it["updated_at"] = updated_at
                        self._write_json_file_atomic(items)
                        logger.info("Updated history entry (json): %s", entry_id)
                        return it
                return None
            else:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                updates, params = [], []
                if name is not None: updates.append("name=?"); params.append(name)
                if query is not None: updates.append("query=?"); params.append(query)
                if result is not None: updates.append("result_json=?"); params.append(_safe_json_dumps(result))
                updates.append("updated_at=?"); params.append(updated_at)
                if not updates: return None
                params.append(entry_id)
                cur.execute(f"UPDATE history SET {', '.join(updates)} WHERE id=?", tuple(params))
                if cur.rowcount == 0:
                    conn.close()
                    return None
                cur.execute("SELECT id, name, query, result_json, created_at, updated_at FROM history WHERE id=?", (entry_id,))
                row = cur.fetchone()
                conn.close()
                updated = {"id": row[0], "name": row[1], "query": row[2], "result": json.loads(row[3]) if row[3] else None, "created_at": row[4], "updated_at": row[5]}
                logger.info("Updated history entry (sqlite): %s", entry_id)
                return updated
        except Exception as e:
            logger.exception("Failed to update history entry")
            raise CustomException(e, sys)

    def list_entries(self, limit: Optional[int] = None, newest_first: bool = True) -> List[Dict[str, Any]]:
        try:
            if self.backend == "json":
                items = self._read_json_file()
                items_sorted = sorted(items, key=lambda x: x.get("created_at", ""), reverse=newest_first)
                return items_sorted[:limit] if limit else items_sorted
            else:
                conn = self._get_sqlite_conn()
                order = "DESC" if newest_first else "ASC"
                q = f"SELECT id, name, query, result_json, created_at, updated_at FROM history ORDER BY created_at {order}"
                if limit: q += f" LIMIT {int(limit)}"
                cur = conn.cursor()
                cur.execute(q)
                rows = cur.fetchall()
                conn.close()
                return [{"id": r[0], "name": r[1], "query": r[2], "result": json.loads(r[3]) if r[3] else None, "created_at": r[4], "updated_at": r[5]} for r in rows]
        except Exception as e:
            logger.exception("Failed to list history entries")
            raise CustomException(e, sys)

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        try:
            if self.backend == "json":
                items = self._read_json_file()
                return next((it for it in items if it.get("id") == entry_id), None)
            else:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                cur.execute("SELECT id, name, query, result_json, created_at, updated_at FROM history WHERE id=?", (entry_id,))
                row = cur.fetchone()
                conn.close()
                return {"id": row[0], "name": row[1], "query": row[2], "result": json.loads(row[3]) if row[3] else None, "created_at": row[4], "updated_at": row[5]} if row else None
        except Exception as e:
            logger.exception("Failed to get history entry")
            raise CustomException(e, sys)

    def delete_entry(self, entry_id: str) -> bool:
        try:
            if self.backend == "json":
                items = self._read_json_file()
                new_items = [it for it in items if it.get("id") != entry_id]
                if len(new_items) == len(items): return False
                self._write_json_file_atomic(new_items)
                return True
            else:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                cur.execute("DELETE FROM history WHERE id=?", (entry_id,))
                affected = cur.rowcount
                conn.commit()
                conn.close()
                return affected > 0
        except Exception as e:
            logger.exception("Failed to delete history entry")
            raise CustomException(e, sys)

    def clear(self) -> None:
        try:
            if self.backend == "json":
                self._write_json_file_atomic([])
            else:
                conn = self._get_sqlite_conn()
                conn.execute("DELETE FROM history")
                conn.commit()
                conn.close()
            logger.info("History cleared")
        except Exception as e:
            logger.exception("Failed to clear history")
            raise CustomException(e, sys)

    def export_json(self, export_path: Optional[str] = None) -> str:
        try:
            export_path = export_path or os.path.join(getattr(config, "DATA_DIR", "./data"), f"history_export_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
            _ensure_dir(os.path.dirname(export_path) or ".")
            items = self.list_entries(newest_first=False)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            logger.info("History exported to %s", export_path)
            return export_path
        except Exception as e:
            logger.exception("Failed to export history")
            raise CustomException(e, sys)

    def import_json(self, import_path: str, keep_ids: bool = True) -> int:
        try:
            if not os.path.exists(import_path):
                raise FileNotFoundError(import_path)
            with open(import_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Import JSON root must be a list")

            count = 0
            for it in data:
                entry_id = it.get("id") if keep_ids and it.get("id") else str(uuid.uuid4())
                created_at = it.get("created_at") if keep_ids else datetime.utcnow().isoformat() + "Z"
                updated_at = it.get("updated_at") if keep_ids else datetime.utcnow().isoformat() + "Z"

                entry = {
                    "id": entry_id,
                    "name": it.get("name", f"Query {entry_id}"),
                    "query": it.get("query", ""),
                    "result": it.get("result"),
                    "created_at": created_at,
                    "updated_at": updated_at,
                }

                if self.backend == "json":
                    items = self._read_json_file()
                    items.append(entry)
                    self._write_json_file_atomic(items)
                else:
                    conn = self._get_sqlite_conn()
                    conn.execute(
                        "INSERT OR REPLACE INTO history (id, name, query, result_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (entry_id, entry["name"], entry["query"], _safe_json_dumps(entry["result"]), created_at, updated_at)
                    )
                    conn.commit()
                    conn.close()
                count += 1

            logger.info("Imported %d history entries from %s", count, import_path)
            return count
        except Exception as e:
            logger.exception("Failed to import history")
            raise CustomException(e, sys)
