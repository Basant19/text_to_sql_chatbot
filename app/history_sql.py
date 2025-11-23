# app/history_sql.py
import os
import sys
import json
import sqlite3
import uuid
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("history_store")


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _safe_json_dumps(obj: Any) -> str:
    """
    Safely dump object to JSON string. Non-serializable objects are converted with str().
    """
    try:
        return json.dumps(obj, ensure_ascii=False, indent=None, default=str)
    except Exception as e:
        logger.warning("json.dumps fallback to str() for object: %s", e)
        return json.dumps(str(obj), ensure_ascii=False)


class HistoryStore:
    """
    Chat / query history storage with two backends:
      - "json"   : file-based storage (default) using JSON file
      - "sqlite" : sqlite3 DB storage (file-based)
    Each entry shape (in JSON backend) is:
        {
          "id": "<uuid>",
          "name": "Query 1",
          "query": "Natural language question",
          "result": {...},           # JSON-serializable (fallback to str)
          "created_at": "2025-11-23T12:34:56Z",
          "updated_at": "2025-11-23T12:35:00Z"
        }

    SQLite schema:
      CREATE TABLE IF NOT EXISTS history (
        id TEXT PRIMARY KEY,
        name TEXT,
        query TEXT,
        result_json TEXT,
        created_at TEXT,
        updated_at TEXT
      );
    """

    def __init__(self, backend: str = "json", path: Optional[str] = None):
        """
        backend: "json" or "sqlite"
        path: optional path to store DB/JSON. If omitted, uses config.DATA_DIR or ./data
        """
        try:
            self.backend = backend.lower()
            base_dir = path or getattr(config, "DATA_DIR", os.path.join(os.getcwd(), "data"))
            _ensure_dir(base_dir)
            if self.backend == "json":
                self.file_path = path or getattr(config, "HISTORY_PATH", os.path.join(base_dir, "history.json"))
                # create file if missing
                if not os.path.exists(self.file_path):
                    with open(self.file_path, "w", encoding="utf-8") as f:
                        json.dump([], f)
            elif self.backend == "sqlite":
                self.db_path = path or getattr(config, "HISTORY_DB_PATH", os.path.join(base_dir, "history.db"))
                _ensure_dir(os.path.dirname(self.db_path))
                self._init_sqlite()
            else:
                raise ValueError("unsupported backend: choose 'json' or 'sqlite'")
        except Exception as e:
            logger.exception("Failed to initialize HistoryStore")
            raise CustomException(e, sys)

    # -------------------------
    # JSON backend operations
    # -------------------------
    def _read_json_file(self) -> List[Dict[str, Any]]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f) or []
                if not isinstance(data, list):
                    logger.warning("history.json root not list; resetting to empty list")
                    data = []
                return data
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.exception("Failed to read history JSON")
            raise CustomException(e, sys)

    def _write_json_file_atomic(self, items: List[Dict[str, Any]]) -> None:
        """
        Write JSON atomically (write to temp, move into place).
        """
        try:
            dirpath = os.path.dirname(self.file_path) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dirpath, prefix="history_", suffix=".tmp")
            os.close(fd)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            shutil.move(tmp_path, self.file_path)
        except Exception as e:
            logger.exception("Failed to write history JSON atomically")
            raise CustomException(e, sys)

    # -------------------------
    # SQLite backend operations
    # -------------------------
    def _init_sqlite(self) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
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
            conn.close()
        except Exception as e:
            logger.exception("Failed to initialize sqlite history DB")
            raise CustomException(e, sys)

    def _get_sqlite_conn(self):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            return conn
        except Exception as e:
            logger.exception("Failed to open sqlite connection")
            raise CustomException(e, sys)

    # -------------------------
    # Public API
    # -------------------------
    def add_entry(self, name: str, query: str, result: Any) -> Dict[str, Any]:
        """
        Add a new history entry and return the stored entry dict.
        """
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
                # sqlite
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO history (id, name, query, result_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (entry_id, name, query, _safe_json_dumps(result), created_at, created_at),
                )
                conn.commit()
                conn.close()

            logger.info("History entry added: %s", entry_id)
            return entry
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Failed to add history entry")
            raise CustomException(e, sys)

    def update_entry(self, entry_id: str, name: Optional[str] = None, query: Optional[str] = None, result: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Update an existing entry by id. Returns the updated entry dict or None if not found.
        Only provided fields will be updated. updated_at is set to now.
        """
        try:
            updated_at = datetime.utcnow().isoformat() + "Z"
            if self.backend == "json":
                items = self._read_json_file()
                found = False
                for it in items:
                    if it.get("id") == entry_id:
                        if name is not None:
                            it["name"] = name
                        if query is not None:
                            it["query"] = query
                        if result is not None:
                            it["result"] = result
                        it["updated_at"] = updated_at
                        found = True
                        updated = it
                        break
                if not found:
                    return None
                self._write_json_file_atomic(items)
                logger.info("Updated history entry (json): %s", entry_id)
                return updated
            else:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                # Build dynamic update
                updates = []
                params = []
                if name is not None:
                    updates.append("name = ?")
                    params.append(name)
                if query is not None:
                    updates.append("query = ?")
                    params.append(query)
                if result is not None:
                    updates.append("result_json = ?")
                    params.append(_safe_json_dumps(result))
                if not updates:
                    # nothing to update, but still touch updated_at
                    updates.append("updated_at = ?")
                    params.append(updated_at)
                else:
                    updates.append("updated_at = ?")
                    params.append(updated_at)
                params.append(entry_id)
                sql = f"UPDATE history SET {', '.join(updates)} WHERE id = ?"
                cur.execute(sql, tuple(params))
                if cur.rowcount == 0:
                    conn.close()
                    return None
                conn.commit()
                # fetch and return
                cur.execute("SELECT id, name, query, result_json, created_at, updated_at FROM history WHERE id = ?", (entry_id,))
                row = cur.fetchone()
                conn.close()
                updated = {
                    "id": row[0],
                    "name": row[1],
                    "query": row[2],
                    "result": json.loads(row[3]) if row[3] else None,
                    "created_at": row[4],
                    "updated_at": row[5],
                }
                logger.info("Updated history entry (sqlite): %s", entry_id)
                return updated
        except Exception as e:
            logger.exception("Failed to update history entry")
            raise CustomException(e, sys)

    def list_entries(self, limit: Optional[int] = None, newest_first: bool = True) -> List[Dict[str, Any]]:
        """
        Return list of entries (optionally limited). newest_first sorts by created_at descending.
        """
        try:
            if self.backend == "json":
                items = self._read_json_file()
                items_sorted = sorted(items, key=lambda x: x.get("created_at", ""), reverse=newest_first)
                if limit:
                    items_sorted = items_sorted[:limit]
                return items_sorted
            else:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                order = "DESC" if newest_first else "ASC"
                q = "SELECT id, name, query, result_json, created_at, updated_at FROM history ORDER BY created_at " + order
                if limit:
                    q += f" LIMIT {int(limit)}"
                cur.execute(q)
                rows = cur.fetchall()
                conn.close()
                out = []
                for r in rows:
                    out.append({
                        "id": r[0],
                        "name": r[1],
                        "query": r[2],
                        "result": json.loads(r[3]) if r[3] else None,
                        "created_at": r[4],
                        "updated_at": r[5] if len(r) > 5 else None,
                    })
                return out
        except Exception as e:
            logger.exception("Failed to list history entries")
            raise CustomException(e, sys)

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Return single entry by id or None if not found.
        """
        try:
            if self.backend == "json":
                items = self._read_json_file()
                for it in items:
                    if it.get("id") == entry_id:
                        return it
                return None
            else:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                cur.execute("SELECT id, name, query, result_json, created_at, updated_at FROM history WHERE id = ?", (entry_id,))
                row = cur.fetchone()
                conn.close()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "name": row[1],
                    "query": row[2],
                    "result": json.loads(row[3]) if row[3] else None,
                    "created_at": row[4],
                    "updated_at": row[5] if len(row) > 5 else None,
                }
        except Exception as e:
            logger.exception("Failed to get history entry")
            raise CustomException(e, sys)

    def delete_entry(self, entry_id: str) -> bool:
        """
        Delete entry by id. Returns True if removed, False if not found.
        """
        try:
            if self.backend == "json":
                items = self._read_json_file()
                initial_len = len(items)
                items = [it for it in items if it.get("id") != entry_id]
                if len(items) == initial_len:
                    return False
                self._write_json_file_atomic(items)
                return True
            else:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                cur.execute("DELETE FROM history WHERE id = ?", (entry_id,))
                affected = cur.rowcount
                conn.commit()
                conn.close()
                return affected > 0
        except Exception as e:
            logger.exception("Failed to delete history entry")
            raise CustomException(e, sys)

    def clear(self) -> None:
        """
        Remove all history entries.
        """
        try:
            if self.backend == "json":
                self._write_json_file_atomic([])
            else:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                cur.execute("DELETE FROM history")
                conn.commit()
                conn.close()
            logger.info("History cleared")
        except Exception as e:
            logger.exception("Failed to clear history")
            raise CustomException(e, sys)

    def export_json(self, export_path: Optional[str] = None) -> str:
        """
        Export current history to a JSON file and return path.
        """
        try:
            export_path = export_path or os.path.join(getattr(config, "DATA_DIR", "./data"), f"history_export_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
            _ensure_dir(os.path.dirname(export_path))
            items = self.list_entries(newest_first=False)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            logger.info("History exported to %s", export_path)
            return export_path
        except Exception as e:
            logger.exception("Failed to export history")
            raise CustomException(e, sys)

    def import_json(self, import_path: str, keep_ids: bool = True) -> int:
        """
        Import a JSON file of history entries. Returns number of entries imported.
        If keep_ids=False, new UUIDs will be generated.
        """
        try:
            if not os.path.exists(import_path):
                raise FileNotFoundError(import_path)
            with open(import_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("import JSON root must be a list of entries")

            count = 0
            for it in data:
                entry_id = it.get("id") if keep_ids and it.get("id") else str(uuid.uuid4())
                entry = {
                    "id": entry_id,
                    "name": it.get("name", f"Query {entry_id}"),
                    "query": it.get("query", ""),
                    "result": it.get("result", None),
                    "created_at": it.get("created_at", datetime.utcnow().isoformat() + "Z"),
                    "updated_at": it.get("updated_at", datetime.utcnow().isoformat() + "Z"),
                }
                # write depending on backend
                if self.backend == "json":
                    items = self._read_json_file()
                    items.append(entry)
                    self._write_json_file_atomic(items)
                else:
                    conn = self._get_sqlite_conn()
                    cur = conn.cursor()
                    cur.execute(
                        "INSERT OR REPLACE INTO history (id, name, query, result_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (entry_id, entry["name"], entry["query"], _safe_json_dumps(entry["result"]), entry["created_at"], entry["updated_at"]),
                    )
                    conn.commit()
                    conn.close()
                count += 1
            logger.info("Imported %d history entries from %s", count, import_path)
            return count
        except Exception as e:
            logger.exception("Failed to import history")
            raise CustomException(e, sys)
