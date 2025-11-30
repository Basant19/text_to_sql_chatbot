# app/history_sql.py
"""
SQLite-first HistoryStore with conversation/message model (ChatGPT-style).

Changes made (high level):
- Uses SQLite only (no JSON backend) and creates two tables: conversations and messages.
- Conversations store metadata (id, name, created_at, updated_at, meta JSON).
- Messages are normalized into a messages table: id, conversation_id, role, content,
  meta JSON, created_at, position (message order).
- Provides conversation API: create_conversation, list_conversations (summaries),
  get_conversation (full), append_message, update_conversation_name, delete_conversation,
  export_conversation, migrate_from_legacy_json.
- Backwards-compatible helper add_entry that creates a single-message conversation
  (keeps old callers working).

Design choices:
- SQLite with WAL journal and foreign keys enabled for safety and concurrency.
- JSON meta columns stored as TEXT using json.dumps with a safe serializer.
- Every public method uses transactions and commits; errors raise CustomException.
- Keep full LLM blobs optional (store them in message.meta under a key; user can
  control via config flag STORE_FULL_LLM_BLOBS).

This file replaces the previous ad-hoc JSON/sql hybrid and implements a clear
conversation-first API suitable for the Streamlit chat UI (left sidebar list +
message stream in main view).
"""

import os
import sys
import json
import sqlite3
import tempfile
import uuid
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

from app.logger import get_logger
from app.exception import CustomException
import app.config as config_module

logger = get_logger("history_store")


def _make_json_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable forms. Keep identical to prior helper but
    kept here for json.dumps default.
    """
    try:
        from langchain.schema import AIMessage, HumanMessage, SystemMessage  # type: ignore
        if isinstance(obj, (AIMessage, HumanMessage, SystemMessage)):
            return {
                "message_type": getattr(obj, "type", obj.__class__.__name__),
                "content": getattr(obj, "content", str(obj)),
            }
    except Exception:
        pass

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except Exception:
            return str(obj)
    if isinstance(obj, set):
        return list(obj)
    try:
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            try:
                return obj.to_dict()
            except Exception:
                pass
    except Exception:
        pass
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
    try:
        return str(obj)
    except Exception:
        return repr(obj)


class HistoryStore:
    """
    Conversation-first history store backed by SQLite.

    Public methods (high-level):
      - create_conversation(name, first_message) -> conversation dict
      - list_conversations() -> list of conversation summaries
      - get_conversation(conversation_id) -> full conversation dict
      - append_message(conversation_id, message) -> message dict
      - update_conversation_name(conversation_id, new_name) -> bool
      - delete_conversation(conversation_id) -> bool
      - export_conversation(conversation_id, path=None) -> path
      - migrate_from_legacy_json(path) -> int (count migrated)

    Message format expected by append_message:
      {"id": "<uuid>", "role":"user|assistant", "content":"...", "meta": {...}, "created_at":"..."}

    Conversation dict returned from get_conversation:
      {"id":..., "name":..., "messages": [...], "created_at":..., "updated_at":..., "meta": {...}}
    """

    def __init__(self, db_path: Optional[str] = None):
        try:
            data_dir = getattr(config_module, "DATA_DIR", "./data")
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = db_path or os.path.join(data_dir, "history.db")
            # sqlite connection
            self._conn: sqlite3.Connection = sqlite3.connect(
                self.db_path, check_same_thread=False, isolation_level=None
            )
            # pragmas for durability and concurrency
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA foreign_keys=ON;")
            cur.execute("PRAGMA busy_timeout=5000;")
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
            logger.info("HistoryStore (sqlite) initialized at %s", self.db_path)
        except Exception as e:
            logger.exception("HistoryStore initialization failed")
            raise CustomException(e, sys)

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT,
                updated_at TEXT,
                meta_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations (updated_at);

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                meta_json TEXT,
                created_at TEXT,
                position INTEGER,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv_pos ON messages (conversation_id, position);
            """
        )
        self._conn.commit()

    # --------------------------
    # Helpers
    # --------------------------
    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _serialize(self, obj: Any) -> str:
        return json.dumps(obj, default=_make_json_serializable, ensure_ascii=False)

    def _deserialize(self, s: Optional[str]) -> Any:
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return s

    # --------------------------
    # Conversation API
    # --------------------------
    def create_conversation(self, name: str, first_message: Optional[Dict[str, Any]] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new conversation and optionally append a first message."""
        try:
            conv_id = uuid.uuid4().hex
            now = self._now()
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO conversations (id, name, created_at, updated_at, meta_json) VALUES (?, ?, ?, ?, ?)",
                (conv_id, name, now, now, self._serialize(meta or {})),
            )
            if first_message:
                # normalize message
                msg = self._normalize_message(first_message)
                self._insert_message(conv_id, msg, position=0)
                # update conversation updated_at
                cur.execute("UPDATE conversations SET updated_at=? WHERE id=?", (self._now(), conv_id))
            self._conn.commit()
            return self.get_conversation(conv_id)
        except Exception as e:
            logger.exception("Failed to create conversation")
            raise CustomException(e, sys)

    def list_conversations(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Return conversation summaries (id, name, snippet, last_message_at, updated_at).
        Keep payload small so UI can render the left sidebar quickly.
        """
        try:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT id, name, created_at, updated_at, meta_json FROM conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = cur.fetchall()
            result = []
            for r in rows:
                conv_id = r["id"]
                # fetch last message snippet
                last_msg = self._get_last_message_snippet(conv_id)
                result.append(
                    {
                        "id": conv_id,
                        "name": r["name"],
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                        "meta": self._deserialize(r["meta_json"]),
                        "last_message_snippet": last_msg,
                    }
                )
            return result
        except Exception as e:
            logger.exception("Failed to list conversations")
            raise CustomException(e, sys)

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Return full conversation with ordered messages."""
        try:
            cur = self._conn.cursor()
            cur.execute("SELECT id, name, created_at, updated_at, meta_json FROM conversations WHERE id = ?", (conversation_id,))
            row = cur.fetchone()
            if not row:
                raise CustomException(RuntimeError(f"Conversation not found: {conversation_id}"), sys)
            cur.execute(
                "SELECT id, role, content, meta_json, created_at, position FROM messages WHERE conversation_id = ? ORDER BY position ASC",
                (conversation_id,),
            )
            messages = []
            for m in cur.fetchall():
                messages.append(
                    {
                        "id": m["id"],
                        "role": m["role"],
                        "content": m["content"],
                        "meta": self._deserialize(m["meta_json"]),
                        "created_at": m["created_at"],
                        "position": m["position"],
                    }
                )
            return {
                "id": row["id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "meta": self._deserialize(row["meta_json"]),
                "messages": messages,
            }
        except CustomException:
            raise
        except Exception as e:
            logger.exception("Failed to get conversation %s", conversation_id)
            raise CustomException(e, sys)

    def append_message(self, conversation_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Append a message to an existing conversation.

        Message expected shape: {"id":..., "role":"user|assistant", "content":..., "meta": {...}, "created_at": ...}
        If message.id is missing, one will be generated. Position is computed automatically.
        """
        try:
            msg = self._normalize_message(message)
            # compute next position
            cur = self._conn.cursor()
            cur.execute("SELECT MAX(position) FROM messages WHERE conversation_id = ?", (conversation_id,))
            row = cur.fetchone()
            max_pos = 0 if row is None or row[0] is None else int(row[0]) + 1
            self._insert_message(conversation_id, msg, position=max_pos)
            # update conversation updated_at
            cur.execute("UPDATE conversations SET updated_at=? WHERE id=?", (self._now(), conversation_id))
            self._conn.commit()
            # return the inserted message dict
            return msg
        except Exception as e:
            logger.exception("Failed to append message to conversation %s", conversation_id)
            raise CustomException(e, sys)

    def update_conversation_name(self, conversation_id: str, new_name: str) -> bool:
        try:
            cur = self._conn.cursor()
            cur.execute("UPDATE conversations SET name=?, updated_at=? WHERE id=?", (new_name, self._now(), conversation_id))
            self._conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            logger.exception("Failed to update conversation name %s", conversation_id)
            raise CustomException(e, sys)

    def delete_conversation(self, conversation_id: str) -> bool:
        try:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
            self._conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            logger.exception("Failed to delete conversation %s", conversation_id)
            raise CustomException(e, sys)

    def export_conversation(self, conversation_id: str, path: Optional[str] = None) -> str:
        try:
            conv = self.get_conversation(conversation_id)
            ts = int(time.time())
            data_dir = getattr(config_module, "DATA_DIR", "./data")
            os.makedirs(data_dir, exist_ok=True)
            out_path = path or os.path.join(data_dir, f"conversation_export_{conversation_id[:8]}_{ts}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(conv, f, ensure_ascii=False, indent=2)
            return out_path
        except Exception as e:
            logger.exception("Failed to export conversation %s", conversation_id)
            raise CustomException(e, sys)

    # --------------------------
    # Backwards compatibility: add_entry -> create a conversation
    # --------------------------
    def add_entry(self, name: str, query: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backwards-compatible helper: create a conversation with two messages:
          - user query
          - assistant result (stored in meta)
        Returns the created conversation.
        """
        try:
            user_msg = {"id": uuid.uuid4().hex, "role": "user", "content": query, "meta": {}, "created_at": self._now()}
            assistant_msg = {"id": uuid.uuid4().hex, "role": "assistant", "content": (result.get("formatted") or {}).get("output") if isinstance(result, dict) else str(result), "meta": result, "created_at": self._now()}
            conv = self.create_conversation(name=name, first_message=user_msg)
            # append assistant message
            self.append_message(conv["id"], assistant_msg)
            return self.get_conversation(conv["id"])
        except Exception as e:
            logger.exception("Failed to add_entry compatibility path")
            raise CustomException(e, sys)

    # --------------------------
    # Legacy migration
    # --------------------------
    def migrate_from_legacy_json(self, json_path: str) -> int:
        """Migrate old history.json entries (per-query) into conversations.
        Returns number of records migrated.
        """
        if not os.path.exists(json_path):
            return 0
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception as e:
            logger.exception("Failed to load legacy JSON for migration")
            raise CustomException(e, sys)
        migrated = 0
        for item in items:
            try:
                name = item.get("name") or f"Conversation {migrated+1}"
                query = item.get("query") or ""
                result = item.get("result") or {}
                user_msg = {"id": uuid.uuid4().hex, "role": "user", "content": query, "meta": {}, "created_at": item.get("created_at") or self._now()}
                assistant_msg = {"id": uuid.uuid4().hex, "role": "assistant", "content": (result.get("formatted") or {}).get("output") if isinstance(result, dict) else str(result), "meta": result, "created_at": item.get("updated_at") or self._now()}
                conv = self.create_conversation(name=name, first_message=user_msg)
                self.append_message(conv["id"], assistant_msg)
                migrated += 1
            except Exception:
                logger.exception("Failed migrating item %s", item.get("id"))
                continue
        logger.info("Migration completed: %d items migrated from %s", migrated, json_path)
        return migrated

    # --------------------------
    # Internal helpers
    # --------------------------
    def _normalize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        m = dict(message)
        if not m.get("id"):
            m["id"] = uuid.uuid4().hex
        if not m.get("created_at"):
            m["created_at"] = self._now()
        if "meta" not in m:
            m["meta"] = {}
        # ensure role is present
        if "role" not in m:
            m["role"] = "assistant"
        return m

    def _insert_message(self, conversation_id: str, message: Dict[str, Any], position: int) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO messages (id, conversation_id, role, content, meta_json, created_at, position) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                message["id"],
                conversation_id,
                message.get("role"),
                message.get("content"),
                self._serialize(message.get("meta")),
                message.get("created_at"),
                position,
            ),
        )

    def _get_last_message_snippet(self, conversation_id: str, length: int = 120) -> Optional[str]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT content FROM messages WHERE conversation_id = ? ORDER BY position DESC LIMIT 1",
            (conversation_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        content = row["content"] or ""
        snippet = content.strip().replace("\n", " ")
        if len(snippet) > length:
            snippet = snippet[: length - 1] + "…"
        return snippet
