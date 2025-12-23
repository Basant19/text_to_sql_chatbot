# app/history_sql.py
"""
SQLite-first HistoryStore with conversation/message model (ChatGPT-style).

Key guarantees:
- create_conversation() NEVER crashes due to missing name
- Safe defaults are applied at the storage layer
- Backward-compatible with older UI + tests
"""

from __future__ import annotations

import os
import sys
import json
import sqlite3
import uuid
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from app.logger import get_logger
from app.exception import CustomException
import app.config as config_module

logger = get_logger("history_store")


# ---------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------
def _make_json_serializable(obj: Any) -> Any:
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
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
    except Exception:
        pass

    try:
        if hasattr(obj, "__dict__"):
            return {k: _make_json_serializable(v) for k, v in vars(obj).items()}
    except Exception:
        pass

    return str(obj)


# ---------------------------------------------------------------------
# HistoryStore
# ---------------------------------------------------------------------
class HistoryStore:
    """
    Conversation-first history store backed by SQLite.

    Public API:
      - create_conversation
      - list_conversations
      - get_conversation
      - append_message
      - rename_conversation      ✅ UI-facing alias
      - update_conversation_name
      - delete_conversation
      - export_conversation
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(self, db_path: Optional[str] = None):
        try:
            data_dir = getattr(config_module, "DATA_DIR", "./data")
            os.makedirs(data_dir, exist_ok=True)

            self.db_path = db_path or os.path.join(data_dir, "history.db")

            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,
                timeout=30.0,
            )
            self._conn.row_factory = sqlite3.Row

            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA foreign_keys=ON;")
            cur.execute("PRAGMA busy_timeout=5000;")

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

            CREATE INDEX IF NOT EXISTS idx_messages_conv_pos
              ON messages (conversation_id, position);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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

    def _default_conversation_name(self) -> str:
        return f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"

    # ------------------------------------------------------------------
    # Conversation API
    # ------------------------------------------------------------------
    def create_conversation(
        self,
        name: Optional[str] = None,
        *,
        first_message: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            conv_id = uuid.uuid4().hex
            now = self._now()
            final_name = (name or "").strip() or self._default_conversation_name()

            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO conversations (id, name, created_at, updated_at, meta_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conv_id, final_name, now, now, self._serialize(meta or {})),
            )

            if first_message:
                msg = self._normalize_message(first_message)
                self._insert_message(conv_id, msg, position=0)

            self._conn.commit()
            return self.get_conversation(conv_id)

        except Exception as e:
            logger.exception("Failed to create conversation")
            raise CustomException(e, sys)

    def update_conversation_name(self, conversation_id: str, new_name: str) -> None:
        if not new_name or not new_name.strip():
            raise ValueError("Conversation name cannot be empty")

        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE conversations
            SET name = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_name.strip(), self._now(), conversation_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # ✅ NEW: UI-compatible alias
    # ------------------------------------------------------------------
    def rename_conversation(self, conversation_id: str, new_name: str) -> None:
        """
        Public alias for update_conversation_name (UI compatibility).
        """
        self.update_conversation_name(conversation_id, new_name)
        logger.info(
            "Conversation renamed | id=%s | name=%s",
            conversation_id[:8],
            new_name,
        )

    def delete_conversation(self, conversation_id: str) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self._conn.commit()
        logger.info("Conversation deleted | id=%s", conversation_id[:8])

    # ------------------------------------------------------------------
    # Read APIs
    # ------------------------------------------------------------------
    def list_conversations(self, limit: int = 100) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT id, name, created_at, updated_at, meta_json
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "meta": self._deserialize(r["meta_json"]),
            }
            for r in rows
        ]

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        row = cur.fetchone()
        if not row:
            raise CustomException(RuntimeError("Conversation not found"), sys)

        cur.execute(
            """
            SELECT id, role, content, meta_json, created_at, position
            FROM messages
            WHERE conversation_id = ?
            ORDER BY position ASC
            """,
            (conversation_id,),
        )

        messages = [
            {
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "meta": self._deserialize(m["meta_json"]),
                "created_at": m["created_at"],
                "position": m["position"],
            }
            for m in cur.fetchall()
        ]

        return {
            "id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "meta": self._deserialize(row["meta_json"]),
            "messages": messages,
        }

    # ------------------------------------------------------------------
    # Message API
    # ------------------------------------------------------------------
    def append_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        msg = self._normalize_message(message)
        cur = self._conn.cursor()

        cur.execute(
            "SELECT COALESCE(MAX(position), -1) FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        position = int(cur.fetchone()[0]) + 1

        self._insert_message(conversation_id, msg, position)
        cur.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (self._now(), conversation_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Export API
    # ------------------------------------------------------------------
    def export_conversation(self, conversation_id: str) -> Dict[str, Any]:
        return self.get_conversation(conversation_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        m = dict(message)
        m.setdefault("id", uuid.uuid4().hex)
        m.setdefault("created_at", self._now())
        m.setdefault("meta", {})
        m.setdefault("role", "assistant")
        return m

    def _insert_message(self, conversation_id: str, message: Dict[str, Any], position: int) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO messages
              (id, conversation_id, role, content, meta_json, created_at, position)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
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
