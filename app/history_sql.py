"""
SQLite-first HistoryStore with conversation/message model (ChatGPT-style).

See module docstring in your repo for full design rationale.
"""

import os
import sys
import json
import sqlite3
import uuid
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from app.logger import get_logger
from app.exception import CustomException
import app.config as config_module

logger = get_logger("history_store")


def _make_json_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable forms.
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

    Public methods:
      - create_conversation
      - list_conversations
      - get_conversation
      - append_message
      - update_conversation_name
      - delete_conversation
      - export_conversation
      - migrate_from_legacy_json
      - add_entry (back-compat)
    """

    def __init__(self, db_path: Optional[str] = None):
        try:
            data_dir = getattr(config_module, "DATA_DIR", "./data")
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = db_path or os.path.join(data_dir, "history.db")

            # Use a reasonable timeout to wait for locks to clear (seconds)
            # check_same_thread=False for Streamlit threads; isolation_level=None -> autocommit mode
            connect_kwargs = {"check_same_thread": False, "isolation_level": None, "timeout": 30.0}
            self._conn: sqlite3.Connection = sqlite3.connect(self.db_path, **connect_kwargs)

            # set row factory early
            self._conn.row_factory = sqlite3.Row

            # Try to set PRAGMAs with a short retry loop to handle transient "database is locked"
            cur = self._conn.cursor()
            max_attempts = 5
            attempt = 0
            backoff = 0.05
            last_exc = None
            while attempt < max_attempts:
                try:
                    cur.execute("PRAGMA journal_mode=WAL;")
                    cur.execute("PRAGMA foreign_keys=ON;")
                    # busy_timeout as PRAGMA for SQLite; also keep connect timeout above
                    cur.execute("PRAGMA busy_timeout=5000;")
                    break
                except sqlite3.OperationalError as oe:
                    last_exc = oe
                    attempt += 1
                    logger.warning("SQLite PRAGMA attempt %d failed (will retry): %s", attempt, oe)
                    time.sleep(backoff)
                    backoff *= 2
                    continue
            else:
                # all attempts failed
                logger.exception("SQLite PRAGMA setup failed after %d attempts: %s", max_attempts, last_exc)
                raise last_exc

            # initialize schema
            self._init_schema()
            logger.info("HistoryStore (sqlite) initialized at %s", self.db_path)
        except CustomException:
            # pass through CustomException unchanged
            raise
        except Exception as e:
            logger.exception("HistoryStore initialization failed")
            # ensure we close connection if partially initialized
            try:
                if hasattr(self, "_conn") and self._conn:
                    try:
                        self._conn.close()
                    except Exception:
                        pass
            finally:
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
        """Return conversation summaries (id, name, snippet, last_message_at, updated_at)."""
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
        """Append a message to an existing conversation."""
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
        try:
            user_msg = {"id": uuid.uuid4().hex, "role": "user", "content": query, "meta": {}, "created_at": self._now()}
            assistant_msg = {
                "id": uuid.uuid4().hex,
                "role": "assistant",
                "content": (result.get("formatted") or {}).get("output") if isinstance(result, dict) else str(result),
                "meta": result,
                "created_at": self._now(),
            }
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
                assistant_msg = {
                    "id": uuid.uuid4().hex,
                    "role": "assistant",
                    "content": (result.get("formatted") or {}).get("output") if isinstance(result, dict) else str(result),
                    "meta": result,
                    "created_at": item.get("updated_at") or self._now(),
                }
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
