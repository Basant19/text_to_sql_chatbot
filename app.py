#D:\text_to_sql_bot\app.py
"""
Streamlit App Entrypoint — Text-to-SQL Bot (Conversational)

Key guarantees:
- No one-shot execution (multi-turn chat supported)
- st.stop() is NEVER used for normal flow (only fatal misconfig)
- Chat history is persisted via HistoryStore
- SchemaStore remains a strict singleton
- Graph execution is stateless
- SQL, rows, and errors are always user-visible
- Rows are NEVER dropped in UI
"""

from __future__ import annotations

import os
import traceback
import hashlib
from typing import Any, Dict

import streamlit as st

# Must be the first Streamlit call
st.set_page_config(page_title="Text-to-SQL Bot", layout="wide")

# ---------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------
from app.csv_loader import CSVLoader
from app.schema_store import SchemaStore
from app.vector_search import VectorSearch
from app.graph.builder import GraphBuilder
from app.history_sql import HistoryStore
from app.logger import get_logger
from app.tools import Tools
from app.gemini_client import GeminiClient
import app.config as config_module

logger = get_logger("app")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _sanitize(obj: Any) -> Any:
    """Make objects JSON-safe for storage."""
    from datetime import datetime, date

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return str(obj)


def _render_exception_ui(exc: Exception) -> None:
    """Render a fatal exception safely in UI."""
    st.error(f"{type(exc).__name__}: {exc}")
    with st.expander("Details"):
        st.code(
            "".join(
                traceback.format_exception(
                    type(exc), exc, exc.__traceback__
                )
            )
        )


# ---------------------------------------------------------------------
# Initialization (STRICT)
# ---------------------------------------------------------------------
config = config_module

# ===============================
# LLM Provider
# ===============================
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("❌ GEMINI_API_KEY not set")
    st.stop()

llm = GeminiClient(
    api_key=api_key,
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
)

# ===============================
# CSV Loader
# ===============================
csv_loader = CSVLoader()

# ===============================
# SchemaStore (STRICT SINGLETON)
# ===============================
schema_store = SchemaStore.get_instance()
SchemaStore.set_instance(schema_store)

# ===============================
# Vector Search (OPTIONAL)
# ===============================
try:
    vector_search = VectorSearch()
except Exception:
    vector_search = None
    logger.warning("VectorSearch disabled")

# ===============================
# Tools (SHARED)
# ===============================
tools = Tools(
    schema_store=schema_store,
    vector_search=vector_search,
    provider_client=llm,
    config=config,
)

# ===============================
# Graph (STATELESS)
# ===============================
graph = GraphBuilder.build_default(tools=tools)

# ===============================
# History Store (STATEFUL)
# ===============================
history_store = HistoryStore()

# ---------------------------------------------------------------------
# Session defaults
# ---------------------------------------------------------------------
st.session_state.setdefault("selected_conversation_id", None)
st.session_state.setdefault("processed_upload_hashes", [])
st.session_state.setdefault("selected_schemas", [])
st.session_state.setdefault("rename_conv_id", None)

# ---------------------------------------------------------------------
# Core Query Handler
# ---------------------------------------------------------------------
def handle_user_query(
    *,
    conv_id: str,
    user_input: str,
    schemas: list[str],
) -> Dict[str, Any]:
    """
    Handle a single user query end-to-end.
    """
    logger.info("Handling user query (schemas=%s)", schemas)

    history_store.append_message(
        conv_id,
        {"role": "user", "content": user_input},
    )

    result = graph.run(
        user_query=user_input,
        schemas=schemas,
        run_query=True,
    )

    history_store.append_message(
        conv_id,
        {
            "role": "assistant",
            "content": result.get("formatted_text", ""),
            "meta": _sanitize(result),
        },
    )

    return result


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
try:
    st.title("📊 Text-to-SQL Bot — Conversational SQL")

    # ==========================================================
    # Sidebar — Conversations
    # ==========================================================
    with st.sidebar:
        st.header("💬 Conversations")

        if st.button("➕ New Conversation", use_container_width=True):
            conv = history_store.create_conversation("New Conversation")
            st.session_state["selected_conversation_id"] = conv["id"]
            st.rerun()

        for c in history_store.list_conversations():
            cols = st.columns([0.65, 0.15, 0.2])
            with cols[0]:
                if st.button(c["name"], key=f"open_{c['id']}", use_container_width=True):
                    st.session_state["selected_conversation_id"] = c["id"]
                    st.rerun()

            with cols[1]:
                if st.button("✏️", key=f"rename_{c['id']}"):
                    st.session_state["rename_conv_id"] = c["id"]

            with cols[2]:
                if st.button("🗑️", key=f"delete_{c['id']}"):
                    history_store.delete_conversation(c["id"])
                    if st.session_state["selected_conversation_id"] == c["id"]:
                        st.session_state["selected_conversation_id"] = None
                    st.rerun()

        # Rename UI
        if st.session_state["rename_conv_id"]:
            cid = st.session_state["rename_conv_id"]
            conv = history_store.get_conversation(cid)
            new_name = st.text_input(
                "Rename conversation",
                value=conv["name"],
                key="rename_input",
            )
            if st.button("Save name"):
                history_store.rename_conversation(cid, new_name)
                st.session_state["rename_conv_id"] = None
                st.rerun()

        st.markdown("---")
        st.subheader("📂 CSV Upload")

        uploads = st.file_uploader(
            "Upload CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )

        if uploads:
            for f in uploads:
                raw = f.read()
                sha = hashlib.sha256(raw).hexdigest()
                if sha in st.session_state["processed_upload_hashes"]:
                    continue

                from io import BytesIO

                bio = BytesIO(raw)
                bio.name = f.name

                path = csv_loader.save_csv(bio)
                schema_store.add_csv(path)
                st.session_state["processed_upload_hashes"].append(sha)
                st.success(f"Uploaded {f.name}")

    # ==========================================================
    # Schema Selector
    # ==========================================================
    st.subheader("Available Schemas")

    schemas = schema_store.list_csvs()
    if not schemas:
        st.info("Upload CSV files to enable querying.")

    st.session_state["selected_schemas"] = st.multiselect(
        "Schemas",
        options=schemas,
        default=st.session_state["selected_schemas"],
    )

    st.markdown("---")

    # ==========================================================
    # Chat Area
    # ==========================================================
    conv_id = st.session_state["selected_conversation_id"]
    if not conv_id:
        st.info("Create or select a conversation.")
    else:
        conv = history_store.get_conversation(conv_id)
        st.subheader(conv["name"])

        # Render full chat history
        for msg in conv.get("messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    sql = msg.get("meta", {}).get("sql")
                    rows = msg.get("meta", {}).get("rows")

                    if sql:
                        st.code(sql, language="sql")
                    if rows is not None:
                        if rows:
                            st.dataframe(rows)
                        else:
                            st.info("Query executed successfully, but returned 0 rows.")

        # Chat input (true multi-turn)
        user_input = st.chat_input("Ask a question")

        if user_input:
            if not st.session_state["selected_schemas"]:
                st.warning("Select at least one schema.")
            else:
                with st.spinner("Thinking..."):
                    result = handle_user_query(
                        conv_id=conv_id,
                        user_input=user_input.strip(),
                        schemas=st.session_state["selected_schemas"],
                    )

                with st.chat_message("assistant"):
                    if result.get("error"):
                        st.warning(result["error"])

                    if result.get("sql"):
                        st.code(result["sql"], language="sql")

                    if result.get("rows") is not None:
                        if result["rows"]:
                            st.dataframe(result["rows"])
                        else:
                            st.info(
                                "Query executed successfully, but returned 0 rows."
                            )
                    elif result.get("formatted_text"):
                        st.markdown(result["formatted_text"])

except Exception as e:
    logger.exception("Fatal UI error")
    _render_exception_ui(e)