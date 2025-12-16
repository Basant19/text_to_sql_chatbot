#D:\text_to_sql_bot\app.py
# app.py
"""
Streamlit app entrypoint for Text-to-SQL Bot.

Key guarantees:
- SchemaStore is a strict singleton across reruns
- GraphBuilder, nodes, and executor share the same SchemaStore
- Clear user-visible warnings for generation / validation / execution errors
- Deterministic CSV upload + schema sync
- Results always render in UI
"""

from __future__ import annotations

import os
import json
import time
import traceback
import hashlib
from typing import Any, Optional

import streamlit as st

# MUST be first Streamlit call
st.set_page_config(page_title="Text-to-SQL Bot", layout="wide")

from app.csv_loader import CSVLoader
from app.schema_store import SchemaStore
from app.vector_search import VectorSearch
from app.graph.builder import GraphBuilder
from app.history_sql import HistoryStore
from app.logger import get_logger
import app.config as config_module

logger = get_logger("app")


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _sanitize_for_history(obj: Any) -> Any:
    """Make objects JSON / history safe."""
    from datetime import datetime, date

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _sanitize_for_history(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_history(v) for v in obj]
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def _safe_rerun() -> None:
    """Streamlit-compatible rerun wrapper."""
    rerun = getattr(st, "experimental_rerun", None)
    if callable(rerun):
        try:
            rerun()
            return
        except Exception:
            pass
    st.session_state["_needs_rerun"] = True


def _render_exception_ui(exc: Exception, hint: Optional[str] = None) -> None:
    msg = f"{type(exc).__name__}: {exc}"
    st.error(f"{hint} — {msg}" if hint else msg)
    with st.expander("Show details"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


# ---------------------------------------------------------------------
# Core initialization (SINGLETON SAFE)
# ---------------------------------------------------------------------
logger.info("Initializing app components")

config = config_module

# CSV Loader
try:
    csv_loader = CSVLoader()
except Exception:
    logger.exception("CSVLoader initialization failed")
    csv_loader = None

# SchemaStore singleton
schema_store = None
try:
    schema_store = SchemaStore.get_instance()
    SchemaStore.set_instance(schema_store)
    logger.info("SchemaStore singleton ready")
except Exception:
    logger.exception("SchemaStore initialization failed")

# Vector search (optional)
try:
    vector_search = VectorSearch()
except Exception:
    logger.exception("VectorSearch init failed")
    vector_search = None

# Graph builder
try:
    graph = GraphBuilder.build_default()
except Exception:
    logger.exception("GraphBuilder initialization failed")
    graph = None

# History store
try:
    history_store = HistoryStore()
except Exception:
    logger.exception("HistoryStore init failed")
    history_store = None

logger.info("App initialization complete")


# ---------------------------------------------------------------------
# Streamlit session defaults
# ---------------------------------------------------------------------
st.session_state.setdefault("selected_conversation_id", None)
st.session_state.setdefault("conversations_meta", [])
st.session_state.setdefault("conversation_cache", {})
st.session_state.setdefault("csv_label_to_path", {})
st.session_state.setdefault("csv_path_to_table", {})
st.session_state.setdefault("processed_upload_hashes", [])

if st.session_state.pop("_needs_rerun", False):
    _safe_rerun()


# ---------------------------------------------------------------------
# Sync UI mappings from SchemaStore
# ---------------------------------------------------------------------
def _sync_session_mappings_from_store() -> None:
    if not schema_store:
        return

    st.session_state["csv_label_to_path"].clear()
    st.session_state["csv_path_to_table"].clear()

    for entry in schema_store.list_csvs_meta() or []:
        path = entry.get("path")
        if not path:
            continue
        canonical = entry.get("canonical") or entry.get("key")
        aliases = entry.get("aliases") or []
        orig = os.path.basename(path)

        names = [canonical] + [a for a in aliases if a and a != canonical]
        for name in names:
            label = f"{name} — {orig}"
            st.session_state["csv_label_to_path"][label] = path
            st.session_state["csv_path_to_table"][path] = canonical


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
try:
    st.title("📊 Text-to-SQL Bot — Conversation Mode")

    if history_store and not st.session_state["conversations_meta"]:
        st.session_state["conversations_meta"] = history_store.list_conversations()

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Conversations")

        if st.button("➕ New Conversation") and history_store:
            conv = history_store.create_conversation(
                name=f"Conversation {len(st.session_state['conversations_meta']) + 1}"
            )
            st.session_state["selected_conversation_id"] = conv["id"]
            st.session_state["conversation_cache"][conv["id"]] = conv
            st.session_state["conversations_meta"] = history_store.list_conversations()
            _safe_rerun()

        for c in st.session_state["conversations_meta"]:
            if st.button(c.get("name", "Conversation"), key=f"conv_{c['id']}"):
                st.session_state["selected_conversation_id"] = c["id"]
                if history_store:
                    st.session_state["conversation_cache"][c["id"]] = history_store.get_conversation(c["id"])
                _safe_rerun()

        st.markdown("---")
        st.subheader("CSV Management")

        uploaded_files = st.file_uploader(
            "Upload CSVs",
            type=["csv"],
            accept_multiple_files=True,
        )

        if uploaded_files and csv_loader:
            for f in uploaded_files:
                try:
                    data = f.read()
                    sha = hashlib.sha256(data).hexdigest()

                    if sha in st.session_state["processed_upload_hashes"]:
                        continue

                    from io import BytesIO
                    bio = BytesIO(data)
                    bio.name = f.name

                    path = csv_loader.save_csv(bio)
                    st.success(f"Uploaded {f.name}")

                    if schema_store:
                        schema_store.add_csv(path)

                    st.session_state["processed_upload_hashes"].append(sha)

                except Exception as e:
                    logger.exception("CSV upload failed")
                    _render_exception_ui(e, "Upload failed")

        _sync_session_mappings_from_store()

        labels = sorted(st.session_state["csv_label_to_path"].keys())
        st.multiselect(
            "Select CSVs for context",
            options=labels,
            key="selected_csvs_sidebar",
        )

    # ---------------- Main Chat ----------------
    if not st.session_state["selected_conversation_id"]:
        st.info("Select or create a conversation from the sidebar.")
    else:
        conv_id = st.session_state["selected_conversation_id"]
        conv = st.session_state["conversation_cache"].get(conv_id, {})

        st.subheader(conv.get("name", "Conversation"))

        for m in conv.get("messages", []):
            st.markdown(f"**{m.get('role', '').capitalize()}:** {m.get('content', '')}")

        user_input = st.text_area("Enter message", height=120)

        if st.button("Send"):
            if not user_input.strip():
                st.warning("Please enter a message.")
                st.stop()

            if history_store:
                history_store.append_message(
                    conv_id,
                    {"role": "user", "content": user_input.strip(), "meta": {}}
                )

            selected_labels = st.session_state.get("selected_csvs_sidebar", [])
            paths = [st.session_state["csv_label_to_path"].get(lbl) for lbl in selected_labels]
            tables = [st.session_state["csv_path_to_table"].get(p) for p in paths if p]

            if not tables:
                st.warning("Select at least one CSV.")
                st.stop()

            if not graph:
                st.error("Agent unavailable.")
                st.stop()

            with st.spinner("Generating answer..."):
                result = graph.run(
                    user_query=user_input.strip(),
                    schemas=tables,
                    run_query=True,
                )

                # -----------------------------
                # User-visible warning handling
                # -----------------------------
                if isinstance(result, dict) and result.get("error"):
                    st.warning(f"⚠️ Query failed: {result['error']}")
                    logger.warning("User-visible error: %s", result["error"])
                    st.stop()

                # -----------------------------
                # Render results to UI
                # -----------------------------
                rows = result.get("rows") or result.get("data")
                formatted_text = result.get("formatted_text") or result.get("formatted", {}).get("text")

                if rows:
                    st.subheader("Query Result")
                    st.dataframe(rows)
                elif formatted_text:
                    st.subheader("Result")
                    st.text(formatted_text)
                else:
                    st.info("No results returned.")

                assistant_msg = {
                    "role": "assistant",
                    "content": formatted_text or str(rows) or str(result),
                    "meta": _sanitize_for_history(result),
                }

                if history_store:
                    history_store.append_message(conv_id, assistant_msg)

                st.session_state["conversation_cache"][conv_id] = history_store.get_conversation(conv_id)
                st.session_state["conversations_meta"] = history_store.list_conversations()

                _safe_rerun()

except Exception as e:
    logger.exception("Fatal app error")
    _render_exception_ui(e, "Internal error")
