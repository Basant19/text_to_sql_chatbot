# app.py (updated: safe rerun, set_page_config at top, top-level error handling)
import streamlit as st
import os
import json
import time
from typing import List, Dict, Any, Optional

# IMPORTANT: set_page_config must be called before any other Streamlit UI calls.
st.set_page_config(page_title="Text-to-SQL Bot", layout="wide")

from app.csv_loader import CSVLoader
from app.schema_store import SchemaStore
from app.vector_search import VectorSearch
from app.graph.builder import GraphBuilder
from app.history_sql import HistoryStore
from app.logger import get_logger
import app.config as config_module

logger = get_logger("app")


def _sanitize_for_history(obj: Any) -> Any:
    from datetime import datetime, date

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            try:
                sanitized[k] = _sanitize_for_history(v)
            except Exception:
                sanitized[k] = str(v)
        return sanitized
    if isinstance(obj, (list, tuple, set)):
        try:
            return [_sanitize_for_history(v) for v in obj]
        except Exception:
            return [str(v) for v in obj]
    try:
        json.dumps(obj, default=str)
        return str(obj)
    except Exception:
        return repr(obj)


# helper: safe rerun (Streamlit builds differ)
# New behavior: prefer experimental_rerun(); if unavailable or it errors, set a session flag
# and let the top-of-script handler call experimental_rerun() on the next run.
def _safe_rerun():
    rerun = getattr(st, "experimental_rerun", None)
    if callable(rerun):
        try:
            # try best-effort immediate rerun
            rerun()
            return
        except Exception:
            # mark that a rerun is desired; do not call st.stop()
            st.session_state["_needs_rerun"] = True
            return
    else:
        st.session_state["_needs_rerun"] = True
        return


# ---------------------------
# Initialize core components
# ---------------------------
config = config_module
csv_loader = CSVLoader()
schema_store = SchemaStore()
vector_search = VectorSearch()

# Build the default agent graph (LangGraph)
try:
    graph = GraphBuilder.build_default()
except Exception:
    logger.exception("Failed to build default GraphBuilder. Check node implementations.")
    graph = None

# Persistent conversation-first history store (SQLite)
history_store = HistoryStore()  # uses DATA_DIR path from config/data dir by default

# Streamlit state initialization (ensure keys exist)
st.session_state.setdefault("selected_conversation_id", None)
st.session_state.setdefault("conversations_meta", [])
st.session_state.setdefault("conversation_cache", {})
st.session_state.setdefault("csv_label_to_path", {})
st.session_state.setdefault("csv_path_to_table", {})

# If a previous run requested a rerun, trigger it now (best-effort)
if st.session_state.pop("_needs_rerun", False):
    rerun = getattr(st, "experimental_rerun", None)
    if callable(rerun):
        try:
            rerun()
        except Exception:
            # nothing more to do; continue rendering the page gracefully
            pass

# Wrap main UI logic in a top-level try/except so exceptions are visible instead of leaving a partial page.
try:
    st.title("📊 Text-to-SQL Bot — Conversation Mode")

    # Freshen conversations meta at startup if empty
    if not st.session_state.get("conversations_meta"):
        try:
            st.session_state.conversations_meta = history_store.list_conversations()
        except Exception:
            logger.exception("Initial list_conversations failed")
            st.session_state.conversations_meta = []

    # ---------------------------
    # Sidebar: Conversations + CSV Management
    # ---------------------------
    with st.sidebar:
        st.header("Conversations")

        # Search/filter box
        q = st.text_input("Search conversations", value="")

        # New conversation
        if st.button("➕ New Conversation"):
            default_name = f"Conversation {len(st.session_state.conversations_meta) + 1}"
            try:
                conv = history_store.create_conversation(name=default_name)
                # refresh meta list & select
                st.session_state.conversations_meta = history_store.list_conversations()
                st.session_state.selected_conversation_id = conv["id"]
                # cache the freshly created conversation to avoid immediate DB round-trip
                st.session_state.conversation_cache[conv["id"]] = conv
                # request a safe rerun (will be best-effort)
                _safe_rerun()
            except Exception:
                logger.exception("Failed to create new conversation")
                st.error("Could not create a new conversation. Check logs.")

        # Conversation list (filtered)
        convs = st.session_state.conversations_meta or []
        if q:
            convs = [
                c
                for c in convs
                if q.lower() in (c.get("name") or "").lower()
                or q.lower() in (c.get("last_message_snippet") or "").lower()
            ]

        # Render list (compact)
        for c in convs:
            # Build a safe display label (avoid embedding raw newlines into button text)
            name = c.get("name") or "Untitled"
            snippet = c.get("last_message_snippet") or ""
            display_label = name if not snippet else f"{name} — {snippet}"
            if st.button(display_label, key=f"select_{c['id']}"):
                st.session_state.selected_conversation_id = c["id"]
                # load into cache
                try:
                    st.session_state.conversation_cache[c["id"]] = history_store.get_conversation(c["id"])
                except Exception:
                    logger.exception("Failed to load conversation %s", c["id"])
                    st.error("Failed to load conversation from storage.")
                _safe_rerun()

        st.markdown("---")
        st.subheader("CSV Management")
        uploaded_files = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                try:
                    saved_path = csv_loader.save_csv(file)
                    st.success(f"Uploaded {file.name}")
                    logger.info(f"Saved {file.name} -> {saved_path}")
                    # auto-register schema with canonical name so context_node and validate_node will find it
                    try:
                        metadata_list = csv_loader.load_and_extract([saved_path])
                        if metadata_list and isinstance(metadata_list, list):
                            for meta in metadata_list:
                                # CSVLoader may return dicts; handle gracefully
                                csv_name = None
                                orig_name = os.path.basename(saved_path)
                                if isinstance(meta, dict):
                                    csv_name = meta.get("table_name") or meta.get("table") or os.path.splitext(os.path.basename(saved_path))[0]
                                    orig_name = meta.get("original_name") or orig_name
                                if not csv_name:
                                    csv_name = os.path.splitext(os.path.basename(saved_path))[0]
                                schema_store.add_csv(saved_path, csv_name=csv_name)
                                display_label = f"{csv_name} — {orig_name}"
                                st.session_state["csv_label_to_path"][display_label] = saved_path
                                st.session_state["csv_path_to_table"][saved_path] = csv_name
                                logger.info("Registered schema for %s (path=%s)", csv_name, saved_path)
                    except Exception:
                        logger.exception("Failed to auto-register schema for uploaded CSV")
                except Exception as e:
                    st.error(f"Failed to save {file.name}: {e}")
                    logger.exception("CSV upload failed")

        # List uploaded CSVs (friendly labels)
        try:
            uploaded_meta = csv_loader.list_uploaded_csvs() or []
        except Exception:
            logger.exception("Failed to list uploaded CSVs")
            uploaded_meta = []

        # Ensure session mappings include any previously uploaded meta (defensive)
        for m in uploaded_meta:
            p = None
            table = None
            orig = None

            if isinstance(m, str):
                p = m
                table = os.path.splitext(os.path.basename(p))[0]
                orig = os.path.basename(p)
            elif isinstance(m, dict):
                p = m.get("path") or m.get("filepath") or m.get("file") or m.get("filename") or ""
                if not p:
                    continue
                table = m.get("table_name") or m.get("table") or os.path.splitext(os.path.basename(p))[0]
                orig = m.get("original_name") or m.get("original") or os.path.basename(p)
            else:
                continue

            label = f"{table} — {orig}"
            if label not in st.session_state["csv_label_to_path"]:
                st.session_state["csv_label_to_path"][label] = p
            if p not in st.session_state["csv_path_to_table"]:
                st.session_state["csv_path_to_table"][p] = table

        csv_display_options = list(st.session_state["csv_label_to_path"].keys())
        csv_display_options.sort()

        selected_labels = st.multiselect(
            "Select CSVs for context (used when sending messages)",
            options=csv_display_options,
            default=[],
            key="selected_csvs_sidebar",
        )

        st.markdown("---")
        if st.button("Refresh Conversations"):
            try:
                st.session_state.conversations_meta = history_store.list_conversations()
            except Exception:
                logger.exception("Failed to refresh conversations_meta")
                st.error("Failed to refresh conversations list.")
            _safe_rerun()

        st.caption("Tip: create a conversation first, then ask questions in the main area.")

    # ---------------------------
    # Main area: Chat view
    # ---------------------------
    col1, col2 = st.columns([3, 1])
    with col1:
        # Conversation header / controls
        if st.session_state.selected_conversation_id:
            conv_id = st.session_state.selected_conversation_id
            if conv_id not in st.session_state.conversation_cache:
                try:
                    st.session_state.conversation_cache[conv_id] = history_store.get_conversation(conv_id)
                except Exception:
                    st.error("Failed to load conversation. It may have been deleted.")
                    st.session_state.selected_conversation_id = None

        if not st.session_state.selected_conversation_id:
            st.info("Select or create a conversation in the left sidebar to begin.")
        else:
            conv = st.session_state.conversation_cache.get(st.session_state.selected_conversation_id) or {}
            header_cols = st.columns([3, 1, 1, 1])
            name_box = header_cols[0].text_input(
                "Conversation name", value=conv.get("name", ""), key=f"name_{conv.get('id')}"
            )
            if name_box != conv.get("name"):
                try:
                    ok = history_store.update_conversation_name(conv["id"], name_box)
                    if ok:
                        st.session_state.conversation_cache[conv["id"]]["name"] = name_box
                        st.session_state.conversations_meta = history_store.list_conversations()
                        st.success("Renamed")
                except Exception:
                    st.error("Rename failed; see logs.")
            if header_cols[1].button("Export"):
                try:
                    path = history_store.export_conversation(conv["id"])
                    st.success(f"Exported to {path}")
                except Exception:
                    logger.exception("Export failed")
                    st.error("Export failed; see logs.")

            delete_flag_key = f"confirm_delete_{conv.get('id')}"
            if header_cols[2].button("Delete"):
                st.session_state[delete_flag_key] = True

            if st.session_state.get(delete_flag_key):
                cf_col1, cf_col2 = st.columns([1, 1])
                if cf_col1.button("Confirm Delete"):
                    try:
                        deleted = history_store.delete_conversation(conv["id"])
                        if deleted:
                            st.session_state.conversations_meta = history_store.list_conversations()
                            st.session_state.conversation_cache.pop(conv["id"], None)
                            st.session_state.selected_conversation_id = None
                            st.session_state.pop(delete_flag_key, None)
                            st.success("Conversation deleted.")
                            _safe_rerun()
                        else:
                            st.error("Conversation not found or could not be deleted.")
                    except Exception:
                        st.error("Delete failed; see logs.")
                        logger.exception("Failed to delete conversation %s", conv.get("id"))
                if cf_col2.button("Cancel"):
                    st.session_state.pop(delete_flag_key, None)

            header_cols[3].markdown(f"Updated: {conv.get('updated_at', '')}")
            st.markdown("---")

            # Messages stream
            messages = conv.get("messages", [])
            for m in messages:
                if not isinstance(m, dict):
                    with st.container():
                        st.markdown(f"**Assistant:** {str(m)}")
                    continue

                if m.get("role") == "user":
                    with st.container():
                        st.markdown(f"**You:** {m.get('content')}")
                else:
                    with st.container():
                        st.markdown(f"**Assistant:** {m.get('content')}")
                        meta = m.get("meta") or {}
                        explain = None
                        if isinstance(meta, dict):
                            formatted = meta.get("formatted") or (meta.get("meta") or {}).get("formatted") or {}
                            if isinstance(formatted, dict):
                                explain = formatted.get("explain") or formatted.get("summary") or None
                            if not explain:
                                explain = meta.get("explain") or meta.get("summary")
                        if explain:
                            st.markdown(f"_Explanation_: {explain}")
                        if meta:
                            with st.expander("Show details (SQL, timings, raw)"):
                                try:
                                    st.json(meta)
                                except Exception:
                                    st.write(str(meta))

            st.markdown("---")

            # Clear-on-next-run mechanism for user input (avoid modifying widget after creation)
            clear_next = st.session_state.pop("__clear_user_input", False)
            initial_user_input = "" if clear_next else st.session_state.get("user_input", "")

            # Input area for new user message (use 'value' param so we can control initial)
            user_input = st.text_area("Enter message", value=initial_user_input, key="user_input", height=120)

            send_col1, send_col2 = st.columns([1, 1])
            execute_sql_checkbox = send_col2.checkbox(
                "Execute SQL if valid?", value=True, key="execute_sql_checkbox"
            )
            if send_col1.button("Send"):
                if not user_input or not user_input.strip():
                    st.warning("Please enter a message.")
                else:
                    user_msg = {"role": "user", "content": user_input.strip(), "meta": {}}
                    try:
                        history_store.append_message(conv["id"], user_msg)
                    except Exception:
                        st.error("Failed to append user message; see logs.")
                        logger.exception("append user message failed")

                    # Resolve selected labels -> canonical table names to pass into graph.run
                    selected_labels = st.session_state.get("selected_csvs_sidebar", [])
                    label_to_path = st.session_state.get("csv_label_to_path", {})
                    path_to_table = st.session_state.get("csv_path_to_table", {})

                    selected_paths = [label_to_path.get(lbl) for lbl in selected_labels if label_to_path.get(lbl)]
                    selected_table_names: List[str] = []
                    for p in selected_paths:
                        if not p:
                            continue
                        t = path_to_table.get(p)
                        if t:
                            selected_table_names.append(t)
                        else:
                            selected_table_names.append(os.path.splitext(os.path.basename(p))[0])

                    # call agent (graph.run) with canonical table names
                    if graph is None:
                        st.error("Agent unavailable. Check server logs.")
                    else:
                        with st.spinner("Generating answer..."):
                            try:
                                result = graph.run(user_input.strip(), selected_table_names, run_query=execute_sql_checkbox)
                            except Exception as e:
                                logger.exception("Agent run failed")
                                result = {"error": str(e)}

                        # build assistant content safely
                        assistant_content = "No answer generated."
                        formatted = result.get("formatted") if isinstance(result, dict) else None
                        if isinstance(formatted, dict):
                            assistant_content = formatted.get("output") or formatted.get("sql") or formatted.get("explain") or assistant_content
                        else:
                            if isinstance(result, dict):
                                assistant_content = result.get("sql") or result.get("error") or assistant_content
                            else:
                                assistant_content = str(result) if result is not None else assistant_content

                        # sanitize meta for storage
                        try:
                            sanitized_meta = _sanitize_for_history(result)
                        except Exception:
                            sanitized_meta = {"error": "failed to sanitize meta"}

                        if not isinstance(sanitized_meta, dict):
                            sanitized_meta = {"_raw": sanitized_meta}

                        assistant_msg = {"role": "assistant", "content": assistant_content, "meta": sanitized_meta}
                        try:
                            history_store.append_message(conv["id"], assistant_msg)
                            # refresh conversation in cache & meta list
                            st.session_state.conversation_cache[conv["id"]] = history_store.get_conversation(conv["id"])
                            st.session_state.conversations_meta = history_store.list_conversations()
                            # request clearing the input on next run instead of mutating widget now
                            st.session_state["__clear_user_input"] = True
                            _safe_rerun()
                        except Exception:
                            st.error("Failed to persist assistant message; see logs.")
                            logger.exception("append assistant message failed")

    with col2:
        st.subheader("Conversation Tools")
        if st.button("Refresh conversation list"):
            try:
                st.session_state.conversations_meta = history_store.list_conversations()
            except Exception:
                logger.exception("Failed to refresh conversations_meta")
                st.error("Failed to refresh conversations list.")
            _safe_rerun()

        st.markdown("Export / Import")
        if st.button("Export all conversations"):
            try:
                all_meta = st.session_state.conversations_meta
                out_path = os.path.join(getattr(config_module, "DATA_DIR", "./data"), f"conversations_meta_{int(time.time())}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(all_meta, f, ensure_ascii=False, indent=2)
                st.success(f"Exported to {out_path}")
            except Exception:
                logger.exception("Failed to export all conversations")
                st.error("Export failed; see logs.")
        st.markdown("---")
        st.info("Developer: raw logs are available in the server console. Use 'Show details' in message to reveal SQL/raw LLM output.")

    # ---------------------------
    # Ensure sessions meta up-to-date on exit
    # ---------------------------
    try:
        st.session_state.conversations_meta = history_store.list_conversations()
    except Exception:
        logger.exception("Failed to refresh conversations_meta at end of request")

except Exception as e:
    # Top-level catch so the page doesn't render partially and to show a user-facing error.
    logger.exception("Top-level error in app.py")
    st.error(f"An internal error occurred while rendering the app: {e}")
