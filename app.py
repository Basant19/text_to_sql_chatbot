"""
Streamlit app entrypoint for Text-to-SQL Bot.

This updated version:
 - Adds more defensive initialization for components (config, CSV loader, schema store, vector search, graph)
 - Improves session-state syncing with SchemaStore
 - Better error handling & logging around uploads, schema registration, and agent runs
 - Keeps the same user-facing layout and behavior while being resilient when optional backends are missing
"""
import os
import json
import time
from typing import List, Dict, Any, Optional

import streamlit as st

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


# ---------------------------
# Utilities
# ---------------------------
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


def _safe_rerun():
    """
    Safe streamlit rerun compatible with older/newer streamlit versions.
    """
    rerun = getattr(st, "experimental_rerun", None)
    if callable(rerun):
        try:
            rerun()
            return
        except Exception:
            st.session_state["_needs_rerun"] = True
            return
    else:
        st.session_state["_needs_rerun"] = True
        return


# ---------------------------
# Initialize core components (defensive)
# ---------------------------
# Config module (keep as module for backward compatibility)
config = config_module

# CSV loader
try:
    csv_loader = CSVLoader()
except Exception:
    logger.exception("Failed to initialize CSVLoader; CSV features may be limited.")
    csv_loader = None

# Schema store
try:
    schema_store = SchemaStore()
except Exception:
    logger.exception("Failed to initialize SchemaStore; schema features may be limited.")
    schema_store = None

# VectorSearch (optional: may depend on faiss/numpy)
try:
    vector_search = VectorSearch()
except Exception:
    logger.exception("Failed to initialize VectorSearch; vector features may be limited.")
    vector_search = None

# Graph builder (agent)
try:
    # GraphBuilder.build_default may rely on other components; wrap to avoid crash
    graph = GraphBuilder.build_default()
except Exception:
    logger.exception("Failed to build default GraphBuilder. Agent/graph features will be unavailable.")
    graph = None

# Persistent conversation-first history store (SQLite)
try:
    history_store = HistoryStore()
except Exception:
    logger.exception("Failed to initialize HistoryStore; conversation persistence will be disabled.")
    history_store = None

# ---------------------------
# Streamlit session-state defaults
# ---------------------------
st.session_state.setdefault("selected_conversation_id", None)
st.session_state.setdefault("conversations_meta", [])
st.session_state.setdefault("conversation_cache", {})
st.session_state.setdefault("csv_label_to_path", {})   # label -> path
st.session_state.setdefault("csv_path_to_table", {})   # path -> canonical table name

# If a previous run requested a rerun, try to rerun now
if st.session_state.pop("_needs_rerun", False):
    try:
        rerun = getattr(st, "experimental_rerun", None)
        if callable(rerun):
            rerun()
    except Exception:
        pass


# ---------------------------
# Helper: session sync from SchemaStore
# ---------------------------
def _sync_session_mappings_from_store():
    """
    Rebuild st.session_state['csv_label_to_path'] and ['csv_path_to_table']
    from schema_store.list_csvs_meta(). This is the canonical sync function.
    It overwrites session mappings for the entries present in the SchemaStore,
    but won't clobber unrelated session mappings.
    """
    try:
        st.session_state["csv_label_to_path"] = st.session_state.get("csv_label_to_path", {}) or {}
        st.session_state["csv_path_to_table"] = st.session_state.get("csv_path_to_table", {}) or {}

        if not schema_store:
            logger.debug("SchemaStore not configured; skipping sync")
            return

        store_meta = schema_store.list_csvs_meta() or []
        for entry in store_meta:
            try:
                path = entry.get("path") or ""
                if not path:
                    continue
                canonical = entry.get("canonical") or entry.get("friendly") or entry.get("key") or os.path.splitext(os.path.basename(path))[0]
                aliases = entry.get("aliases") or []
                orig = os.path.basename(path)
                # ensure canonical present as first alias (but do not duplicate)
                aliases = [a for a in aliases if a]  # filter empties
                if canonical and canonical not in aliases:
                    aliases.insert(0, canonical)
                # register labels (avoid duplicates)
                for alias in aliases:
                    label = f"{alias} — {orig}"
                    existing = st.session_state["csv_label_to_path"].get(label)
                    if not existing or existing == path:
                        st.session_state["csv_label_to_path"][label] = path
                # register path->canonical mapping (authoritative from store)
                st.session_state["csv_path_to_table"][path] = canonical
            except Exception:
                logger.exception("Failed to register schema_store entry %s", entry)
        logger.info("Session mappings synced from SchemaStore (entries=%d)", len(store_meta))
    except Exception:
        logger.exception("Failed to sync session mappings from SchemaStore")


# ---------------------------
# Small helpers used during upload/registration
# ---------------------------
def _find_store_meta_by_path(path: str) -> Optional[Dict[str, Any]]:
    """
    Return the SchemaStore metadata entry for a given absolute path (if present),
    otherwise None. Uses list_csvs_meta() for simplicity and compatibility.
    """
    if not schema_store:
        return None
    try:
        for e in schema_store.list_csvs_meta() or []:
            p = e.get("path") or ""
            if not p:
                continue
            # normalize absolute paths for comparison
            try:
                if os.path.abspath(p) == os.path.abspath(path):
                    return e
            except Exception:
                if p == path:
                    return e
    except Exception:
        logger.exception("Failed to search SchemaStore for path %s", path)
    return None


def _ensure_label_registered(label: str, path: str):
    """
    Register a UI label -> path mapping, but only if it doesn't conflict with
    an existing mapping to a different path.
    """
    cmap = st.session_state.setdefault("csv_label_to_path", {})
    existing = cmap.get(label)
    if existing and existing != path:
        # conflicting label already present — don't overwrite
        logger.debug("Label %s already maps to different path (%s); skipping registration for %s", label, existing, path)
        return
    cmap[label] = path


# ---------------------------
# Top-level UI
# ---------------------------
try:
    st.title("📊 Text-to-SQL Bot — Conversation Mode")

    # Freshen conversations meta at startup
    if history_store and not st.session_state.get("conversations_meta"):
        try:
            st.session_state.conversations_meta = history_store.list_conversations()
        except Exception:
            logger.exception("Initial list_conversations failed")
            st.session_state.conversations_meta = []

    # ---------------------------
    # Sidebar: Conversations + CSV Management + Schema Viewer
    # ---------------------------
    with st.sidebar:
        st.header("Conversations")

        # Search/filter
        q = st.text_input("Search conversations", value="")

        # New conversation
        if st.button("➕ New Conversation"):
            if not history_store:
                st.error("History backend not available; cannot create conversations.")
            else:
                default_name = f"Conversation {len(st.session_state.conversations_meta) + 1}"
                try:
                    conv = history_store.create_conversation(name=default_name)
                    st.session_state.conversations_meta = history_store.list_conversations()
                    st.session_state.selected_conversation_id = conv["id"]
                    st.session_state.conversation_cache[conv["id"]] = conv
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

        for c in convs:
            name = c.get("name") or "Untitled"
            snippet = c.get("last_message_snippet") or ""
            display_label = name if not snippet else f"{name} — {snippet}"
            if st.button(display_label, key=f"select_{c['id']}"):
                st.session_state.selected_conversation_id = c["id"]
                if history_store:
                    try:
                        st.session_state.conversation_cache[c["id"]] = history_store.get_conversation(c["id"])
                    except Exception:
                        logger.exception("Failed to load conversation %s", c["id"])
                        st.error("Failed to load conversation from storage.")
                _safe_rerun()

        st.markdown("---")
        st.subheader("CSV Management")
        uploaded_files = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
        if uploaded_files and csv_loader:
            for file in uploaded_files:
                try:
                    saved_path = csv_loader.save_csv(file)
                    st.success(f"Uploaded {file.name}")
                    logger.info("Saved %s -> %s", file.name, saved_path)

                    # try to extract metadata and register schema
                    try:
                        metadata_list = csv_loader.load_and_extract([saved_path])
                        if metadata_list and isinstance(metadata_list, list):
                            for meta in metadata_list:
                                orig_name = os.path.basename(saved_path)
                                canonical = None
                                aliases = []
                                if isinstance(meta, dict):
                                    canonical = meta.get("canonical_name") or meta.get("table_name") or os.path.splitext(os.path.basename(saved_path))[0]
                                    aliases = meta.get("aliases") or meta.get("alias") or []
                                    orig_name = meta.get("original_name") or orig_name
                                if not canonical:
                                    canonical = os.path.splitext(os.path.basename(saved_path))[0]

                                # Check whether this path is already present in SchemaStore
                                existing_meta = _find_store_meta_by_path(saved_path)
                                if schema_store and not existing_meta:
                                    # Persist schema trying with provided canonical & aliases
                                    try:
                                        store_key = schema_store.add_csv(saved_path, csv_name=canonical, aliases=aliases)
                                        logger.info("schema_store.add_csv returned key=%s for path=%s", store_key, saved_path)
                                        # read back authoritative meta for this path
                                        existing_meta = _find_store_meta_by_path(saved_path)
                                    except Exception:
                                        logger.exception("schema_store.add_csv failed for %s (canonical=%s aliases=%s)", saved_path, canonical, aliases)
                                        # fallback: attempt once without names (idempotent in SchemaStore)
                                        try:
                                            schema_store.add_csv(saved_path)
                                            existing_meta = _find_store_meta_by_path(saved_path)
                                        except Exception:
                                            logger.exception("schema_store.add_csv fallback failed for %s", saved_path)
                                else:
                                    if existing_meta:
                                        logger.debug("Path already registered in SchemaStore: %s", saved_path)

                                # Determine canonical to use in UI mapping (prefer authoritative store meta)
                                ui_canonical = None
                                if existing_meta and isinstance(existing_meta, dict):
                                    ui_canonical = existing_meta.get("canonical") or existing_meta.get("friendly") or os.path.splitext(os.path.basename(saved_path))[0]
                                    aliases_from_store = existing_meta.get("aliases") or []
                                    # ensure we use aliases from store if available
                                    if aliases_from_store:
                                        aliases = aliases_from_store
                                else:
                                    ui_canonical = canonical

                                # ensure canonical present in aliases list and register UI labels
                                aliases = [a for a in (aliases or []) if a]
                                if ui_canonical and ui_canonical not in aliases:
                                    aliases.insert(0, ui_canonical)
                                for alias in aliases:
                                    if not alias:
                                        continue
                                    label = f"{alias} — {orig_name}"
                                    _ensure_label_registered(label, saved_path)

                                # store path->canonical mapping (prefer authoritative)
                                st.session_state["csv_path_to_table"][saved_path] = ui_canonical
                                logger.info("Registered schema for %s (canonical=%s, aliases=%s)", saved_path, ui_canonical, aliases)
                    except Exception:
                        logger.exception("Failed to auto-register schema for uploaded CSV")
                except Exception as e:
                    st.error(f"Failed to save {file.name}: {e}")
                    logger.exception("CSV upload failed")

        # One-time sync from SchemaStore to session maps (idempotent)
        try:
            _sync_session_mappings_from_store()
        except Exception:
            logger.exception("Failed to read schema_store metadata")

        # Defensive fallback: include raw uploaded files (older behaviour)
        try:
            uploaded_paths = csv_loader.list_uploaded_csvs() if csv_loader else []
        except Exception:
            logger.exception("Failed to list uploaded CSVs")
            uploaded_paths = []

        for p in uploaded_paths:
            try:
                path = p if isinstance(p, str) else (p.get("path") or p.get("filepath") or p.get("file") or p.get("filename") or "")
                if not path:
                    continue
                # If the path is already known from SchemaStore or previous mappings, skip
                if path in st.session_state["csv_path_to_table"]:
                    continue
                table = os.path.splitext(os.path.basename(path))[0]
                orig = os.path.basename(path)
                label = f"{table} — {orig}"
                if label not in st.session_state["csv_label_to_path"]:
                    st.session_state["csv_label_to_path"][label] = path
                if path not in st.session_state["csv_path_to_table"]:
                    st.session_state["csv_path_to_table"][path] = table
            except Exception:
                logger.exception("Failed to register fallback uploaded path %s", p)

        csv_display_options = list(st.session_state["csv_label_to_path"].keys())
        csv_display_options.sort()

        selected_labels = st.multiselect(
            "Select CSVs for context (used when sending messages)",
            options=csv_display_options,
            default=[],
            key="selected_csvs_sidebar",
        )

        st.markdown("---")

        # Schema viewer
        with st.expander("Schema viewer (inspect canonical names, aliases, columns)"):
            try:
                if not schema_store:
                    st.info("SchemaStore not available. Upload CSVs to populate schema store.")
                else:
                    store_meta = schema_store.list_csvs_meta() or []
                    if not store_meta:
                        st.info("No schemas available. Upload CSVs to populate the SchemaStore.")
                    else:
                        for entry in store_meta:
                            try:
                                key = entry.get("key") or "<key>"
                                canonical = entry.get("canonical") or entry.get("friendly") or key
                                path = entry.get("path") or ""
                                columns = entry.get("columns") or []
                                aliases = entry.get("aliases") or []
                                sample_rows = entry.get("sample_rows") or []
                                header = f"{canonical}  —  {os.path.basename(path) if path else key}"
                                with st.expander(header):
                                    st.write("Store key:", key)
                                    st.write("Canonical:", canonical)
                                    st.write("Path:", path)
                                    if aliases:
                                        st.write("Aliases:", ", ".join(aliases))
                                    if columns:
                                        st.write("Columns:")
                                        st.write(columns)
                                    if sample_rows:
                                        st.write("Sample rows (up to 3):")
                                        try:
                                            for r in sample_rows[:3]:
                                                st.write(r)
                                        except Exception:
                                            st.write(sample_rows[:3])
                            except Exception:
                                logger.exception("Failed to render schema entry %s", entry)

                if st.button("Sync schemas to UI (refresh labels)"):
                    try:
                        _sync_session_mappings_from_store()
                        st.success("Synced schema mappings from SchemaStore.")
                        _safe_rerun()
                    except Exception:
                        logger.exception("Schema sync failed")
                        st.error("Schema sync failed; check logs.")
            except Exception:
                logger.exception("Failed to display Schema viewer")
                st.write("Could not load schema metadata.")

        st.markdown("---")
        if st.button("Refresh Conversations"):
            if history_store:
                try:
                    st.session_state.conversations_meta = history_store.list_conversations()
                except Exception:
                    logger.exception("Failed to refresh conversations_meta")
                    st.error("Failed to refresh conversations list.")
            else:
                st.error("History backend unavailable.")
            _safe_rerun()

        st.caption("Tip: create a conversation first, then ask questions in the main area.")

    # ---------------------------
    # Main area: Chat view
    # ---------------------------
    col1, col2 = st.columns([3, 1])
    with col1:
        # Ensure selected conversation loaded
        if st.session_state.selected_conversation_id:
            conv_id = st.session_state.selected_conversation_id
            if conv_id not in st.session_state.conversation_cache and history_store:
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
            if name_box != conv.get("name") and history_store:
                try:
                    ok = history_store.update_conversation_name(conv["id"], name_box)
                    if ok:
                        st.session_state.conversation_cache[conv["id"]]["name"] = name_box
                        st.session_state.conversations_meta = history_store.list_conversations()
                        st.success("Renamed")
                except Exception:
                    st.error("Rename failed; see logs.")

            if header_cols[1].button("Export"):
                if history_store:
                    try:
                        path = history_store.export_conversation(conv["id"])
                        st.success(f"Exported to {path}")
                    except Exception:
                        logger.exception("Export failed")
                        st.error("Export failed; see logs.")
                else:
                    st.error("History backend unavailable.")

            delete_flag_key = f"confirm_delete_{conv.get('id')}"
            if header_cols[2].button("Delete"):
                st.session_state[delete_flag_key] = True

            if st.session_state.get(delete_flag_key):
                cf_col1, cf_col2 = st.columns([1, 1])
                if cf_col1.button("Confirm Delete"):
                    if history_store:
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
                    else:
                        st.error("History backend unavailable.")
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

            # Input area
            clear_next = st.session_state.pop("__clear_user_input", False)
            initial_user_input = "" if clear_next else st.session_state.get("user_input", "")

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
                    if history_store:
                        try:
                            history_store.append_message(conv["id"], user_msg)
                        except Exception:
                            st.error("Failed to append user message; see logs.")
                            logger.exception("append user message failed")
                    else:
                        logger.debug("HistoryStore not configured; skipping append_message for user_msg.")

                    # Resolve selected CSV labels -> canonical table names
                    selected_labels = st.session_state.get("selected_csvs_sidebar", []) or []
                    label_to_path = st.session_state.get("csv_label_to_path", {}) or {}
                    path_to_table = st.session_state.get("csv_path_to_table", {}) or {}

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

                    if not graph:
                        st.error("Agent unavailable. Check server logs.")
                        result = {"error": "agent unavailable"}
                    else:
                        with st.spinner("Generating answer..."):
                            try:
                                # graph.run expected signature: graph.run(prompt, table_names, run_query=bool)
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
                    if history_store:
                        try:
                            history_store.append_message(conv["id"], assistant_msg)
                            # refresh conversation in cache & meta list
                            st.session_state.conversation_cache[conv["id"]] = history_store.get_conversation(conv["id"])
                            st.session_state.conversations_meta = history_store.list_conversations()
                            st.session_state["__clear_user_input"] = True
                            _safe_rerun()
                        except Exception:
                            st.error("Failed to persist assistant message; see logs.")
                            logger.exception("append assistant message failed")
                    else:
                        logger.debug("HistoryStore not configured; skipping append_message for assistant_msg.")
                        # still update in-memory cache so UI shows result immediately
                        cache_conv = st.session_state.conversation_cache.get(conv.get("id"), {"messages": []})
                        cache_conv["messages"] = cache_conv.get("messages", []) + [user_msg, assistant_msg]
                        st.session_state.conversation_cache[conv.get("id")] = cache_conv
                        st.session_state["__clear_user_input"] = True
                        _safe_rerun()

    with col2:
        st.subheader("Conversation Tools")
        if st.button("Refresh conversation list"):
            if history_store:
                try:
                    st.session_state.conversations_meta = history_store.list_conversations()
                except Exception:
                    logger.exception("Failed to refresh conversations_meta")
                    st.error("Failed to refresh conversations list.")
            else:
                st.error("History backend unavailable.")
            _safe_rerun()

        st.markdown("Export / Import")
        if st.button("Export all conversations"):
            try:
                all_meta = st.session_state.conversations_meta or []
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
        if history_store:
            st.session_state.conversations_meta = history_store.list_conversations()
    except Exception:
        logger.exception("Failed to refresh conversations_meta at end of request")

except Exception as e:
    logger.exception("Top-level error in app.py")
    st.error(f"An internal error occurred while rendering the app: {e}")
