# D:\text_to_sql_bot\app.py
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
import traceback
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
            logger.debug("experimental_rerun failed, marking session for rerun")
            return
    else:
        st.session_state["_needs_rerun"] = True
        logger.debug("experimental_rerun not available, setting _needs_rerun")
        return


def _render_exception_ui(exc: Exception, hint: Optional[str] = None) -> None:
    """
    Show a friendly error message in UI plus an expandable detailed stack trace.
    """
    msg = f"{type(exc).__name__}: {str(exc)}"
    if hint:
        st.error(f"{hint} — {msg}")
    else:
        st.error(msg)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    with st.expander("Show details"):
        st.code(tb)


# ---------------------------
# Initialize core components (defensive)
# ---------------------------
logger.info("Starting app initialization")

# Config module (keep as module for backward compatibility)
config = config_module

# CSV loader
csv_loader = None
try:
    csv_loader = CSVLoader()
    logger.info("CSVLoader initialized at upload_dir=%s", getattr(csv_loader, "upload_dir", "<unknown>"))
except Exception as e:
    logger.exception("Failed to initialize CSVLoader; CSV features may be limited.")
    csv_loader = None

# Schema store
schema_store = None
try:
    schema_store = SchemaStore()
    logger.info("SchemaStore initialized with store_path=%s", getattr(schema_store, "store_path", "<unknown>"))
except Exception as e:
    logger.exception("Failed to initialize SchemaStore; schema features may be limited.")
    schema_store = None

# VectorSearch (optional: may depend on faiss/numpy)
vector_search = None
try:
    vector_search = VectorSearch()
    logger.info("VectorSearch initialized (dim=%s)", getattr(vector_search, "dim", "<unknown>"))
except Exception as e:
    logger.exception("Failed to initialize VectorSearch; vector features may be limited.")
    vector_search = None

# Graph builder (agent)
graph = None
try:
    # GraphBuilder.build_default may rely on other components; wrap to avoid crash
    graph = GraphBuilder.build_default()
    logger.info("GraphBuilder built successfully")
except Exception as e:
    logger.exception("Failed to build default GraphBuilder. Agent/graph features will be unavailable.")
    graph = None

# Persistent conversation-first history store (SQLite)
history_store = None
try:
    history_store = HistoryStore()
    logger.info("HistoryStore initialized")
except Exception as e:
    logger.exception("Failed to initialize HistoryStore; conversation persistence will be disabled.")
    history_store = None

logger.info("App initialization complete")


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
            logger.debug("Performed delayed experimental_rerun")
    except Exception:
        logger.debug("Delayed rerun failed")


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
    logger.debug("Syncing session mappings from SchemaStore")
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
                    logger.debug("Skipping schema entry with no path: %s", entry)
                    continue
                canonical = entry.get("canonical") or entry.get("friendly") or entry.get("key") or os.path.splitext(os.path.basename(path))[0]
                aliases = entry.get("aliases") or []
                orig = os.path.basename(path)
                aliases = [a for a in aliases if a]
                if canonical and canonical not in aliases:
                    aliases.insert(0, canonical)
                for alias in aliases:
                    label = f"{alias} — {orig}"
                    existing = st.session_state["csv_label_to_path"].get(label)
                    if not existing or existing == path:
                        st.session_state["csv_label_to_path"][label] = path
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
        logger.debug("Label %s already maps to different path (%s); skipping registration for %s", label, existing, path)
        return
    cmap[label] = path
    logger.debug("Registered label '%s' -> %s", label, path)


# ---------------------------
# Top-level UI
# ---------------------------
try:
    st.title("📊 Text-to-SQL Bot — Conversation Mode")

    # Freshen conversations meta at startup
    if history_store and not st.session_state.get("conversations_meta"):
        try:
            st.session_state.conversations_meta = history_store.list_conversations()
            logger.debug("Loaded conversations_meta (count=%d)", len(st.session_state.conversations_meta))
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
                logger.warning("New conversation requested but HistoryStore unavailable")
            else:
                default_name = f"Conversation {len(st.session_state.conversations_meta) + 1}"
                try:
                    conv = history_store.create_conversation(name=default_name)
                    st.session_state.conversations_meta = history_store.list_conversations()
                    st.session_state.selected_conversation_id = conv["id"]
                    st.session_state.conversation_cache[conv["id"]] = conv
                    logger.info("Created conversation id=%s name=%s", conv["id"], default_name)
                    _safe_rerun()
                except Exception as e:
                    logger.exception("Failed to create new conversation")
                    _render_exception_ui(e, hint="Could not create a new conversation")

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
                        logger.debug("Loaded conversation %s into cache", c["id"])
                    except Exception as e:
                        logger.exception("Failed to load conversation %s", c["id"])
                        _render_exception_ui(e, hint="Failed to load conversation from storage")
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
                        t0 = time.time()
                        metadata_list = csv_loader.load_and_extract([saved_path])
                        logger.debug("Extracted metadata for %s in %.3fs", saved_path, time.time() - t0)
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

                                existing_meta = _find_store_meta_by_path(saved_path)
                                if schema_store and not existing_meta:
                                    try:
                                        t_start = time.time()
                                        store_key = schema_store.add_csv(saved_path, csv_name=canonical, aliases=aliases)
                                        logger.info("schema_store.add_csv returned key=%s for path=%s (t=%.3fs)", store_key, saved_path, time.time()-t_start)
                                        existing_meta = _find_store_meta_by_path(saved_path)
                                    except Exception as e:
                                        logger.exception("schema_store.add_csv failed for %s (canonical=%s aliases=%s)", saved_path, canonical, aliases)
                                        try:
                                            logger.info("Attempting fallback schema_store.add_csv without names for %s", saved_path)
                                            schema_store.add_csv(saved_path)
                                            existing_meta = _find_store_meta_by_path(saved_path)
                                        except Exception:
                                            logger.exception("schema_store.add_csv fallback also failed for %s", saved_path)
                                else:
                                    if existing_meta:
                                        logger.debug("Path already registered in SchemaStore: %s", saved_path)

                                # Determine canonical to use in UI mapping (prefer authoritative store meta)
                                ui_canonical = None
                                if existing_meta and isinstance(existing_meta, dict):
                                    ui_canonical = existing_meta.get("canonical") or existing_meta.get("friendly") or os.path.splitext(os.path.basename(saved_path))[0]
                                    aliases_from_store = existing_meta.get("aliases") or []
                                    if aliases_from_store:
                                        aliases = aliases_from_store
                                else:
                                    ui_canonical = canonical

                                aliases = [a for a in (aliases or []) if a]
                                if ui_canonical and ui_canonical not in aliases:
                                    aliases.insert(0, ui_canonical)
                                for alias in aliases:
                                    if not alias:
                                        continue
                                    label = f"{alias} — {orig_name}"
                                    _ensure_label_registered(label, saved_path)

                                st.session_state["csv_path_to_table"][saved_path] = ui_canonical
                                logger.info("Registered schema for %s (canonical=%s, aliases=%s)", saved_path, ui_canonical, aliases)

                                # === Auto-select newly uploaded CSV label in sidebar multiselect ===
                                try:
                                    if aliases:
                                        new_label = f"{aliases[0]} — {orig_name}"
                                        selected = st.session_state.get("selected_csvs_sidebar", []) or []
                                        if new_label not in selected:
                                            selected.append(new_label)
                                            st.session_state["selected_csvs_sidebar"] = selected
                                            logger.info("Auto-selected uploaded CSV in sidebar multiselect: %s", new_label)
                                    # request a rerun so the multiselect updates immediately
                                    _safe_rerun()
                                except Exception:
                                    logger.exception("Failed to auto-select uploaded csv label in UI")

                    except Exception as e:
                        logger.exception("Failed to auto-register schema for uploaded CSV")
                        _render_exception_ui(e, hint=f"Failed to parse or register schema for {file.name}")
                except Exception as e:
                    logger.exception("CSV upload failed for %s", getattr(file, "name", "<unknown>"))
                    _render_exception_ui(e, hint=f"Failed to save {getattr(file, 'name', '<file>')}")

        # One-time sync from SchemaStore to session maps (idempotent)
        try:
            _sync_session_mappings_from_store()
        except Exception:
            logger.exception("Failed to read schema_store metadata")

        # Defensive fallback: include raw uploaded files (older behavior)
        uploaded_paths = []
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
                        t0 = time.time()
                        _sync_session_mappings_from_store()
                        st.success("Synced schema mappings from SchemaStore.")
                        logger.info("Manual sync from SchemaStore completed in %.3fs", time.time() - t0)
                        _safe_rerun()
                    except Exception as e:
                        logger.exception("Schema sync failed")
                        _render_exception_ui(e, hint="Schema sync failed")
            except Exception as e:
                logger.exception("Failed to display Schema viewer")
                _render_exception_ui(e, hint="Could not load schema metadata")

        st.markdown("---")
        if st.button("Refresh Conversations"):
            if history_store:
                try:
                    st.session_state.conversations_meta = history_store.list_conversations()
                    logger.debug("Refreshed conversations_meta")
                except Exception as e:
                    logger.exception("Failed to refresh conversations_meta")
                    _render_exception_ui(e, hint="Failed to refresh conversations list")
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
                    logger.debug("Populated conversation cache for %s", conv_id)
                except Exception as e:
                    logger.exception("Failed to load conversation. It may have been deleted.")
                    _render_exception_ui(e, hint="Failed to load conversation")
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
                        logger.info("Renamed conversation %s -> %s", conv["id"], name_box)
                except Exception as e:
                    logger.exception("Rename failed")
                    _render_exception_ui(e, hint="Rename failed")

            if header_cols[1].button("Export"):
                if history_store:
                    try:
                        path = history_store.export_conversation(conv["id"])
                        st.success(f"Exported to {path}")
                        logger.info("Exported conversation %s to %s", conv["id"], path)
                    except Exception as e:
                        logger.exception("Export failed")
                        _render_exception_ui(e, hint="Export failed")
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
                                logger.info("Deleted conversation %s", conv["id"])
                                _safe_rerun()
                            else:
                                st.error("Conversation not found or could not be deleted.")
                        except Exception as e:
                            logger.exception("Failed to delete conversation %s", conv.get("id"))
                            _render_exception_ui(e, hint="Delete failed")
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
                                    logger.exception("Failed to render message meta JSON")
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
                            logger.debug("Appended user message to history for conv=%s", conv["id"])
                        except Exception as e:
                            logger.exception("append user message failed")
                            _render_exception_ui(e, hint="Failed to append user message")
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

                    # === Guard: do not call agent when no table selected ===
                    if not selected_table_names:
                        try:
                            # If there is exactly one available label, auto-select it for convenience.
                            available_labels = list(st.session_state.get("csv_label_to_path", {}).keys())
                            if len(available_labels) == 1:
                                auto_label = available_labels[0]
                                st.session_state["selected_csvs_sidebar"] = [auto_label]
                                # compute table name from path mapping
                                p = st.session_state["csv_label_to_path"].get(auto_label)
                                t = st.session_state.get("csv_path_to_table", {}).get(p) if p else None
                                if t:
                                    selected_table_names = [t]
                                else:
                                    selected_table_names = [os.path.splitext(os.path.basename(p))[0]] if p else []
                                st.warning("No CSV was selected — automatically using the only uploaded CSV for context.")
                                logger.info("Auto-injected single available CSV label into selected_csvs_sidebar: %s", auto_label)
                                # proceed to agent run with injected selection
                            else:
                                # Provide a friendly assistant message saying selection is required and abort the run.
                                st.warning("No CSV selected. Please select one or more CSVs from the left sidebar before sending the query.")
                                logger.warning("User attempted to run graph with no selected tables; aborting run.")
                                # Build assistant message explaining what to do and persist it.
                                assistant_content = "Please select a CSV in the left sidebar for schema context and try again."
                                assistant_msg = {"role": "assistant", "content": assistant_content, "meta": {"error": "no_table_selected"}}
                                if history_store:
                                    try:
                                        history_store.append_message(conv["id"], assistant_msg)
                                        st.session_state.conversation_cache[conv["id"]] = history_store.get_conversation(conv["id"])
                                        st.session_state.conversations_meta = history_store.list_conversations()
                                    except Exception:
                                        logger.exception("Failed to append assistant message for no_table_selected")
                                else:
                                    cache_conv = st.session_state.conversation_cache.get(conv.get("id"), {"messages": []})
                                    cache_conv["messages"] = cache_conv.get("messages", []) + [user_msg, assistant_msg]
                                    st.session_state.conversation_cache[conv.get("id")] = cache_conv
                                _safe_rerun()
                                # Abort this run (prevent calling the agent with no schema)
                                st.stop()
                        except Exception:
                            logger.exception("Guard for missing selected tables failed")
                            st.stop()

                    if not graph:
                        st.error("Agent unavailable. Check server logs.")
                        result = {"error": "agent unavailable"}
                        logger.warning("User requested agent run but graph is unavailable")
                    else:
                        with st.spinner("Generating answer..."):
                            try:
                                t0 = time.time()
                                # Helpful debug: log session mappings if selected_table_names is empty (should not be here)
                                if not selected_table_names:
                                    logger.debug("No selected_table_names at run time. csv_label_to_path keys=%s, csv_path_to_table=%s",
                                                 list(st.session_state.get("csv_label_to_path", {}).keys()),
                                                 st.session_state.get("csv_path_to_table", {}))
                                logger.info("Running graph.run for conversation=%s tables=%s", conv.get("id"), selected_table_names)
                                # graph.run expected signature: graph.run(prompt, table_names, run_query=bool)
                                result = graph.run(user_input.strip(), selected_table_names, run_query=execute_sql_checkbox)
                                duration = time.time() - t0
                                logger.info("Graph run completed in %.3fs", duration)
                            except Exception as e:
                                logger.exception("Agent run failed")
                                result = {"error": str(e)}
                                _render_exception_ui(e, hint="Agent run failed")

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
                    except Exception as e:
                        logger.exception("Failed to sanitize result for history")
                        sanitized_meta = {"error": "failed to sanitize meta"}

                    if not isinstance(sanitized_meta, dict):
                        sanitized_meta = {"_raw": sanitized_meta}

                    assistant_msg = {"role": "assistant", "content": assistant_content, "meta": sanitized_meta}
                    if history_store:
                        try:
                            history_store.append_message(conv["id"], assistant_msg)
                            logger.debug("Appended assistant message for conv=%s", conv["id"])
                            # refresh conversation in cache & meta list
                            st.session_state.conversation_cache[conv["id"]] = history_store.get_conversation(conv["id"])
                            st.session_state.conversations_meta = history_store.list_conversations()
                            st.session_state["__clear_user_input"] = True
                            _safe_rerun()
                        except Exception as e:
                            logger.exception("append assistant message failed")
                            _render_exception_ui(e, hint="Failed to persist assistant message")
                    else:
                        logger.debug("HistoryStore not configured; skipping append_message for assistant_msg.")
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
                    logger.debug("Refreshed conversations_meta via UI button")
                except Exception as e:
                    logger.exception("Failed to refresh conversations_meta")
                    _render_exception_ui(e, hint="Failed to refresh conversations list")
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
                logger.info("Exported all conversations to %s", out_path)
            except Exception as e:
                logger.exception("Failed to export all conversations")
                _render_exception_ui(e, hint="Export failed")
        st.markdown("---")
        st.info("Developer: raw logs are available in the server console. Use 'Show details' in message to reveal SQL/raw LLM output.")

    # ---------------------------
    # Ensure sessions meta up-to-date on exit
    # ---------------------------
    try:
        if history_store:
            st.session_state.conversations_meta = history_store.list_conversations()
            logger.debug("Refreshed conversations_meta at end of request")
    except Exception:
        logger.exception("Failed to refresh conversations_meta at end of request")

except Exception as e:
    logger.exception("Top-level error in app.py")
    try:
        _render_exception_ui(e, hint="An internal error occurred while rendering the app")
    except Exception:
        # If rendering the UI fails, at least log and raise so server logs capture it
        logger.exception("Also failed to render exception to UI")
        raise
