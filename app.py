# app.py
import streamlit as st
import os
import json
from typing import List, Dict, Any

from app.csv_loader import CSVLoader
from app.schema_store import SchemaStore
from app.vector_search import VectorSearch
from app.graph.builder import GraphBuilder
from app.history_sql import HistoryStore
from app.logger import get_logger
import app.config as config_module

logger = get_logger("app")


def _sanitize_for_history(obj: Any) -> Any:
    """
    Convert obj into a JSON-friendly structure:
      - keep primitives (str/int/float/bool/None)
      - convert datetime/date to isoformat (if present)
      - recursively sanitize dicts, lists, tuples
      - convert other objects to str(obj)

    This is intentionally conservative: it's OK to lose complex internals for persistence.
    """
    # import locally to avoid top-level import dependencies
    from datetime import datetime, date

    # primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # datetime-like
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # dict -> sanitize items
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            try:
                sanitized[k] = _sanitize_for_history(v)
            except Exception:
                sanitized[k] = str(v)
        return sanitized

    # list/tuple/set -> sanitize elements
    if isinstance(obj, (list, tuple, set)):
        try:
            return [_sanitize_for_history(v) for v in obj]
        except Exception:
            return [str(v) for v in obj]

    # attempt to JSON-dump unknown objects with default=str
    try:
        json.dumps(obj, default=str)
        # if dumps didn't raise, return str(obj) to keep content stable (avoid complex types)
        return str(obj)
    except Exception:
        pass

    # fallback: string representation
    try:
        return str(obj)
    except Exception:
        return repr(obj)


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
except Exception as e:
    logger.exception("Failed to build default GraphBuilder. Check node implementations.")
    graph = None

# Persistent history store (JSON by default, SQLite optional)
history_store = HistoryStore(backend=getattr(config, "HISTORY_BACKEND", "json"))

# Streamlit session state for chat history
if "chat_history" not in st.session_state:
    # Try to seed session from persisted history so users see existing items
    try:
        persisted = history_store.list_entries()
        # Keep the persisted structure as-is for display, don't attempt to rehydrate complex objects
        st.session_state.chat_history = persisted or []
    except Exception:
        st.session_state.chat_history = []

st.set_page_config(page_title="Text-to-SQL Bot", layout="wide")
st.title("📊 Text-to-SQL Bot")

# ---------------------------
# Sidebar: CSV Management
# ---------------------------
st.sidebar.header("CSV Management")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSVs", type=["csv"], accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        try:
            saved_path = csv_loader.save_csv(file)
            st.sidebar.success(f"Uploaded {file.name}")
            logger.info(f"Saved {file.name} -> {saved_path}")
        except Exception as e:
            st.sidebar.error(f"Failed to save {file.name}: {e}")
            logger.exception("CSV upload failed")

# List uploaded CSVs
try:
    all_csvs: List[str] = csv_loader.list_uploaded_csvs()
except Exception:
    logger.exception("Failed to list uploaded CSVs")
    all_csvs = []

selected_csvs = st.sidebar.multiselect(
    "Select CSVs for context", options=all_csvs, default=all_csvs
)

load_button = st.sidebar.button("Load Selected CSVs")
if load_button and selected_csvs:
    try:
        schemas = csv_loader.load_and_extract(selected_csvs)
        for meta in schemas:
            path = meta.get("path")
            name = meta.get("table_name") or os.path.basename(path)
            schema_store.add_csv(path, csv_name=name)

        # Index schemas for vector search (RAG)
        if hasattr(vector_search, "index_schemas"):
            try:
                vector_search.index_schemas(schemas)
            except Exception:
                logger.exception("vector_search.index_schemas failed; continuing")

        st.sidebar.success("Schemas loaded and vector index updated!")
    except Exception as e:
        st.sidebar.error(f"Failed to load selected CSVs: {e}")
        logger.exception("Load selected CSVs failed")

# ---------------------------
# Main area: Query input
# ---------------------------
st.header("Ask a Question in Natural Language")
user_query = st.text_area("Enter your question here:", height=120)
default_name = f"Query {len(st.session_state.chat_history) + 1}"
query_name = st.text_input("Optional: Give this query a name", value=default_name)
run_query = st.checkbox("Execute SQL if valid?", value=True)
execute_button = st.button("Run Query")

# ---------------------------
# Execute / Generate SQL
# ---------------------------
if execute_button and user_query:
    if not selected_csvs:
        st.error("Select at least one CSV in the sidebar for context.")
    elif graph is None:
        st.error("Agent pipeline unavailable. Check server logs.")
    else:
        with st.spinner("Generating SQL..."):
            try:
                result = graph.run(user_query, selected_csvs, run_query=run_query)
            except Exception as e:
                st.error(f"Error running agent: {e}")
                logger.exception("Agent run failed")
                result = {"error": str(e)}

        # Prepare a sanitized copy for persistent storage (so history JSON / sqlite won't fail)
        try:
            sanitized_result = _sanitize_for_history(result)
            entry = history_store.add_entry(name=query_name, query=user_query, result=sanitized_result)
            # keep full (non-sanitized) result in session for UI, but store persisted metadata from entry
            session_entry = {
                "id": entry.get("id"),
                "name": entry.get("name"),
                "query": entry.get("query"),
                "result": result,
                "created_at": entry.get("created_at"),
                "updated_at": entry.get("updated_at"),
            }
            st.session_state.chat_history.append(session_entry)
        except Exception:
            # if persistent storage fails for any reason, still keep session-only copy (unsanitized)
            logger.exception("Failed to persist history entry; using session-only storage")
            st.session_state.chat_history.append({
                "id": None,
                "name": query_name,
                "query": user_query,
                "result": result,
            })

# ---------------------------
# Chat History UI
# ---------------------------
if st.session_state.chat_history:
    st.subheader("💬 Query History")
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        if st.button("Clear History (memory only)"):
            st.session_state.chat_history = []
            # no explicit rerun call - button click already triggers a rerun
    with col_b:
        if st.button("Export History JSON (persisted)"):
            try:
                path = history_store.export_json()
                st.success(f"Exported history to {path}")
            except Exception:
                st.exception("Failed to export history")
    with col_c:
        st.info(f"{len(st.session_state.chat_history)} queries stored in this session")

    # Render query history (robust: stable keys, per-entry error handling)
    for display_idx, chat in enumerate(reversed(st.session_state.chat_history)):
        # compute a stable index into session list
        real_index = len(st.session_state.chat_history) - 1 - display_idx
        # prefer stable id for keys (use index only when id is missing)
        entry_id = chat.get("id") or f"idx_{real_index}"
        disp_name = chat.get("name", f"Query {display_idx+1}")

        # keys for inner widgets still used to keep widget identity stable
        rename_key = f"rename_{entry_id}"
        delete_key = f"del_{entry_id}"

        try:
            # NOTE: Streamlit 1.51.0 does not accept a 'key' arg on st.expander in your environment,
            # so we avoid passing it here. The inner widgets (text_input, button) keep stable keys.
            with st.expander(f"{disp_name}", expanded=False):
                # Rename input (safe: unique key per entry)
                try:
                    new_name = st.text_input("Rename query", value=disp_name, key=rename_key)
                    if new_name and new_name != disp_name:
                        # update session state
                        st.session_state.chat_history[real_index]["name"] = new_name
                        # attempt to update persisted entry if available
                        if chat.get("id"):
                            try:
                                updated = history_store.update_entry(chat["id"], name=new_name)
                                if updated:
                                    st.session_state.chat_history[real_index].update({
                                        "updated_at": updated.get("updated_at")
                                    })
                            except Exception:
                                logger.exception("Failed to update persisted history entry (rename)")
                except Exception:
                    logger.exception("Rename widget failed for entry %s", entry_id)
                    st.warning("Rename unavailable for this entry.")

                # Main content
                try:
                    st.markdown(f"**User Query:** {chat.get('query')}")
                except Exception:
                    st.markdown("**User Query:** (could not render)")
                res = chat.get("result") or {}
                if not res:
                    st.info("No result stored for this query.")
                    continue

                # SQL
                try:
                    st.markdown("**Generated SQL:**")
                    st.code(res.get("sql") or "No SQL generated.", language="sql")
                    st.write(f"Valid SQL: {res.get('valid')}")
                except Exception:
                    logger.exception("Failed to render SQL for entry %s", entry_id)
                    st.error("Failed to render SQL for this entry.")

                # Execution results (dataframe)
                if res.get("execution"):
                    try:
                        st.markdown("**Execution Results:**")
                        exec_res = res["execution"]
                        rows = exec_res.get("rows") if isinstance(exec_res, dict) else exec_res
                        try:
                            st.dataframe(rows)
                        except Exception:
                            # fallback to plain write if dataframe fails
                            st.write(rows)
                        # metadata may not exist
                        if isinstance(exec_res, dict):
                            st.write("Metadata:", exec_res.get("meta", {}))
                    except Exception:
                        logger.exception("Failed to render execution results for entry %s", entry_id)
                        st.error("Failed to render execution results for this entry.")

                # Formatted output / explanation
                if res.get("formatted"):
                    try:
                        st.markdown("**Formatted Output / Explanation:**")
                        formatted_val = res["formatted"]
                        if isinstance(formatted_val, (dict, list)):
                            st.json(formatted_val)
                        else:
                            st.write(formatted_val)
                    except Exception:
                        logger.exception("Failed to render formatted output for entry %s", entry_id)
                        st.error("Failed to render formatted output for this entry.")

                # Raw LLM output
                if res.get("raw"):
                    try:
                        with st.expander("Raw LLM / Agent Output"):
                            # raw may be complex; try to show as JSON if dict-like, else string
                            raw_val = res["raw"]
                            if isinstance(raw_val, (dict, list)):
                                st.json(raw_val)
                            else:
                                st.write(str(raw_val))
                    except Exception:
                        logger.exception("Failed to render raw output for entry %s", entry_id)
                        st.write("Raw output present but could not be displayed.")

                # Error and timings
                try:
                    if res.get("error"):
                        st.error(f"Error: {res['error']}")
                    if res.get("timings"):
                        with st.expander("Execution Timings"):
                            st.json(res["timings"])
                except Exception:
                    logger.exception("Failed to render error/timings for entry %s", entry_id)

                # Delete entry (persisted + session)
                try:
                    if st.button("Delete entry (persisted + session)", key=delete_key):
                        if chat.get("id"):
                            try:
                                history_store.delete_entry(chat["id"])
                            except Exception:
                                logger.exception("Failed to delete persisted history entry %s", chat.get("id"))
                        # remove from session - button click already triggers a rerun
                        st.session_state.chat_history.pop(real_index)
                except Exception:
                    logger.exception("Delete button failed for entry %s", entry_id)
                    st.warning("Could not delete this entry.")
        except Exception as e:
            # If the expander itself fails, log and display the problem and continue rendering other entries.
            logger.exception("Failed rendering history entry %s: %s", entry_id, e)
            st.error(f"Failed to render history entry '{disp_name}' (id={entry_id}). See server logs.")
            continue

# ---------------------------
# Sidebar Footer
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Text-to-SQL Bot Team")
