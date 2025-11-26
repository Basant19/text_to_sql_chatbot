# app.py
import streamlit as st
import os
from typing import List, Dict, Any

from app.csv_loader import CSVLoader
from app.schema_store import SchemaStore
from app.vector_search import VectorSearch
from app.graph.builder import GraphBuilder
from app.history_sql import HistoryStore
from app.logger import get_logger
import app.config as config_module

logger = get_logger("app")

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

        # Persist to history store + session
        try:
            entry = history_store.add_entry(name=query_name, query=user_query, result=result)
            st.session_state.chat_history.append({
                "id": entry.get("id"),
                "name": entry.get("name"),
                "query": entry.get("query"),
                "result": entry.get("result"),
                "created_at": entry.get("created_at"),
                "updated_at": entry.get("updated_at"),
            })
        except Exception:
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
            st.experimental_rerun()
    with col_b:
        if st.button("Export History JSON (persisted)"):
            try:
                path = history_store.export_json()
                st.success(f"Exported history to {path}")
            except Exception:
                st.exception("Failed to export history")
    with col_c:
        st.info(f"{len(st.session_state.chat_history)} queries stored in this session")

    for display_idx, chat in enumerate(reversed(st.session_state.chat_history)):
        real_index = len(st.session_state.chat_history) - 1 - display_idx
        disp_name = chat.get("name", f"Query {display_idx+1}")
        expander_key = f"expander_{real_index}"

        with st.expander(f"{disp_name}", expanded=False, key=expander_key):
            rename_key = f"rename_{real_index}"
            new_name = st.text_input("Rename query", value=disp_name, key=rename_key)
            if new_name and new_name != disp_name:
                st.session_state.chat_history[real_index]["name"] = new_name
                entry_id = st.session_state.chat_history[real_index].get("id")
                if entry_id:
                    try:
                        updated = history_store.update_entry(entry_id, name=new_name)
                        if updated:
                            st.session_state.chat_history[real_index].update({
                                "updated_at": updated.get("updated_at")
                            })
                    except Exception:
                        logger.exception("Failed to update persisted history entry (rename)")

            st.markdown(f"**User Query:** {chat.get('query')}")
            res = chat.get("result") or {}
            if not res:
                st.info("No result stored for this query.")
                continue

            st.markdown("**Generated SQL:**")
            st.code(res.get("sql") or "No SQL generated.", language="sql")
            st.write(f"Valid SQL: {res.get('valid')}")

            if res.get("execution"):
                st.markdown("**Execution Results:**")
                exec_res = res["execution"]
                rows = exec_res.get("rows") if isinstance(exec_res, dict) else exec_res
                try:
                    st.dataframe(rows)
                except Exception:
                    st.write(rows)
                st.write("Metadata:", exec_res.get("meta", {}))

            if res.get("formatted"):
                st.markdown("**Formatted Output / Explanation:**")
                st.json(res["formatted"])

            if res.get("raw"):
                with st.expander("Raw LLM / Agent Output"):
                    st.json(res["raw"])

            if res.get("error"):
                st.error(f"Error: {res['error']}")

            if res.get("timings"):
                with st.expander("Execution Timings"):
                    st.json(res["timings"])

            if st.button("Delete entry (persisted + session)", key=f"del_{real_index}"):
                entry_id = st.session_state.chat_history[real_index].get("id")
                if entry_id:
                    try:
                        history_store.delete_entry(entry_id)
                    except Exception:
                        logger.exception("Failed to delete persisted history entry")
                st.session_state.chat_history.pop(real_index)
                st.experimental_rerun()

# ---------------------------
# Sidebar Footer
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Text-to-SQL Bot Team")
