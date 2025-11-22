# app.py
import streamlit as st
import os
from app.csv_loader import CSVLoader
from app.schema_store import SchemaStore
from app.vector_search import VectorSearch
from app.database import get_connection, execute_query, load_csv_table, table_exists, list_tables

from app.graph.agent import AgentGraph
from app.logger import get_logger
from app.config import Config

logger = get_logger("app")

# Initialize components
config = Config()
csv_loader = CSVLoader()
schema_store = SchemaStore()
vector_search = VectorSearch()
db_manager = DuckDBManager()
agent = AgentGraph()  # wraps GraphBuilder internally

# Initialize Streamlit session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit app
st.set_page_config(page_title="Text-to-SQL Bot", layout="wide")
st.title("📊 Text-to-SQL Bot")

# ================================
# Sidebar - CSV Upload & Selection
# ================================
st.sidebar.header("CSV Management")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSVs", type=["csv"], accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        csv_loader.save_csv(file)
        st.sidebar.success(f"Uploaded {file.name}")

# List uploaded CSVs
all_csvs = csv_loader.list_uploaded_csvs()
selected_csvs = st.sidebar.multiselect(
    "Select CSVs for context", options=all_csvs, default=all_csvs
)

load_button = st.sidebar.button("Load Selected CSVs")

if load_button and selected_csvs:
    # 1) Extract schemas
    schemas = csv_loader.load_and_extract(selected_csvs)
    schema_store.save_schemas(schemas)
    # 2) Update FAISS embeddings for RAG retrieval
    vector_search.index_schemas(schemas)
    st.sidebar.success("Schemas loaded and vector index updated!")

# ======================
# Main Area - Query Input
# ======================
st.header("Ask a Question in Natural Language")
user_query = st.text_area("Enter your question here:")

# Optional rename for this query
query_name = st.text_input("Optional: Give this query a name", value=f"Query {len(st.session_state.chat_history)+1}")

run_query = st.checkbox("Execute SQL if valid?", value=True)
execute_button = st.button("Run Query")

# ======================
# Query Execution
# ======================
if execute_button and user_query and selected_csvs:
    with st.spinner("Generating SQL..."):
        try:
            # Run agent graph
            result = agent.run(user_query, selected_csvs, run_query=run_query)
        except Exception as e:
            st.error(f"Error running agent: {e}")
            logger.exception("Agent run failed")
            result = {"error": str(e)}

    # Save query and results in chat history
    st.session_state.chat_history.append({
        "name": query_name,
        "query": user_query,
        "result": result
    })

# ======================
# Display Chat History
# ======================
if st.session_state.chat_history:
    st.subheader("💬 Query History")
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"{chat['name']}"):
            st.markdown(f"**User Query:** {chat['query']}")
            res = chat["result"]
            if res:
                st.markdown("**Generated SQL:**")
                st.code(res.get("sql") or "No SQL generated.", language="sql")
                st.write(f"Valid SQL: {res.get('valid')}")
                if res.get("execution"):
                    st.markdown("**Execution Results:**")
                    st.dataframe(res["execution"].get("rows", []))
                    st.write("Metadata:", res["execution"].get("meta", {}))
                if res.get("formatted"):
                    st.markdown("**Formatted Output / Explanation:**")
                    st.json(res["formatted"])
                if res.get("raw"):
                    with st.expander("Raw LLM Output"):
                        st.json(res["raw"])
                if res.get("error"):
                    st.error(f"Error: {res['error']}")
                if res.get("timings"):
                    with st.expander("Execution Timings"):
                        st.json(res["timings"])

# ======================
# Footer / Notes
# ======================
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Text-to-SQL Bot Team")
