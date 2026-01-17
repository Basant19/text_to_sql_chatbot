text-to-SQL Bot (Graph-Driven, Safe by Design)

A production-grade Text-to-SQL system that converts natural-language questions into safe, validated SQL over uploaded CSV files.

The system is built using LangGraph, DuckDB, and Gemini (via LangChain), and follows a compiler-style pipeline where each step is isolated, testable, and safe by construction.

✨ Key Highlights

🔒 SQL Safety First (SELECT-only, validation gates)

🧩 Graph-driven architecture (LangGraph)

🧠 LLM used only for SQL generation (never execution)

🧱 Schema hallucination prevention

🧪 Fully testable node-based design

📜 Conversation-aware (SQL history & context)

⚡ Deterministic fallbacks (no hard dependency on embeddings)

🐳 Dockerized deployment (Python 3.11, Docker Compose)

📂 Full Project Structure
text-to-sql/
├── app/
│   ├── __init__.py
│   ├── logger.py
│   ├── exception.py
│   ├── gemini_client.py
│   ├── config.py
│   ├── database.py
│   ├── csv_loader.py
│   ├── schema_store.py
│   ├── vector_search.py
│   ├── utils.py
│   ├── sql_executor.py
│   ├── llm_flow.py
│   ├── langsmith_client.py
│   ├── tools.py
│   ├── history_sql.py
│   └── graph/
│       ├── builder.py
│       ├── agent.py
│       └── nodes/
│           ├── generate_node.py
│           ├── prompt_node.py
│           ├── retrieve_node.py
│           ├── validate_node.py
│           ├── execute_node.py
│           ├── summary_node.py
│           ├── format_node.py
│           ├── context_node.py
│           └── error_node.py
├── tests/
│   ├── test_generate_node.py
│   ├── test_execute_node.py
│   ├── test_csv_loader.py
│   ├── test_history_sql.py
│   └── test_summary_node.py
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .github/workflows/ci.yml
└── PROJECT_REPORT.md

🧭 Overall Application Flow (Bird’s-Eye View)
User (Streamlit UI)
        ↓
app.py
        ↓
GraphBuilder (LangGraph)
        ↓
[ Context → Retrieve → Generate → Validate → Execute → Format ]
        ↓
Results shown in UI + SQL History stored


The application is stateless per request, deterministic where possible, and safe by design.

🚀 1️⃣ App Startup Flow (Cold Start)
When you run:
streamlit run app.py

1.1 Configuration & Environment

Files

config.py

.env

Purpose

Loads API keys

Model names

Paths (DuckDB, FAISS, schema store)

1.2 CSV & Schema Infrastructure

Files

csv_loader.py

schema_store.py

database.py

Purpose

CSVs → DuckDB tables

Schema metadata persisted in schema_store.json

Prevents LLM schema hallucination

1.3 Vector Search (Optional RAG)

File

vector_search.py

Purpose

Semantic retrieval for hints (columns, docs)

Gracefully degrades to deterministic behavior

1.4 LLM & Tools Initialization

Files

gemini_client.py

tools.py

Purpose

Wraps Gemini via LangChain

Single Tools object injected into all nodes

1.5 Graph Construction

Files

graph/builder.py

graph/agent.py

Purpose

Wires nodes into a deterministic pipeline

Stateless, reusable per request

⚙️ 2️⃣ Runtime Flow (User Query)
Example user input:

“Which app has the highest installs?”

2.1 UI Layer

File

app.py

Responsibilities

Read user input

Select active schemas

Call graph.run(...)

Render results or warnings

2.2 Context Node

File

graph/nodes/context_node.py

Purpose

Fetch recent SQL history

Enables follow-up questions

Never affects correctness

Output

{
  "conversation_history": [...],
  "last_successful_sql": "SELECT ..."
}

2.3 Retrieve Node (Optional RAG)

File

graph/nodes/retrieve_node.py

Purpose

Semantic lookup (docs, hints)

Safe to skip

Never blocks the pipeline

2.4 Generate Node (Core Intelligence)

File

graph/nodes/generate_node.py

Purpose

Converts natural language → SQL

Enforces:

SELECT-only

Safe casting

Dirty-data handling

🚨 This node NEVER executes SQL

2.5 Validate Node (Safety Gate)

File

graph/nodes/validate_node.py

Purpose

SQL validation via sqlglot

Rejects:

Non-SELECT queries

Forbidden tables

Invalid SQL

Invalid SQL → ErrorNode

2.6 Execute Node

Files

graph/nodes/execute_node.py

database.py

sql_executor.py

Purpose

Executes validated SQL

Read-only enforcement

Measures execution time

2.7 Format Node

File

graph/nodes/format_node.py

Purpose

Pretty SQL formatting

UI-friendly tables

2.8 History Store

File

history_sql.py

Purpose

Persist:

User query

Generated SQL

Success / failure

Enables conversational memory

2.9 Graph Completion

File

graph/builder.py

Returns final structured output to app.py.

❌ 3️⃣ Error Flow
Generate → Validate ❌
        ↓
ErrorNode
        ↓
Structured error → UI warning


File

graph/nodes/error_node.py

The graph never crashes — all failures are captured and reported safely.

🧠 4️⃣ Mental Model for Contributors

Think of this system as:

“A compiler pipeline for SQL, driven by a graph, with LLMs acting only as a controlled code generator.”

🧩 5️⃣ How to Add a New Node (Walkthrough)
Step 1: Create the Node File
# graph/nodes/audit_node.py
from typing import Dict, Any
from app.logger import get_logger

logger = get_logger("audit_node")

class AuditNode:
    def __init__(self):
        logger.info("AuditNode initialized")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        sql = state.get("sql")
        logger.info("Auditing SQL length=%s", len(sql or ""))
        return {"audit_passed": True}

Step 2: Register the Node in GraphBuilder
from app.graph.nodes.audit_node import AuditNode

audit_node = AuditNode()

graph.add_node("audit", audit_node.run)
graph.add_edge("validate", "audit")
graph.add_edge("audit", "execute")

Step 3: Define Input / Output Contract

Takes state: Dict[str, Any]

Returns a partial update

Must not mutate unrelated keys

Step 4: Add Tests
def test_audit_node():
    node = AuditNode()
    out = node.run({"sql": "SELECT 1"})
    assert out["audit_passed"] is True

Step 5: Log Everything

Every node must log:

Initialization

Run start

Key decisions

🧪 Testing Philosophy

One test file per node

No LLM calls in unit tests

Deterministic inputs & outputs

🛡️ Why This Architecture Works

✅ Stateless execution

✅ Multiple safety layers

✅ Deterministic fallbacks

✅ Easy extensibility

✅ Production-grade logging

🐳 Docker & Deployment
Prerequisites

Docker

Docker Compose

Environment Setup

Create .env:

GOOGLE_API_KEY=your_api_key_here

Build & Run (Recommended)
docker-compose up --build


Open:

http://localhost:8501

Stop Cleanly
Ctrl + C
docker-compose down

Why Docker?

Reproducible environment

Python version pinned (3.11)

One-command startup

Safe isolation of dependencies

Resume-grade deployment practice

🧪 CI & Quality Gates

GitHub Actions (.github/workflows/ci.yml)

Runs:

Unit tests

Lint checks

Prevents unsafe changes to core pipeline

📌 Final Notes

This project demonstrates:

LLM safety engineering

Graph-driven system design

Production-ready deployment

Testable, deterministic AI pipelines

It is intentionally designed to be:

Safe by default, debuggable by design, and extensible without fear.