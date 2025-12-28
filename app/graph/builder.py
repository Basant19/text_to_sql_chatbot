# File: app/graph/builder.py
from __future__ import annotations

"""
GraphBuilder
============

Stateless execution engine for the Text-to-SQL pipeline.

Pipeline (fixed contract):
    Context
      → Retrieve
      → Generate
      → Validate
      → Execute
      → Format

Design rules:
- GraphBuilder is STATELESS
- Nodes may enrich state but must never delete fields
- Rows, columns, and sql must survive end-to-end
- Only ExecuteNode enforces SQL safety
"""

import sys
import time
import inspect
from typing import Any, Dict, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

from app.graph.nodes.context_node import ContextNode
from app.graph.nodes.retrieve_node import RetrieveNode
from app.graph.nodes.generate_node import GenerateNode
from app.graph.nodes.validate_node import ValidateNode
from app.graph.nodes.execute_node import ExecuteNode
from app.graph.nodes.error_node import ErrorNode
from app.graph.nodes.format_node import FormatNode

logger = get_logger("graph_builder")


# ------------------------------------------------------------------
# Safe node invocation
# ------------------------------------------------------------------
def _safe_run(node: Any, **kwargs) -> Any:
    """
    Call node.run(**kwargs) safely by filtering unsupported args.

    This allows nodes to evolve independently without breaking the graph.
    """
    if node is None:
        return None

    fn = node.run if hasattr(node, "run") else node
    sig = inspect.signature(fn)

    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


class GraphBuilder:
    """
    GraphBuilder is the single execution coordinator for the SQL pipeline.

    It is:
    - Stateless
    - Deterministic
    - Safe to reuse across UI reruns
    """

    _global_tools: Optional[Tools] = None

    # ------------------------------------------------------------------
    # Tools lifecycle
    # ------------------------------------------------------------------
    @classmethod
    def set_global_tools(cls, tools: Tools) -> None:
        cls._global_tools = tools
        logger.info("Global Tools registered in GraphBuilder")

    @classmethod
    def get_global_tools(cls) -> Tools:
        if cls._global_tools is None:
            cls._global_tools = Tools()
            logger.warning("Global Tools auto-created (fallback)")
        return cls._global_tools

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def build_default(cls, *, tools: Optional[Tools] = None) -> "GraphBuilder":
        """
        Factory for default graph configuration.
        """
        if tools:
            cls.set_global_tools(tools)
        return cls(tools=tools)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self, *, tools: Optional[Tools] = None):
        self.tools = tools or self.get_global_tools()

        self.context_node = ContextNode(self.tools)
        self.retrieve_node = RetrieveNode(self.tools)
        self.generate_node = GenerateNode(
            self.tools,
            provider_client=self.tools.get_provider_client(),
        )
        self.validate_node = ValidateNode(self.tools)
        self.execute_node = ExecuteNode(self.tools)
        self.format_node = FormatNode(pretty=True)
        self.error_node = ErrorNode()

        logger.info("GraphBuilder initialized (stateless)")

    # ------------------------------------------------------------------
    # Schema normalization
    # ------------------------------------------------------------------
    def _normalize_schemas(self, schemas: Any) -> Dict[str, Dict[str, Any]]:
        """
        Normalize schemas input into a consistent mapping.

        Accepts:
        - Dict[str, schema]
        - List[str] (CSV names)

        Returns:
        - Dict[str, schema]
        """
        if isinstance(schemas, dict):
            return schemas

        if isinstance(schemas, list):
            from app.schema_store import SchemaStore

            store = SchemaStore.get_instance()
            out: Dict[str, Dict[str, Any]] = {}

            for name in schemas:
                schema = store.get(name)
                if schema:
                    out[name] = schema

            return out

        raise CustomException(f"Invalid schemas type: {type(schemas)}", sys)

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------
    def run(
        self,
        user_query: str,
        schemas: Any,
        *,
        run_query: bool = True,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute the full Text-to-SQL pipeline.

        Returns a stable result contract:
        {
            sql: str
            valid: bool
            rows: list
            columns: list
            rowcount: int
            error: Optional[str]
            timings: dict
        }
        """

        start = time.time()
        timings: Dict[str, float] = {}

        result: Dict[str, Any] = {
            "sql": "",
            "valid": False,
            "rows": [],
            "columns": [],
            "rowcount": 0,
            "error": None,
            "timings": timings,
        }

        try:
            # 1️⃣ Normalize schemas
            schemas = self._normalize_schemas(schemas)
            timings["schemas"] = time.time() - start

            # 2️⃣ Context
            t0 = time.time()
            context = _safe_run(
                self.context_node,
                csv_names=list(schemas.keys()),
            )
            timings["context"] = time.time() - t0

            # 3️⃣ Retrieve (RAG)
            t0 = time.time()
            retrieved = _safe_run(
                self.retrieve_node,
                user_query=user_query,
                schemas=schemas,
                top_k=top_k,
            ) or []
            timings["retrieve"] = time.time() - t0

            # 4️⃣ Generate SQL
            t0 = time.time()
            gen = _safe_run(
                self.generate_node,
                user_query=user_query,
                schemas=schemas,
                context=context,
                retrieved=retrieved,
            )
            timings["generate"] = time.time() - t0

            sql = gen.get("sql", "") if isinstance(gen, dict) else str(gen)
            result["sql"] = sql.strip()

            # 5️⃣ Validate
            t0 = time.time()
            val = _safe_run(self.validate_node, sql=result["sql"], schemas=schemas)
            result["valid"] = bool(val.get("valid")) if isinstance(val, dict) else bool(val)
            timings["validate"] = time.time() - t0

            # 6️⃣ Execute
            exec_out: Dict[str, Any] = {}
            if run_query and result["valid"]:
                t0 = time.time()
                exec_out = _safe_run(
                    self.execute_node,
                    sql=result["sql"],
                    table_schemas=schemas,
                    read_only=True,
                ) or {}
                timings["execute"] = time.time() - t0

                # Preserve execution outputs
                result.update(exec_out)

            # 7️⃣ Format (🚫 must NOT drop rows)
            t0 = time.time()
            formatted = _safe_run(
                self.format_node,
                sql=result["sql"],
                schemas=schemas,
                retrieved=retrieved,
                execution=exec_out,
                raw=gen,
            )
            timings["format"] = time.time() - t0

            if isinstance(formatted, dict):
                result.update(formatted)

            timings["total"] = time.time() - start

            logger.info(
                "Graph run complete | valid=%s | rows=%s | total=%.3fs",
                result["valid"],
                len(result.get("rows", [])),
                timings["total"],
            )

            return result

        except Exception as e:
            logger.exception("Graph run failed")
            return self.error_node.run(e)