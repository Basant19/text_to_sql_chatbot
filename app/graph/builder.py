# File: app/graph/builder.py
from __future__ import annotations

import sys
import time
import inspect
from typing import Any, Dict, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

from app.graph.nodes.retrieve_node import RetrieveNode
from app.graph.nodes.generate_node import GenerateNode
from app.graph.nodes.validate_node import ValidateNode
from app.graph.nodes.execute_node import ExecuteNode
from app.graph.nodes.context_node import ContextNode
from app.graph.nodes.error_node import ErrorNode
from app.graph.nodes.format_node import FormatNode  # 🔥 Ensure UI output

logger = get_logger("graph_builder")


# ------------------------------------------------------------------
# Safe node invocation helper
# ------------------------------------------------------------------
def _try_call_run(node: Any, **kwargs) -> Any:
    """
    Safely invoke a node by adapting to its actual run() signature.
    Guarantees:
    - Only parameters accepted by the node are passed
    - Prevents "unexpected keyword argument" errors
    - Prevents missing required arguments
    """
    if node is None:
        return None

    fn = getattr(node, "run", node)
    if not callable(fn):
        raise CustomException(f"Node {node} is not callable", sys)

    try:
        sig = inspect.signature(fn)
        accepted = sig.parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return fn(**filtered_kwargs)
    except TypeError as e:
        logger.exception("Signature mismatch in %s", node.__class__.__name__)
        raise CustomException(f"{node.__class__.__name__} signature mismatch: {e}", sys)
    except Exception as e:
        logger.exception("Node execution failed: %s", node.__class__.__name__)
        raise CustomException(f"{node.__class__.__name__} failed: {e}", sys)


class GraphBuilder:
    """
    GraphBuilder
    ------------
    Orchestrates the Text-to-SQL pipeline:

        context -> retrieve -> generate -> validate -> execute -> format

    Guarantees:
    - Nodes are invoked only with parameters they explicitly accept
    - Schema metadata is always propagated
    - Failures are returned as structured errors
    - UI-ready output is always produced via FormatNode
    """

    def __init__(
        self,
        *,
        tools: Optional[Tools] = None,
        context_node: Optional[Any] = None,
        retrieve_node: Optional[Any] = None,
        generate_node: Optional[Any] = None,
        validate_node: Optional[Any] = None,
        execute_node: Optional[Any] = None,
        error_node: Optional[Any] = None,
        format_node: Optional[FormatNode] = None,
    ):
        self.tools = tools or Tools()

        self.context_node = context_node or ContextNode(self.tools)
        self.retrieve_node = retrieve_node or RetrieveNode(self.tools)
        self.generate_node = generate_node or GenerateNode(self.tools, provider_client=self.tools.get_provider_client())
        self.validate_node = validate_node or ValidateNode(self.tools)
        self.execute_node = execute_node or ExecuteNode(self.tools)
        self.error_node = error_node or ErrorNode()
        self.format_node = format_node or FormatNode(pretty=True)  # 🔥 Always format for UI

        logger.info("GraphBuilder initialized with FormatNode integration")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def build_default(cls) -> "GraphBuilder":
        logger.info("Building default GraphBuilder")
        try:
            return cls()
        except Exception as e:
            logger.exception("GraphBuilder build failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Schema normalization
    # ------------------------------------------------------------------
    def _normalize_schemas(self, schemas: Any) -> Dict[str, Dict[str, Any]]:
        if isinstance(schemas, dict):
            return schemas

        if isinstance(schemas, list):
            from app.schema_store import SchemaStore

            store = SchemaStore.get_instance()
            resolved: Dict[str, Dict[str, Any]] = {}
            for name in schemas:
                meta = store.get(name)
                if meta:
                    resolved[name] = meta
                else:
                    logger.warning("Schema not found: %s", name)
            return resolved

        raise CustomException(f"Invalid schemas type: {type(schemas)}", sys)

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------
    def run(
        self,
        user_query: str,
        schemas: Any,
        *,
        run_query: bool = True,
        top_k: int = 5,
    ) -> Dict[str, Any]:

        start = time.time()
        timings: Dict[str, float] = {}
        result: Dict[str, Any] = {
            "sql": None,
            "rows": [],
            "columns": [],
            "rowcount": 0,
            "valid": False,
            "error": None,
            "timings": timings,
        }

        try:
            # 1️⃣ Normalize schemas
            t0 = time.time()
            schemas = self._normalize_schemas(schemas)
            timings["schemas"] = time.time() - t0
            logger.debug("Schemas resolved: %s", list(schemas.keys()))

            # 2️⃣ Collect context
            t1 = time.time()
            context_map = _try_call_run(self.context_node, csv_names=list(schemas.keys()))
            timings["context"] = time.time() - t1

            # 3️⃣ Retrieve context
            t2 = time.time()
            retrieved = _try_call_run(
                self.retrieve_node,
                query=user_query,
                schemas=schemas,
                top_k=top_k,
            ) or []
            timings["retrieve"] = time.time() - t2

            # 4️⃣ Generate SQL
            t3 = time.time()
            gen = _try_call_run(
                self.generate_node,
                query=user_query,
                schemas=schemas,
                retrieved=retrieved,
                context_map=context_map,
            )
            timings["generate"] = time.time() - t3

            sql = gen.get("sql") if isinstance(gen, dict) else str(gen)
            result["sql"] = sql

            # 5️⃣ Validate SQL
            t4 = time.time()
            val = _try_call_run(self.validate_node, sql=sql, schemas=schemas)
            timings["validate"] = time.time() - t4

            if isinstance(val, dict):
                result["valid"] = bool(val.get("valid", False))
                sql = val.get("sql", sql)
            else:
                result["valid"] = bool(val)
            result["sql"] = sql

            # 6️⃣ Execute SQL (read-only)
            exec_out = {}
            if run_query and result["valid"]:
                t5 = time.time()
                exec_out = _try_call_run(
                    self.execute_node,
                    sql=sql,
                    table_schemas=schemas,
                    read_only=True,
                ) or {}
                timings["execute"] = time.time() - t5

                result["rows"] = exec_out.get("rows", [])
                result["columns"] = exec_out.get("columns", [])
                result["rowcount"] = exec_out.get("rowcount", 0)

            # 7️⃣ Format output for UI
            t6 = time.time()
            formatted = _try_call_run(
                self.format_node,
                sql=sql,
                schemas=schemas,
                retrieved=retrieved,
                execution=exec_out,
                raw=gen,
            )
            timings["format"] = time.time() - t6

            # Merge formatted output into result
            if isinstance(formatted, dict):
                result.update(formatted)

            timings["total"] = time.time() - start
            logger.info(
                "Graph run complete | valid=%s | total=%.3fs",
                result["valid"],
                timings["total"],
            )

            return result

        except Exception as e:
            # Use ErrorNode to normalize the exception
            timings["total"] = time.time() - start
            try:
                return self.error_node.run(e, step="graph_run", context={"timings": timings})
            except Exception:
                # Fallback minimal payload
                logger.exception("GraphBuilder.run failed and ErrorNode failed")
                return {**result, "error": str(e), "timings": timings}
