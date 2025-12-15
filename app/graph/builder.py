# app/graph/builder.py
import sys
import time
import inspect
from typing import Any, Dict, Optional, List

from app.logger import get_logger
from app.exception import CustomException

logger = get_logger("graph_builder")


def _try_call_run(node: Any, *args, **kwargs) -> Any:
    ...
    # (UNCHANGED – keep your existing implementation)
    ...


class GraphBuilder:
    """
    Orchestrator wiring nodes into a linear flow:
        context_node -> retrieve_node -> prompt_node -> generate_node ->
        validate_node -> execute_node -> format_node
    """

    def __init__(
        self,
        context_node: Any,
        retrieve_node: Any,
        prompt_node: Any,
        generate_node: Any,
        validate_node: Any,
        execute_node: Any,
        format_node: Any,
        error_node: Optional[Any] = None,
    ):
        self.context_node = context_node
        self.retrieve_node = retrieve_node
        self.prompt_node = prompt_node
        self.generate_node = generate_node
        self.validate_node = validate_node
        self.execute_node = execute_node
        self.format_node = format_node
        self.error_node = error_node
        logger.info("GraphBuilder initialized")

    # ------------------------------------------------------------------
    # 🔑 NEW: schema normalization
    # ------------------------------------------------------------------
    def _normalize_schemas(self, schemas: Any) -> Dict[str, Dict[str, Any]]:
        """
        Ensure schemas is always a canonical schema map:
          { canonical_table_name -> metadata dict }

        Accepts:
          - Dict (already valid)
          - List[str] (table names)
        """
        if isinstance(schemas, dict):
            return schemas

        if isinstance(schemas, list):
            # Attempt to resolve via SchemaStore
            try:
                from app.schema_store import SchemaStore
                store = SchemaStore.get_instance()
            except Exception as e:
                raise CustomException(
                    f"SchemaStore unavailable while normalizing schemas: {e}",
                    sys,
                )

            resolved: Dict[str, Dict[str, Any]] = {}
            for name in schemas:
                meta = store.get(name)
                if meta:
                    resolved[name] = meta
                else:
                    logger.warning(
                        "GraphBuilder: schema '%s' not found in SchemaStore",
                        name,
                    )

            return resolved

        # Anything else is invalid
        raise CustomException(
            f"Invalid schemas type passed through graph: {type(schemas)}",
            sys,
        )

    def run(self, user_query: str, csv_names: list, run_query: bool = False) -> Dict[str, Any]:
        start_all = time.time()
        timings: Dict[str, float] = {}

        result: Dict[str, Any] = {
            "prompt": None,
            "sql": None,
            "valid": False,
            "execution": None,
            "formatted": None,
            "raw": None,
            "error": None,
            "timings": timings,
        }

        ctx_for_error = {"user_query": user_query, "csv_names": csv_names, "timings": timings}

        try:
            # ----------------------------------------------------------
            # context_node → schemas
            # ----------------------------------------------------------
            t0 = time.time()
            try:
                raw_schemas = _try_call_run(self.context_node, csv_names)
                schemas = self._normalize_schemas(raw_schemas)
            except Exception as e:
                return self._handle_error_node(e, ctx_for_error, step="context")
            timings["context"] = time.time() - t0

            logger.debug(
                "GraphBuilder: normalized schemas keys=%s",
                list(schemas.keys()),
            )

            # ----------------------------------------------------------
            # retrieve_node
            # ----------------------------------------------------------
            t1 = time.time()
            try:
                retrieved = _try_call_run(self.retrieve_node, user_query, schemas)
                if retrieved is None:
                    retrieved = []
            except Exception as e:
                return self._handle_error_node(e, ctx_for_error, step="retrieve")
            timings["retrieve"] = time.time() - t1

            # ----------------------------------------------------------
            # prompt_node
            # ----------------------------------------------------------
            t2 = time.time()
            try:
                prompt_ret = _try_call_run(self.prompt_node, user_query, schemas, retrieved)
                prompt_text = (
                    prompt_ret.get("prompt")
                    if isinstance(prompt_ret, dict)
                    else str(prompt_ret)
                )
                result["prompt"] = prompt_ret
            except Exception as e:
                return self._handle_error_node(e, ctx_for_error, step="prompt")
            timings["prompt"] = time.time() - t2

            # ----------------------------------------------------------
            # generate_node
            # ----------------------------------------------------------
            t3 = time.time()
            try:
                gen = _try_call_run(self.generate_node, prompt_text, schemas, retrieved)
            except Exception as e:
                return self._handle_error_node(e, ctx_for_error, step="generate")
            timings["generate"] = time.time() - t3

            raw = gen.get("raw") if isinstance(gen, dict) else gen
            sql = gen.get("sql") if isinstance(gen, dict) else str(gen)
            result["raw"] = raw
            result["sql"] = sql

            # ----------------------------------------------------------
            # validate_node
            # ----------------------------------------------------------
            t4 = time.time()
            try:
                val = _try_call_run(self.validate_node, sql, schemas)
            except Exception as e:
                return self._handle_error_node(e, ctx_for_error, step="validate")
            timings["validate"] = time.time() - t4

            if isinstance(val, dict):
                sql = val.get("sql", sql)
                result["valid"] = bool(val.get("valid", False))
                result["sql"] = sql
            else:
                result["valid"] = bool(val)

            # ----------------------------------------------------------
            # execute_node (FIXED: schemas guaranteed valid)
            # ----------------------------------------------------------
            execution_result = None
            if run_query and result["valid"]:
                t5 = time.time()
                try:
                    execution_result = _try_call_run(self.execute_node, sql, schemas)
                except Exception as e:
                    ctx_for_error.update({"sql": sql})
                    return self._handle_error_node(e, ctx_for_error, step="execute")
                timings["execute"] = time.time() - t5
                result["execution"] = execution_result

            # ----------------------------------------------------------
            # format_node
            # ----------------------------------------------------------
            t6 = time.time()
            try:
                formatted = _try_call_run(
                    self.format_node, sql, schemas, retrieved, execution_result, raw
                )
            except Exception as e:
                return self._handle_error_node(e, ctx_for_error, step="format")
            timings["format"] = time.time() - t6
            result["formatted"] = formatted

            timings["total"] = time.time() - start_all
            logger.info(
                "GraphBuilder: run complete valid=%s total=%.3fs",
                result["valid"],
                timings["total"],
            )
            return result

        except Exception as e:
            logger.exception("GraphBuilder.run encountered unexpected error")
            return self._handle_error_node(e, ctx_for_error, step="run")
