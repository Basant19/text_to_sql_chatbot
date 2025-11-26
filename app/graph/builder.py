# app/graph/builder.py
import sys
import time
from typing import Any, Dict, Optional

from app.logger import get_logger
from app.exception import CustomException

logger = get_logger("graph_builder")


class GraphBuilder:
    """
    Simple orchestrator that wires LangGraph-like nodes together into a linear flow:

        context_node -> retrieve_node -> prompt_node -> generate_node ->
        validate_node -> execute_node -> format_node

    Each node is expected to expose a .run(...) method, signature depending on the node.
    You can inject any node instances for testing.

    The builder.run(...) method returns a unified dict:
      {
        "prompt": <str>,
        "sql": <str>,
        "valid": <bool>,
        "execution": <dict|None>,
        "formatted": <any|None>,
        "raw": <raw generation output|None>,
        "error": <str|None>,
        "timings": {...}
      }
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

    @staticmethod
    def build_default():
        """
        Build a default graph using the project's node implementations.
        (lazy import so module can be imported even if graph isn't used)

        Wiring rules:
          - Attempt to instantiate a GeminiClient provider (preferred) if possible.
          - Instantiate LangSmithClient as tracer only if LANGSMITH_API_KEY is present.
          - Both provider and tracer instantiation failures are non-fatal; they are logged
            and the GenerateNode will be constructed with None for those clients (tests can inject mocks).
        """
        try:
            from app.graph.nodes.context_node import ContextNode
            from app.graph.nodes.retrieve_node import RetrieveNode
            from app.graph.nodes.prompt_node import PromptNode
            from app.graph.nodes.generate_node import GenerateNode
            from app.graph.nodes.validate_node import ValidateNode
            from app.graph.nodes.execute_node import ExecuteNode
            from app.graph.nodes.format_node import FormatNode
            from app.graph.nodes.error_node import ErrorNode
        except Exception as e:
            logger.exception("Failed to import default nodes")
            raise CustomException(e, sys)

        provider_client = None
        tracer_client = None

        # Avoid forcing config validation here; instantiate config in permissive mode
        try:
            from app.config import Config

            cfg = Config(require_keys=False)
        except Exception:
            cfg = None

        # Try to create a Gemini provider (preferred) — non-fatal
        try:
            try:
                from app.gemini_client import GeminiClient  # type: ignore

                gemini_api_key = getattr(cfg, "GEMINI_API_KEY", None) if cfg is not None else None
                if gemini_api_key:
                    provider_client = GeminiClient(api_key=gemini_api_key)
                else:
                    # Try to instantiate with no args (use config module-level fallback inside GeminiClient)
                    provider_client = GeminiClient()
                logger.info("GraphBuilder: GeminiClient instantiated as provider_client")
            except Exception as e:
                provider_client = None
                logger.debug("GraphBuilder: GeminiClient unavailable or failed to instantiate: %s", e)
        except Exception:
            provider_client = None

        # Try to create LangSmith tracer (observability only) — non-fatal
        try:
            try:
                from app.langsmith_client import LangSmithClient  # type: ignore

                langsmith_key = getattr(cfg, "LANGSMITH_API_KEY", None) if cfg is not None else None
                if langsmith_key:
                    tracer_client = LangSmithClient(api_key=langsmith_key)
                    logger.info("GraphBuilder: LangSmithClient instantiated as tracer_client")
                else:
                    tracer_client = None
                    logger.debug("GraphBuilder: LANGSMITH_API_KEY not set; tracer_client not created")
            except Exception as e:
                tracer_client = None
                logger.debug("GraphBuilder: LangSmithClient import/instantiation failed (tracer disabled): %s", e)
        except Exception:
            tracer_client = None

        # Instantiate nodes, wiring provider/tracer into GenerateNode
        try:
            context_node = ContextNode()
            retrieve_node = RetrieveNode()
            prompt_node = PromptNode()
            # Pass provider_client and tracer_client into GenerateNode constructor
            generate_node = GenerateNode(provider_client=provider_client, tracer_client=tracer_client)
            validate_node = ValidateNode()
            execute_node = ExecuteNode()
            format_node = FormatNode()
            error_node = ErrorNode()
        except Exception as e:
            logger.exception("Failed to instantiate nodes for default graph")
            raise CustomException(e, sys)

        return GraphBuilder(
            context_node=context_node,
            retrieve_node=retrieve_node,
            prompt_node=prompt_node,
            generate_node=generate_node,
            validate_node=validate_node,
            execute_node=execute_node,
            format_node=format_node,
            error_node=error_node,
        )

    def run(self, user_query: str, csv_names: list, run_query: bool = False) -> Dict[str, Any]:
        """
        Execute the graph end-to-end.

        - user_query: user's natural language question
        - csv_names: list of csv/table names that form the context
        - run_query: if True, execution node will be invoked (when SQL is valid)

        Returns the result dict described in class docstring.
        """
        start_all = time.time()
        timings = {}
        result = {
            "prompt": None,
            "sql": None,
            "valid": False,
            "execution": None,
            "formatted": None,
            "raw": None,
            "error": None,
            "timings": timings,
        }

        try:
            t0 = time.time()
            schemas = self.context_node.run(csv_names)
            timings["context"] = time.time() - t0

            t1 = time.time()
            # retrieve_node.run signature historically accepted (query, csv_names) or (query, schemas)
            # keep passing user_query and schemas (many implementations expect schemas)
            retrieved = self.retrieve_node.run(user_query, schemas)
            timings["retrieve"] = time.time() - t1

            t2 = time.time()
            prompt = self.prompt_node.run(user_query, schemas, retrieved_docs=retrieved)
            timings["prompt"] = time.time() - t2
            result["prompt"] = prompt

            t3 = time.time()
            gen = self.generate_node.run(prompt)
            timings["generate"] = time.time() - t3

            # normalize generation output
            raw = gen.get("raw") if isinstance(gen, dict) else gen
            sql = gen.get("sql") if isinstance(gen, dict) else (gen[0] if isinstance(gen, tuple) else str(gen))
            result["raw"] = raw
            result["sql"] = sql

            t4 = time.time()
            val = self.validate_node.run(sql, schemas)
            timings["validate"] = time.time() - t4

            result["valid"] = bool(val.get("valid", False))

            execution_result = None
            if run_query and result["valid"]:
                t5 = time.time()
                execution_result = self.execute_node.run(sql, schemas)
                timings["execute"] = time.time() - t5
                result["execution"] = execution_result

            t6 = time.time()
            formatted = self.format_node.run(sql, schemas, retrieved, execution_result, raw)
            timings["format"] = time.time() - t6
            result["formatted"] = formatted

            total = time.time() - start_all
            timings["total"] = total

            logger.info("GraphBuilder: run complete valid=%s total=%.3fs", result["valid"], total)
            return result

        except Exception as e:
            logger.exception("GraphBuilder.run encountered an error")
            # Use error_node if available to format error response
            try:
                if self.error_node:
                    # prefer handle(), else run()
                    if hasattr(self.error_node, "handle"):
                        return self.error_node.handle(e, {"user_query": user_query, "csv_names": csv_names})
                    elif hasattr(self.error_node, "run"):
                        return self.error_node.run(e, {"user_query": user_query, "csv_names": csv_names})
            except Exception:
                logger.exception("ErrorNode itself failed")

            # fallback error shape
            return {
                "prompt": None,
                "sql": None,
                "valid": False,
                "execution": None,
                "formatted": None,
                "raw": None,
                "error": str(e),
                "timings": timings,
            }
