#D:\text_to_sql_bot\app\graph\builder.py
import sys
import time
import inspect
from typing import Any, Dict, Optional

from app.logger import get_logger
from app.exception import CustomException

logger = get_logger("graph_builder")


def _try_call_run(node: Any, *args, **kwargs) -> Any:
    """
    Resilient caller for node.run:
      - Bind args/kwargs to signature if possible to allow runtime TypeErrors to propagate.
      - Otherwise attempt adaptive calling strategies to handle signature variations.
    """
    if node is None:
        raise RuntimeError("Node is None, cannot call run()")

    run_fn = getattr(node, "run", None)
    if run_fn is None or not callable(run_fn):
        raise AttributeError(f"Node {node} has no callable run()")

    last_exc = None

    # 0) Try to bind args/kwargs to signature first
    try:
        sig = inspect.signature(run_fn)
        try:
            sig.bind_partial(*args, **kwargs)
            return run_fn(*args, **kwargs)
        except TypeError:
            pass
    except (ValueError, TypeError):
        try:
            return run_fn(*args, **kwargs)
        except TypeError as e:
            last_exc = e
        except Exception:
            raise

    # 1) Try progressively fewer positional args (keep kwargs)
    for n in range(len(args), -1, -1):
        try:
            return run_fn(*args[:n], **kwargs)
        except TypeError as e:
            last_exc = e
        except Exception:
            raise

    # 2) Try mapping positional args by parameter names
    try:
        sig = inspect.signature(run_fn)
        call_kwargs = {}
        params = list(sig.parameters.values())
        for i, a in enumerate(args):
            if i < len(params):
                name = params[i].name
                call_kwargs[name] = a
        call_kwargs.update(kwargs)
        accepted = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        filtered = {k: v for k, v in call_kwargs.items() if k in accepted}
        return run_fn(**filtered)
    except TypeError as e:
        last_exc = e
    except Exception:
        raise

    raise last_exc or RuntimeError("Failed to call node.run() for unknown reasons")


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

    @staticmethod
    def build_default():
        """
        Build a default graph using the project's node implementations. Lazy imports.
        This function is defensive: nodes may have different constructor signatures,
        and provider/tracer clients are optional.
        """
        try:
            from app.graph.nodes.context_node import ContextNode
            from app.graph.nodes.retrieve_node import RetrieveNode
            from app.graph.nodes.prompt_node import PromptNode
            from app.graph.nodes.generate_node import GenerateNode
            from app.graph.nodes.validate_node import ValidateNode
            from app.graph.nodes.execute_node import ExecuteNode
            from app.graph.nodes.format_node import FormatNode, FormatAdapter
            from app.graph.nodes.error_node import ErrorNode
            from app.tools import Tools
        except Exception as e:
            logger.exception("Failed to import default nodes")
            raise CustomException(e, sys)

        provider_client = None
        tracer_client = None

        # permissive config (don't require provider keys)
        try:
            from app.config import Config
            cfg = Config(require_keys=False)
        except Exception:
            cfg = None

        # Try instantiate provider (non-fatal)
        try:
            try:
                from app.gemini_client import GeminiClient  # type: ignore
                gemini_api_key = getattr(cfg, "GEMINI_API_KEY", None) if cfg is not None else None
                if gemini_api_key:
                    provider_client = GeminiClient(api_key=gemini_api_key)
                else:
                    provider_client = GeminiClient()
                logger.info("GraphBuilder: GeminiClient instantiated as provider_client")
            except Exception as e:
                provider_client = None
                logger.debug("GraphBuilder: GeminiClient unavailable: %s", e)
        except Exception:
            provider_client = None

        # Try LangSmith tracer (non-fatal)
        try:
            try:
                from app.langsmith_client import LangSmithClient  # type: ignore
                langsmith_key = getattr(cfg, "LANGSMITH_API_KEY", None) if cfg is not None else None
                if langsmith_key:
                    tracer_client = LangSmithClient(api_key=langsmith_key)
                    logger.info("GraphBuilder: LangSmithClient instantiated as tracer_client")
                else:
                    tracer_client = None
            except Exception as e:
                tracer_client = None
                logger.debug("GraphBuilder: LangSmithClient unavailable: %s", e)
        except Exception:
            tracer_client = None

        # single Tools instance to share SchemaStore, VectorSearch, etc.
        try:
            tools = Tools(provider_client=provider_client, tracer_client=tracer_client)
        except Exception:
            tools = None
            logger.warning("Could not instantiate Tools() for DI; nodes may still work if they don't require it.")

        # Instantiate nodes; prefer passing tools so nodes share same SchemaStore/VectorSearch
        try:
            # Context & Retrieve accept tools in many implementations
            try:
                context_node = ContextNode(tools=tools) if tools is not None else ContextNode()
            except TypeError:
                context_node = ContextNode()

            try:
                retrieve_node = RetrieveNode(tools=tools) if tools is not None else RetrieveNode()
            except TypeError:
                retrieve_node = RetrieveNode()

            # PromptNode: prefer to pass tools when available; handle varied constructors
            try:
                if tools is not None:
                    prompt_node = PromptNode(tools=tools)
                else:
                    prompt_node = PromptNode()
            except TypeError:
                try:
                    prompt_node = PromptNode(prompt_builder=None)
                except Exception:
                    prompt_node = PromptNode()

            # GenerateNode may accept kwargs provider_client/tracer_client/tools — try flexible instantiation.
            try:
                if tools is not None:
                    generate_node = GenerateNode(tools=tools)
                else:
                    generate_node = GenerateNode(provider_client=provider_client, tracer_client=tracer_client, tools=tools)
            except TypeError:
                try:
                    generate_node = GenerateNode(provider_client, tracer_client)
                except Exception:
                    try:
                        generate_node = GenerateNode(provider_client)
                    except Exception:
                        generate_node = GenerateNode()

            validate_node = ValidateNode()

            # ExecuteNode: try flexible constructor signatures
            try:
                execute_node = ExecuteNode(tools=tools) if tools is not None else ExecuteNode()
            except TypeError:
                try:
                    execute_node = ExecuteNode(tools)
                except TypeError:
                    execute_node = ExecuteNode()

            raw_format = FormatNode()
            format_node = FormatAdapter(raw_format)
            error_node = ErrorNode()
        except Exception as e:
            logger.exception("Failed to instantiate nodes for default graph")
            raise CustomException(e, sys)

        # Return the fully constructed GraphBuilder
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

    def _handle_error_node(self, exc: Exception, ctx: Dict[str, Any], step: Optional[str] = None) -> Dict[str, Any]:
        """
        Attempt to use the configured error_node to produce a friendly result payload.
        Pass the 'step' to the error_node so it can include it in the formatted output.
        """
        logger.exception("GraphBuilder: handling exception at step=%s via error_node: %s", step, exc)
        try:
            if self.error_node:
                # Prefer run(exc, step=..., context=...) signature
                if hasattr(self.error_node, "run"):
                    try:
                        return self.error_node.run(exc, step=step, context=ctx)
                    except TypeError:
                        try:
                            return self.error_node.run(exc, step, ctx)
                        except Exception:
                            try:
                                return self.error_node.run(exc)
                            except Exception:
                                pass
                if hasattr(self.error_node, "handle"):
                    try:
                        return self.error_node.handle(exc, ctx)
                    except Exception:
                        pass
        except Exception:
            logger.exception("ErrorNode failed while handling exception at step=%s", step)

        return {
            "prompt": None,
            "sql": None,
            "valid": False,
            "execution": None,
            "formatted": None,
            "raw": None,
            "error": str(exc),
            "timings": ctx.get("timings", {}),
            "step": step,
        }

    def run(self, user_query: str, csv_names: list, run_query: bool = False) -> Dict[str, Any]:
        """
        Execute the graph end-to-end in a resilient manner.
        """
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
            # context_node -> schemas
            t0 = time.time()
            try:
                schemas = _try_call_run(self.context_node, csv_names)
            except Exception as e:
                return self._handle_error_node(e, ctx_for_error, step="context")
            timings["context"] = time.time() - t0

            # retrieve_node -> retrieved docs
            t1 = time.time()
            try:
                retrieved = _try_call_run(self.retrieve_node, user_query, schemas)
                if retrieved is None:
                    retrieved = []
            except Exception as e:
                return self._handle_error_node(e, ctx_for_error, step="retrieve")
            timings["retrieve"] = time.time() - t1

            # prompt_node -> prompt text (may return dict with 'prompt' key)
            t2 = time.time()
            try:
                prompt_ret = _try_call_run(self.prompt_node, user_query, schemas, retrieved)
                # If prompt_node returned a dict {prompt: ..., pieces: ...}, extract the text
                prompt_text = None
                if isinstance(prompt_ret, dict):
                    prompt_text = prompt_ret.get("prompt") or prompt_ret.get("text") or None
                else:
                    prompt_text = str(prompt_ret)
                # fallback
                if prompt_text is None:
                    prompt_text = str(prompt_ret)
                prompt = prompt_ret  # keep original for debugging/storage
            except Exception as e:
                return self._handle_error_node(e, ctx_for_error, step="prompt")
            timings["prompt"] = time.time() - t2
            result["prompt"] = prompt

            # generate_node -> generation result (pass prompt_text)
            t3 = time.time()
            try:
                gen = _try_call_run(self.generate_node, prompt_text, schemas, retrieved)
            except Exception as e:
                return self._handle_error_node(e, ctx_for_error, step="generate")
            timings["generate"] = time.time() - t3

            # Normalize generation output
            raw = gen.get("raw") if isinstance(gen, dict) else gen
            sql = None
            if isinstance(gen, dict):
                sql = gen.get("sql") or gen.get("text") or gen.get("output") or ""
            elif isinstance(gen, (list, tuple)) and len(gen) > 0:
                sql = gen[0]
            else:
                sql = str(gen) if gen is not None else ""
            result["raw"] = raw
            result["sql"] = sql

            # validate_node -> possibly modified sql + validity info
            t4 = time.time()
            try:
                val = _try_call_run(self.validate_node, sql, schemas)
            except Exception as e:
                return self._handle_error_node(e, ctx_for_error, step="validate")
            timings["validate"] = time.time() - t4

            if isinstance(val, dict):
                sql = val.get("sql", sql)
                result["sql"] = sql
                result["valid"] = bool(val.get("valid", False))
                if val.get("errors"):
                    result.setdefault("validation_errors", []).extend(val.get("errors"))
            else:
                result["valid"] = bool(val)

            # optionally execute SQL
            execution_result = None
            if run_query and result["valid"]:
                t5 = time.time()
                try:
                    # pass schemas as second param so execute_node can auto-load tables when given a mapping
                    execution_result = _try_call_run(self.execute_node, sql, schemas)
                except Exception as e:
                    ctx_for_error.update({"sql": sql, "execution_error": str(e)})
                    return self._handle_error_node(e, ctx_for_error, step="execute")
                timings["execute"] = time.time() - t5
                result["execution"] = execution_result

            # format_node -> produce formatted output
            t6 = time.time()
            try:
                formatted = _try_call_run(self.format_node, sql, schemas, retrieved, execution_result, raw)
            except Exception as e:
                ctx_for_error.update({"sql": sql, "raw": raw, "retrieved": retrieved, "execution": execution_result})
                return self._handle_error_node(e, ctx_for_error, step="format")
            timings["format"] = time.time() - t6
            result["formatted"] = formatted

            # final timings and return
            total = time.time() - start_all
            timings["total"] = total
            logger.info("GraphBuilder: run complete valid=%s total=%.3fs", result.get("valid"), total)
            return result

        except Exception as e:
            logger.exception("GraphBuilder.run encountered an unexpected error")
            return self._handle_error_node(e, ctx_for_error, step="run")
