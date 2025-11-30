# app/graph/nodes/generate_node.py
import sys
import time
import threading
import json
import re
from typing import Optional, Dict, Any, Union

from app.logger import get_logger
from app.exception import CustomException
from app import utils, config
from app.tools import Tools

logger = get_logger("generate_node")


_DEFAULT_RETRIES = getattr(config, "GEN_RETRIES", 2)
_DEFAULT_TIMEOUT = getattr(config, "GEN_TIMEOUT", 30)
_DEFAULT_BACKOFF = getattr(config, "GEN_BACKOFF", 1.0)
_DEFAULT_MAX_RAW_SUMMARY = getattr(config, "MAX_RAW_SUMMARY_CHARS", 800)


def _safe_serialize_raw(raw: Any) -> Any:
    """
    Try to make 'raw' JSON-serializable for storing in meta (used when STORE_FULL_LLM_BLOBS=False).
    """
    try:
        return json.loads(json.dumps(raw, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        try:
            return str(raw)[:_DEFAULT_MAX_RAW_SUMMARY]
        except Exception:
            return "<unserializable raw response>"


def _extract_sql_from_text(text: str) -> Optional[str]:
    """
    Robust SQL extraction:
      1) ```sql ... ```
      2) JSON {"sql": "..."}
      3) first SQL-looking statement (heuristic)
    """
    if not text:
        return None

    # 1) sql code fence
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        if candidate:
            return candidate

    # 2) JSON object with "sql" key
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for k in ("sql", "query", "query_text", "output"):
                if k in parsed and isinstance(parsed[k], str):
                    return parsed[k].strip()
    except Exception:
        pass

    # 3) Heuristic: first line/paragraph that starts with common SQL keywords
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    sql_kw = re.compile(r"^(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b", flags=re.IGNORECASE)
    for line in lines:
        if sql_kw.match(line):
            # capture until a semicolon or end of block
            # attempt to join following lines until semicolon
            idx = lines.index(line)
            collected = [line]
            for nxt in lines[idx + 1 : idx + 20]:
                collected.append(nxt)
                if ";" in nxt:
                    break
            return "\n".join(collected).strip()

    # 4) fallback to utils.extract_sql_from_text (existing heuristic)
    try:
        fallback = utils.extract_sql_from_text(text)
        if fallback:
            return fallback.strip()
    except Exception:
        pass

    return None


class GenerateNode:
    """
    Node responsible for generating SQL from a prompt via an LLM.

    Behavior summary:
      - Strong system instruction appended to user prompt to force SQL-only outputs.
      - Retries with exponential backoff on provider.generate failures.
      - Robust SQL extraction; store controlled raw blob/meta.
      - Fire-and-forget tracer call for observability (does not affect result).
    """

    def __init__(
        self,
        tools: Optional[Tools] = None,
        client: Optional[Any] = None,
        provider_client: Optional[Any] = None,
        tracer_client: Optional[Any] = None,
        model: Optional[str] = None,
        max_tokens: int = 512,
        retries: int = _DEFAULT_RETRIES,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        try:
            self.tools = tools or Tools()
            self.client = client
            self.provider_client = provider_client
            self.tracer_client = tracer_client
            self.model = model or getattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
            self.max_tokens = max_tokens
            self.retries = retries
            self.timeout = timeout

            logger.info(
                "GenerateNode initialized (model=%s, max_tokens=%s, retries=%s, timeout=%s, has_provider=%s, has_tracer=%s)",
                self.model,
                self.max_tokens,
                self.retries,
                self.timeout,
                bool(self.provider_client or self.client),
                bool(self.tracer_client),
            )
        except Exception as e:
            logger.exception("Failed to initialize GenerateNode")
            raise CustomException(e, sys)

    # --------------------------
    # Tracing (fire-and-forget)
    # --------------------------
    def _call_tracer(self, prompt: str, sql: Optional[str], meta: Dict[str, Any]) -> None:
        if not self.tracer_client:
            return

        def _trace():
            try:
                if hasattr(self.tracer_client, "trace_run"):
                    payload = {
                        "model": self.model,
                        "prompt_snapshot": (prompt[:2000] + "...") if len(prompt) > 2000 else prompt,
                        "sql_present": bool(sql),
                        "meta_summary": {k: meta.get(k) for k in ("raw_summary", "extraction") if k in meta},
                        "timestamp": time.time(),
                    }
                    try:
                        self.tracer_client.trace_run(name="generate_node.run", **payload)
                    except TypeError:
                        # Some tracer APIs expect (name, payload)
                        try:
                            self.tracer_client.trace_run("generate_node.run", payload)
                        except Exception:
                            pass
            except Exception as e:
                logger.debug("Tracer error ignored: %s", e)

        threading.Thread(target=_trace, daemon=True).start()

    # --------------------------
    # Provider call with retries
    # --------------------------
    def _call_provider_generate(self, provider: Any, prompt: str) -> Dict[str, Any]:
        last_exc = None
        backoff = getattr(config, "GEN_BACKOFF", _DEFAULT_BACKOFF)
        for attempt in range(max(1, self.retries + 1)):
            try:
                # provider.generate expected to return dict-like {"text":..., "raw":...} or string
                if hasattr(provider, "generate") and callable(provider.generate):
                    out = provider.generate(prompt, model=self.model, max_tokens=self.max_tokens, timeout=self.timeout)
                elif hasattr(provider, "run") and callable(provider.run):
                    out = provider.run(prompt)
                else:
                    raise CustomException("Provider does not implement generate/run", sys)

                # normalize to dict with 'text' and 'raw'
                if isinstance(out, dict):
                    text = out.get("text") or out.get("output") or out.get("content") or ""
                    raw = out.get("raw", out)
                else:
                    text = str(out)
                    raw = out
                return {"text": str(text), "raw": raw}
            except Exception as e:
                last_exc = e
                logger.warning("Generate attempt %d/%d failed: %s", attempt + 1, max(1, self.retries + 1), e)
                # exponential backoff (but don't sleep after last attempt)
                if attempt < self.retries:
                    time.sleep(backoff * (2 ** attempt))
                continue
        # all attempts exhausted
        logger.exception("All provider.generate attempts failed")
        raise CustomException(last_exc or RuntimeError("generate failed"), sys)

    # --------------------------
    # Core run()
    # --------------------------
    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Generate SQL from prompt.

        Returns:
          {
            "prompt": <prompt used>,
            "raw": <provider raw object or raw_summary>,
            "sql": <extracted SQL string or ''>,
            "meta": { ... }   # includes extraction info, raw_summary or raw_blob_path, timings
          }
        """
        start_time = time.time()
        timings = {"start": start_time}
        try:
            # build a strict instruction wrapper that prefers a single SQL output
            system_instruction = (
                "You are a SQL generation assistant. Given the user's question and the provided table schemas, "
                "output only the SQL query that answers the user's question. "
                "Prefer returning the SQL inside a triple-backtick block annotated as ```sql ... ``` or as a JSON object like {\"sql\":\"...\"}. "
                "Do NOT include additional commentary, explanation, or table creation statements. "
                "If you cannot answer, return an empty string for sql."
            )

            # combine instruction + user prompt (keep both visible)
            prompt_text = f"{system_instruction}\n\nUSER QUERY:\n{prompt}"

            # choose provider (prefer provider_client, then client, then lazy GeminiClient)
            provider = self.provider_client or self.client
            if provider is None:
                try:
                    from app.gemini_client import GeminiClient  # lazy import
                    provider = GeminiClient()
                    logger.debug("GenerateNode: lazily instantiated GeminiClient")
                except Exception as e:
                    logger.exception("Failed to instantiate GeminiClient lazily")
                    raise CustomException(e, sys)

            # call provider with retries
            t0 = time.time()
            resp = self._call_provider_generate(provider, prompt_text)
            timings["provider_time"] = time.time() - t0

            text = resp.get("text") if isinstance(resp, dict) else str(resp)
            raw = resp.get("raw", resp)

            # extract SQL robustly
            extracted_sql = _extract_sql_from_text(text) or ""
            # fallback to utils helper in case extraction not successful
            if not extracted_sql:
                try:
                    fallback = utils.extract_sql_from_text(text)
                    if fallback:
                        extracted_sql = fallback
                except Exception:
                    pass

            # prepare meta
            meta: Dict[str, Any] = {}
            store_full = getattr(config, "STORE_FULL_LLM_BLOBS", False)
            if store_full:
                # attempt to store raw blob on disk (blobs/ directory)
                try:
                    data_dir = getattr(config, "DATA_DIR", "./data")
                    blobs_dir = os.path.join(data_dir, "blobs")
                    os.makedirs(blobs_dir, exist_ok=True)
                    blob_name = f"llm_raw_{int(time.time())}_{uuid_short()}.json"
                    blob_path = os.path.join(blobs_dir, blob_name)
                    with open(blob_path, "w", encoding="utf-8") as f:
                        json.dump(_safe_serialize_raw(raw), f, ensure_ascii=False, indent=2)
                    meta["raw_blob_path"] = blob_path
                    meta["raw_summary"] = str(text)[:_DEFAULT_MAX_RAW_SUMMARY]
                except Exception as e:
                    logger.exception("Failed to persist raw LLM blob; storing summary instead")
                    meta["raw_summary"] = str(text)[:_DEFAULT_MAX_RAW_SUMMARY]
            else:
                # do not keep full blob, only a sanitized summary
                meta["raw_summary"] = str(text)[:_DEFAULT_MAX_RAW_SUMMARY]
                meta["raw_preview"] = _safe_serialize_raw(raw)

            # extraction diagnostics
            meta["extraction"] = {
                "method": "fenced/json/heuristic",
                "sql_found": bool(extracted_sql),
                "extracted_sql_length": len(extracted_sql or ""),
            }

            # timings
            timings["total_elapsed"] = time.time() - start_time
            meta["timings"] = timings

            # tracer
            try:
                self._call_tracer(prompt_text, extracted_sql or None, meta)
            except Exception:
                logger.debug("Tracer call failed silently")

            # return structured result
            result = {
                "prompt": prompt_text,
                "raw": meta.get("raw_preview", meta.get("raw_summary", "<no raw>")),
                "sql": extracted_sql,
                "meta": meta,
            }

            logger.info("GenerateNode completed: sql_len=%d provider_time=%.3fs total=%.3fs", len(extracted_sql or ""), timings.get("provider_time", 0.0), timings["total_elapsed"])
            return result

        except CustomException:
            raise
        except Exception as e:
            logger.exception("GenerateNode.run failed")
            raise CustomException(e, sys)


# small helper for short unique id (no extra imports)
def uuid_short() -> str:
    import uuid
    return uuid.uuid4().hex[:8]
