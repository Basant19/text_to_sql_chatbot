# app/graph/nodes/generate_node.py
import sys
import time
import threading
import json
import re
import os
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
    for i, line in enumerate(lines):
        if sql_kw.match(line):
            # capture until a semicolon or end of block
            collected = [line]
            for nxt in lines[i + 1 : i + 20]:
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
      - Includes schema listing into the prompt (reads schema_store.json if needed).
      - Retries with exponential backoff on provider.generate failures.
      - Robust SQL extraction; store controlled raw blob/meta.
      - Attempts best-effort name-normalization/repair when validation fails.
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
    # SQL repair helpers
    # --------------------------
    def _load_schema_store(self) -> Optional[Dict[str, Any]]:
        """
        Try to load schema store JSON from disk (best-effort). Returns mapping or None.
        """
        try:
            data_dir = getattr(config, "DATA_DIR", "./data")
            store_path = os.path.join(data_dir, "schema_store.json")
            if not os.path.exists(store_path):
                return None
            with open(store_path, "r", encoding="utf-8") as fh:
                items = json.load(fh)
                if isinstance(items, dict):
                    return items
        except Exception as e:
            logger.debug("Failed to load schema_store.json: %s", e)
        return None

    def _attempt_repair_sql(self, sql: str, schemas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Attempt to map missing tokens in sql to canonical store keys and produce a repaired SQL.
        Returns {"repaired_sql": <str>, "mapping": {found_token: canonical_store_key}}.
        This is a best-effort heuristic — it does not guarantee correctness.
        """
        result = {"repaired_sql": sql, "mapping": {}}
        try:
            missing_tables = utils.validate_tables_in_sql(sql, schemas)
            missing_columns = utils.validate_columns_in_sql(sql, schemas)

            if not missing_tables and not missing_columns:
                return result

            # Build index to speed repeated lookups
            # utils.find_matching_schema_key(name, schemas) already does fuzzy/alias lookup
            repaired = sql

            # repair table tokens first
            for tkn in missing_tables:
                try:
                    match_key = utils.find_matching_schema_key(tkn, schemas)
                    if match_key:
                        # Use canonical store value (prefer explicit canonical field)
                        canonical_name = (schemas.get(match_key) or {}).get("canonical") or match_key
                        # Replace tokens bounded by word boundaries (case-insensitive)
                        repaired = re.sub(rf"\b{re.escape(tkn)}\b", canonical_name, repaired, flags=re.IGNORECASE)
                        result["mapping"][tkn] = canonical_name
                except Exception:
                    continue

            # then repair columns: attempt to map column tokens to union columns; if column appears as table.col replace appropriately
            for col in missing_columns:
                try:
                    # Try to find a schema key that contains this column
                    found_key = None
                    for store_key, meta in (schemas or {}).items():
                        cols = []
                        if isinstance(meta, dict):
                            cols = meta.get("columns") or meta.get("columns_normalized") or []
                        elif isinstance(meta, (list, tuple)):
                            cols = meta
                        cols_norm = {utils._canonicalize_name(str(c)): c for c in (cols or [])}
                        if utils._canonicalize_name(col) in cols_norm:
                            found_key = store_key
                            break
                    if found_key:
                        canonical_table = (schemas.get(found_key) or {}).get("canonical") or found_key
                        # replace occurrences of col or table.col with canonical_table.column
                        repaired = re.sub(rf"\b{re.escape(col)}\b", col, repaired, flags=re.IGNORECASE)  # keep col as is
                        # if column used with unknown table qualifier, try to qualify it
                        # e.g. "SELECT app_name FROM apps" -> "SELECT app_name FROM googleplaystore"
                        result["mapping"][col] = f"{canonical_table}.{col}"
                except Exception:
                    continue

            result["repaired_sql"] = repaired
        except Exception as e:
            logger.debug("SQL repair attempt failed: %s", e, exc_info=True)
        return result

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

            # Attempt to include schema store into prompt (if present). This prevents the model from inventing table names.
            schema_text = ""
            schemas_obj = None
            try:
                schemas_obj = self._load_schema_store()
                if schemas_obj:
                    # flatten_schema accepts dict-shaped store
                    schema_text = utils.flatten_schema(schemas_obj)
            except Exception:
                schema_text = ""

            # Compose prompt: system instruction + available schemas + user prompt + firm instruction about canonical names.
            prompt_parts = [system_instruction]
            if schema_text:
                prompt_parts.append("Available tables and columns:\n" + schema_text)
            # include the user-visible query
            prompt_parts.append(f"USER QUERY:\n{prompt}")
            # explicit safety instruction to not invent table names
            prompt_parts.append(
                "IMPORTANT: Use ONLY the table and column names listed above exactly as written. "
                "Do NOT invent or use any other table names (for example: 'apps'). If you must refer to a column, use its exact name."
            )

            prompt_text = "\n\n".join(prompt_parts)

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

            # DEBUG: log selected table names + prompt preview so we can inspect what the model receives.
            try:
                logger.info("Provider call debug: prompt preview (first 1200 chars): %s", prompt_text[:1200])
                if schema_text:
                    logger.info("Provider call debug: schema preview (first 400 chars): %s", schema_text[:400])
                try:
                    with open("debug_prompt.txt", "w", encoding="utf-8") as fh:
                        fh.write(prompt_text)
                except Exception:
                    logger.debug("Could not write debug_prompt.txt")
            except Exception:
                logger.exception("Failed to emit provider debug logs")

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

            # If validation fails, attempt a best-effort repair using the schema store.
            repair_info = {}
            try:
                # prefer schemas_obj loaded from disk; if not present and we have a Tools.schema_store, try that
                schemas_for_validation = schemas_obj
                try:
                    if not schemas_for_validation and hasattr(self.tools, "schema_store") and getattr(self.tools, "schema_store"):
                        # Try to access a ._store attribute if present to get dict shape
                        ss = getattr(self.tools.schema_store, "_store", None)
                        if isinstance(ss, dict) and ss:
                            schemas_for_validation = ss
                except Exception:
                    pass

                if schemas_for_validation:
                    missing_tables = utils.validate_tables_in_sql(extracted_sql, schemas_for_validation)
                    missing_cols = utils.validate_columns_in_sql(extracted_sql, schemas_for_validation)
                    if missing_tables or missing_cols:
                        logger.warning("ValidateNode: validation failed: tables=%s cols=%s", missing_tables, missing_cols)
                        # attempt repair
                        repair = self._attempt_repair_sql(extracted_sql, schemas_for_validation)
                        if repair and repair.get("repaired_sql") and repair.get("repaired_sql") != extracted_sql:
                            logger.info("GenerateNode: SQL repaired via schema mapping: %s", repair.get("mapping", {}))
                            repair_info = repair
                            extracted_sql = repair.get("repaired_sql", extracted_sql)
                else:
                    logger.debug("No schema store available for validation/repair")
            except Exception as e:
                logger.debug("Validation/repair step failed: %s", e, exc_info=True)

            # timings
            timings["total_elapsed"] = time.time() - start_time
            meta["timings"] = timings
            if repair_info:
                meta["repair"] = repair_info

            # tracer (fire-and-forget)
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
