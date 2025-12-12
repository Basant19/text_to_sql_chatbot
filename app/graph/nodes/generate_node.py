#D:\text_to_sql_bot\app\graph\nodes\generate_node.py


from __future__ import annotations
import sys
import logging
import re
from typing import Dict, Any, Optional, Union, List
import inspect
import json
from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

logger = get_logger("generate_node")
LOG = logging.getLogger(__name__)


def _is_langsmith_client(obj: object) -> bool:
    """Heuristic: detect LangSmith client by class name to avoid using it for generation."""
    try:
        name = obj.__class__.__name__.lower()
        return "langsmith" in name
    except Exception:
        return False


def _provider_name(provider: object) -> str:
    try:
        return provider.__class__.__name__
    except Exception:
        return str(type(provider))


class GenerateNode:
    """
    GenerateNode:
      - Wraps the LLM generation step (calls a provider via Tools.get_provider_client).
      - If the configured client is LangSmith (tracer-only), we skip calling its generate().
      - Falls back to app.gemini_client.GeminiClient if available.
    """

    def __init__(self, tools: Optional[Tools] = None, provider_client: Optional[Any] = None):
        try:
            self._tools = tools or Tools()
            # resolution order:
            # 1) explicit provider_client arg
            # 2) tools.get_provider_client()
            # 3) instantiate app.gemini_client.GeminiClient() as fallback
            provider = provider_client if provider_client is not None else None

            if provider is None:
                try:
                    provider = getattr(self._tools, "get_provider_client", lambda: None)()
                except Exception:
                    provider = None

            # If provider is LangSmith client, don't use it for generation (tracing-only)
            if provider is not None and _is_langsmith_client(provider):
                LOG.info(
                    "GenerateNode: detected LangSmith client; skipping it for generation (will act as tracer only)."
                )
                provider = None

            # Last-resort fallback: try to import GeminiClient
            if provider is None:
                try:
                    from app.gemini_client import GeminiClient  # type: ignore

                    try:
                        provider = GeminiClient()
                        LOG.info("GenerateNode: using GeminiClient fallback as provider.")
                    except Exception as e:
                        LOG.debug(
                            "GenerateNode: GeminiClient import succeeded but instantiation failed: %s", e
                        )
                        provider = None
                except Exception:
                    provider = None

            self._provider = provider
            LOG.debug(
                "GenerateNode initialized (provider=%s)",
                _provider_name(self._provider) if self._provider is not None else None,
            )

        except Exception as e:
            logger.exception("Failed to initialize GenerateNode")
            raise CustomException(e, sys)

    def _extract_first_select(self, text: str) -> Optional[str]:
        """
        Try to find the first SELECT statement in `text`.
        Returns the SELECT block (naively up to last semicolon or end).
        """
        if not text:
            return None
        # Normalize whitespace
        t = text.strip()
        # remove common fences like ```sql ... ```
        if t.startswith("```"):
            # remove leading fence
            parts = t.split("```")
            # parts may be ['', 'sql\nSELECT ...', '']
            if len(parts) >= 2:
                t = "```".join(parts[1:]).strip()
                # if remaining again has ``` strip trailing fence
                if t.endswith("```"):
                    t = t[:-3].strip()

        # Remove single backticks wrapping
        t = t.strip("` \n\t")

        # Find the first occurrence of "select" (case-insensitive)
        m = re.search(r"(select\b.*?)(;|$)", t, flags=re.IGNORECASE | re.DOTALL)
        if m:
            sel = m.group(1).strip()
            # if the match contains extra trailing text that is obviously not SQL,
            # we still return the captured select clause.
            return sel if sel else None

        # As fallback, also attempt to locate lines that look like `SELECT ...`
        lines = t.splitlines()
        for i, L in enumerate(lines):
            if re.match(r"^\s*select\b", L, flags=re.IGNORECASE):
                # return from this line to next semicolon or end
                tail = "\n".join(lines[i:])
                m2 = re.search(r"(select\b.*?)(;|$)", tail, flags=re.IGNORECASE | re.DOTALL)
                if m2:
                    return m2.group(1).strip()
                return tail.strip()
        return None

    def run(
        self,
        prompt: Union[str, Dict[str, Any], List[Dict[str, Any]], None],
        metadata: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run generation using the configured provider. Returns dict:
          {"text": <str>, "raw": <raw provider response or None>, "metadata": {...}}
        Behaviors:
          - Normalizes prompt into safe types (str or list-of-messages).
          - Tries multiple provider call patterns adaptively.
          - Returns safe payload with provider_error metadata when provider fails.
        """
        try:
            # no provider configured (e.g. LangSmith used as tracer only)
            if not self._provider:
                LOG.warning(
                    "GenerateNode: no provider client configured for generation (tracer-only LangSmith will not be used)."
                )
                return {"text": "", "raw": None, "metadata": {"provider": None}}

            # ------------------------
            # Prompt normalization
            # ------------------------
            # Accept:
            #  - strings -> trimmed string
            #  - None -> empty string
            #  - list of message dicts (keep as-is for chat)
            #  - dict/other -> convert to safe string
            if prompt is None:
                norm_prompt: Union[str, List[Dict[str, str]]] = ""
            elif isinstance(prompt, str):
                norm_prompt = prompt.strip()
            elif isinstance(prompt, list):
                # assume list of messages (leave to chat path); coerce content to strings
                msgs = []
                for m in prompt:
                    try:
                        role = m.get("role", "user") if isinstance(m, dict) else "user"
                        content = m.get("content", "") if isinstance(m, dict) else str(m)
                        msgs.append({"role": str(role), "content": str(content)})
                    except Exception:
                        # fallback to stringified item
                        msgs.append({"role": "user", "content": str(m)})
                norm_prompt = msgs
            else:
                # dict or other object -> stringify safely for providers that expect text
                try:
                    # if it's a dict, try to extract 'content' or join key/vals
                    if isinstance(prompt, dict):
                        if "content" in prompt and isinstance(prompt["content"], str):
                            norm_prompt = prompt["content"].strip()
                        elif "text" in prompt and isinstance(prompt["text"], str):
                            norm_prompt = prompt["text"].strip()
                        else:
                            # flatten dict to a short string
                            pieces = []
                            for k, v in prompt.items():
                                try:
                                    pieces.append(f"{k}={v}")
                                except Exception:
                                    pieces.append(f"{k}=<unserializable>")
                            norm_prompt = "; ".join(pieces)[:2000]
                    else:
                        # numeric or other -> convert to str
                        norm_prompt = str(prompt)
                except Exception:
                    norm_prompt = str(prompt)

            raw = None
            params = params or {}

            # safe caller that filters kwargs to those accepted by the callable
            def _safe_call(fn, *args, **kwargs):
                try:
                    try:
                        sig = inspect.signature(fn)
                    except Exception:
                        sig = None

                    if sig:
                        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
                        has_var_kw = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                        )
                        if has_var_kw:
                            accepted = {**kwargs}
                        return fn(*args, **accepted)
                    else:
                        return fn(*args, **kwargs)

                except TypeError:
                    # let TypeError bubble to allow pattern fallback
                    raise
                except Exception:
                    LOG.exception("Provider call raised an exception")
                    raise

            # Build attempt patterns
            attempts = []

            if hasattr(self._provider, "generate"):
                # For providers expecting a string, ensure we pass a string
                if isinstance(norm_prompt, list):
                    # pass stringified prompt to generate variants as a fallback
                    text_for_generate = " ".join(m.get("content", "") for m in norm_prompt)
                else:
                    text_for_generate = norm_prompt

                attempts.append(("generate_kw", lambda: _safe_call(self._provider.generate, prompt=text_for_generate, **params)))
                attempts.append(("generate_pos", lambda: _safe_call(self._provider.generate, text_for_generate)))
                attempts.append(("generate_prompt_only", lambda: _safe_call(self._provider.generate, prompt=text_for_generate)))

            if hasattr(self._provider, "create"):
                # create often accepts prompt kw
                text_for_create = norm_prompt if isinstance(norm_prompt, str) else (" ".join(m.get("content", "") for m in norm_prompt) if isinstance(norm_prompt, list) else str(norm_prompt))
                attempts.append(("create_kw", lambda: _safe_call(self._provider.create, prompt=text_for_create, **params)))

            if hasattr(self._provider, "chat"):
                # chat typically expects a list of message dicts
                if isinstance(norm_prompt, list):
                    messages = norm_prompt
                else:
                    messages = [{"role": "user", "content": str(norm_prompt)}]
                attempts.append((
                    "chat_kw",
                    lambda: _safe_call(self._provider.chat, messages=messages, **params),
                ))

            if hasattr(self._provider, "run"):
                # run usually accepts a plain string
                run_arg = norm_prompt if isinstance(norm_prompt, str) else (" ".join(m.get("content", "") for m in norm_prompt) if isinstance(norm_prompt, list) else str(norm_prompt))
                attempts.append(("run_pos", lambda: _safe_call(self._provider.run, run_arg)))

            if callable(self._provider):
                # provider as callable (LLM wrappers)
                callable_arg = norm_prompt if isinstance(norm_prompt, str) else (" ".join(m.get("content", "") for m in norm_prompt) if isinstance(norm_prompt, list) else str(norm_prompt))
                attempts.append(("callable", lambda: _safe_call(self._provider, callable_arg, **params)))
                attempts.append(("callable_prompt_only", lambda: _safe_call(self._provider, callable_arg)))

            last_exc = None
            for name, fn in attempts:
                try:
                    LOG.debug("GenerateNode: attempting provider call pattern=%s", name)
                    raw = fn()
                    LOG.debug("GenerateNode: provider call succeeded with pattern=%s", name)
                    break
                except TypeError as te:
                    # TypeError often means wrong signature for this attempt — continue to next pattern
                    LOG.debug("GenerateNode: provider call pattern=%s TypeError: %s", name, te)
                    last_exc = te
                    continue
                except Exception as e:
                    # normalize message for checks
                    msg = str(e) or ""
                    lower = msg.lower()

                    # if provider is actually LangSmith and complains about disabled generate, skip it gracefully
                    if "langsmith" in lower and "disabled" in lower:
                        LOG.warning("GenerateNode: encountered LangSmith disabled-generate; skipping provider. message=%s", msg)
                        last_exc = e
                        continue

                    # if provider complains about unexpected kw arg, try next pattern
                    if "unexpected keyword argument" in lower or "unexpected keyword" in lower:
                        LOG.debug("GenerateNode: provider rejected kwargs for pattern=%s: %s", name, msg)
                        last_exc = e
                        continue

                    # Other errors: log and continue trying other patterns
                    LOG.exception("GenerateNode: provider call pattern=%s raised exception", name)
                    last_exc = e
                    continue

            # If nothing succeeded
            if raw is None:
                LOG.error("GenerateNode: all provider call attempts failed. last_exc=%s", last_exc)
                return {
                    "text": "",
                    "raw": None,
                    "metadata": {
                        "provider": _provider_name(self._provider),
                        "provider_error": str(last_exc) if last_exc else None,
                    },
                }

            # If provider returned an error payload like {"error": "..."} give a safe response
            if isinstance(raw, dict) and "error" in raw and raw.get("error"):
                err_text = raw.get("error")
                LOG.error("GenerateNode: provider returned error payload: %s", err_text)
                return {
                    "text": "",
                    "raw": raw,
                    "metadata": {"provider": _provider_name(self._provider), "provider_error": str(err_text)},
                }

            # Normalize output to text
            text = ""
            if isinstance(raw, dict):
                # attempt common keys
                # prefer nested shapes: candidates, outputs, etc.
                # we will leave raw intact for debugging
                # attempt to extract textual content heuristically
                possible = []
                for key in ("text", "output", "result", "content"):
                    v = raw.get(key)
                    if isinstance(v, str) and v.strip():
                        possible.append(v.strip())
                # check 'candidates' or 'choices' lists
                for key in ("choices", "candidates", "results", "outputs"):
                    v = raw.get(key)
                    if isinstance(v, (list, tuple)) and v:
                        first = v[0]
                        if isinstance(first, dict):
                            for k in ("text", "output", "content", "message"):
                                vv = first.get(k)
                                if isinstance(vv, str) and vv.strip():
                                    possible.append(vv.strip())
                                    break
                        elif isinstance(first, str):
                            possible.append(first.strip())
                if possible:
                    text = possible[0]
                else:
                    # fallback to JSON dump
                    try:
                        text = json.dumps(raw)
                    except Exception:
                        text = str(raw)

            elif isinstance(raw, (list, tuple)):
                # join list-like results
                # try to pick first string-like candidate
                for item in raw:
                    if isinstance(item, str) and item.strip():
                        text = item.strip()
                        break
                if not text:
                    text = " ".join(str(x) for x in raw)
            elif raw is not None:
                text = str(raw)

            # Post-process text: strip fences and extract SELECT if present
            extracted_select = None
            if isinstance(text, str):
                # common markdown/code fences
                t = text.strip()
                # strip leading/trailing whitespace
                # if text contains ```sql or ``` remove fences
                t = re.sub(r"^```(?:sql)?\s*", "", t, flags=re.IGNORECASE)
                t = re.sub(r"\s*```$", "", t)
                # remove surrounding triple backticks left if any (again)
                t = t.strip("` \n\t")
                # attempt to extract first SELECT
                extracted_select = self._extract_first_select(t)
                if extracted_select:
                    text = extracted_select
                else:
                    # if the provider returned an explicit machine fallback token:
                    if t.strip().upper().startswith("CANNOT_ANSWER") or t.strip().upper().startswith("I CANNOT"):
                        # mark as non-answer
                        LOG.debug("GenerateNode: provider indicates cannot answer: %s", t[:200])
                        return {
                            "text": "",
                            "raw": raw,
                            "metadata": {"provider": _provider_name(self._provider), "provider_error": "CANNOT_ANSWER", **(metadata or {})},
                        }
                    text = t

            return {
                "text": text,
                "raw": raw,
                "metadata": {"provider": _provider_name(self._provider), **(metadata or {})},
            }

        except Exception as e:
            LOG.exception("GenerateNode.run failed")
            raise CustomException(e, sys)
