# app/graph/nodes/generate_node.py
from __future__ import annotations
import sys
import logging
from typing import Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

logger = get_logger("generate_node")
LOG = logging.getLogger(__name__)


class GenerateNode:
    """
    GenerateNode:
      - Wraps the LLM generation step (calls a provider via Tools.get_provider_client or Tools._provider_client).
      - Accepts a prompt dict/context and optional params and returns LLM output.
      - This node is purposely thin; real logic (prompt templates, retries) live in llm_flow or provider client.
    """

    def __init__(self, tools: Optional[Tools] = None, provider_client: Optional[Any] = None):
        try:
            self._tools = tools or Tools()
            self._provider = provider_client or getattr(self._tools, "get_provider_client", lambda: getattr(self._tools, "_provider_client", None))()
        except Exception as e:
            logger.exception("Failed to initialize GenerateNode")
            raise CustomException(e, sys)

    def run(self, prompt: str, metadata: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Return structure: {"text": <str>, "raw": <provider_raw>, "metadata": {...}}
        """
        try:
            if not self._provider:
                LOG.warning("GenerateNode: no provider client configured")
                # return a safe empty structure
                return {"text": "", "raw": None, "metadata": {"provider": None}}

            # Preferred API: provider.generate(prompt, params)
            try:
                if hasattr(self._provider, "generate"):
                    raw = self._provider.generate(prompt=prompt, params=params or {})
                elif callable(self._provider):
                    raw = self._provider(prompt, params or {})
                else:
                    # attempt common method names
                    raw = getattr(self._provider, "run", None) and self._provider.run(prompt)
            except Exception:
                LOG.exception("Provider generation failed for prompt")
                raw = None

            # Normalize output (best-effort)
            text = ""
            if isinstance(raw, dict):
                text = raw.get("text") or raw.get("output") or ""
            elif isinstance(raw, (list, tuple)):
                # join if multiple parts
                text = " ".join([str(x) for x in raw])
            elif raw is not None:
                text = str(raw)

            return {"text": text, "raw": raw, "metadata": metadata or {}}
        except Exception as e:
            logger.exception("GenerateNode.run failed")
            raise CustomException(e, sys)
