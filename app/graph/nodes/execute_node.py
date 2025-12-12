# File: app/graph/nodes/execute_node.py
from __future__ import annotations
import sys
import time
import re
import logging
import os
from typing import Dict, Any, Optional

from app.logger import get_logger
from app.exception import CustomException
from app.tools import Tools

logger = get_logger("execute_node")
LOG = logging.getLogger(__name__)

# Disallowed SQL operations in read-only mode (word-boundary aware)
_READONLY_DISALLOWED = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bUPDATE\b",
    r"\bINSERT\b",
    r"\bALTER\b",
    r"\bTRUNCATE\b",
    r"\bCREATE\b",
    r"\bATTACH\b",
    r"\bDETACH\b",
    r"\bREPLACE\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
]


def _is_read_only_sql(sql: str) -> bool:
    s = sql or ""
    for patt in _READONLY_DISALLOWED:
        if re.search(patt, s, flags=re.IGNORECASE):
            return False
    return True


def _sanitize_sql(sql: str) -> str:
    """Normalize SQL: strip trailing semicolons, collapse whitespace, remove backticks."""
    if not sql:
        return sql
    sql = sql.replace("`", "")
    sql = sql.strip()
    sql = re.sub(r";+$", "", sql)
    sql = re.sub(r"\s+", " ", sql)
    return sql


class ExecuteNode:
    """
    ExecuteNode:
      - Executes SQL using Tools.execute_sql or Tools._executor as fallback.
      - Enforces read_only by default and supports limit injection by caller.
      - Normalizes results into {'rows': [...], 'columns': [...], 'rowcount': N}
    """

    def __init__(self, tools: Optional[Tools] = None, default_limit: Optional[int] = None):
        try:
            self._tools = tools or Tools()
            self._default_limit = default_limit
        except Exception as e:
            logger.exception("Failed to initialize ExecuteNode")
            raise CustomException(e, sys)

    def _format_result(self, res: Any, as_dataframe: bool = False) -> Dict[str, Any]:
        """Normalize possible executor return shapes into a consistent dict."""
        try:
            if isinstance(res, dict):
                rows = res.get("rows") or res.get("data") or []
                columns = res.get("columns") or res.get("cols") or []
                rowcount = int(res.get("rowcount") or (len(rows) if hasattr(rows, "__len__") else 0))
                df = res.get("as_dataframe") if res.get("as_dataframe") is not None else None
                return {"rows": rows, "columns": columns, "rowcount": rowcount, "as_dataframe": df}

            # pandas DataFrame
            try:
                import pandas as pd

                if pd and isinstance(res, pd.DataFrame):
                    rows = res.to_dict(orient="records")
                    columns = list(res.columns)
                    rowcount = len(res)
                    return {"rows": rows, "columns": columns, "rowcount": rowcount, "as_dataframe": res if as_dataframe else None}
            except Exception:
                pass

            if isinstance(res, (list, tuple)):
                rows = list(res)
                columns = []
                if rows and isinstance(rows[0], dict):
                    columns = list(rows[0].keys())
                rowcount = len(rows)
                return {"rows": rows, "columns": columns, "rowcount": rowcount, "as_dataframe": None}

            # Single scalar fallback
            return {"rows": [res], "columns": [], "rowcount": 1, "as_dataframe": None}
        except Exception as e:
            LOG.exception("ExecuteNode._format_result failed: %s", e)
            return {"rows": [], "columns": [], "rowcount": 0, "as_dataframe": None}

    def _enrich_read_only_mapping_with_paths(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given a mapping canonical->metadata (may not include 'path'), try to enrich each entry
        by querying the SchemaStore via Tools if available. Returns a new mapping with absolute paths.
        """
        if not isinstance(mapping, dict):
            return mapping
        enriched: Dict[str, Any] = {}
        try:
            ss = getattr(self._tools, "_schema_store", None)
            for key, meta in mapping.items():
                try:
                    entry = dict(meta) if isinstance(meta, dict) else {"columns": meta}
                    if not entry.get("path"):
                        path = None
                        if ss is not None:
                            for fn in ("get_path", "get_csv_path", "get_file", "get_file_path"):
                                if hasattr(ss, fn):
                                    try:
                                        p = getattr(ss, fn)(key)
                                        if p:
                                            path = p
                                            break
                                    except Exception:
                                        continue

                            if not path:
                                for fn in ("get_entry", "get", "get_metadata", "get_schema_entry"):
                                    if hasattr(ss, fn):
                                        try:
                                            ent = getattr(ss, fn)(key)
                                            if isinstance(ent, dict):
                                                for k in ("path", "csv_path", "file", "source"):
                                                    if ent.get(k):
                                                        path = ent.get(k)
                                                        break
                                            if path:
                                                break
                                        except Exception:
                                            continue

                            if not path and hasattr(ss, "schemas"):
                                try:
                                    s = getattr(ss, "schemas")
                                    if isinstance(s, dict) and key in s:
                                        ent = s.get(key)
                                        if isinstance(ent, dict):
                                            for k in ("path", "csv_path", "file", "source"):
                                                if ent.get(k):
                                                    path = ent.get(k)
                                                    break
                                except Exception:
                                    pass

                        if path:
                            try:
                                entry["path"] = os.path.abspath(path)
                            except Exception:
                                entry["path"] = path

                    enriched[key] = entry
                except Exception:
                    enriched[key] = meta
        except Exception:
            LOG.debug("ExecuteNode: enrichment of read_only mapping failed", exc_info=True)
            return mapping
        return enriched

    def run(self, sql: str, read_only: bool = True, limit: Optional[int] = None, as_dataframe: bool = False) -> Dict[str, Any]:
        try:
            if not sql:
                return {"rows": [], "columns": [], "rowcount": 0, "as_dataframe": None}

            sql_original = str(sql)
            sql = _sanitize_sql(sql_original)

            effective_limit = limit if limit is not None else self._default_limit

            # inject naive LIMIT if requested and not present
            if effective_limit is not None and not re.search(r"\blimit\b", sql, flags=re.IGNORECASE):
                sql = f"{sql} LIMIT {int(effective_limit)}"

            # Enforce read-only
            if read_only:
                if not _is_read_only_sql(sql):
                    LOG.error("ExecuteNode: blocked non-read-only SQL: %s", sql)
                    raise CustomException("SQL includes disallowed operations in read-only mode", sys)

            start = time.time()

            # If read_only is a dict mapping of schemas, try to enrich it with CSV paths using SchemaStore.
            read_only_arg = read_only
            try:
                if isinstance(read_only, dict):
                    read_only_arg = self._enrich_read_only_mapping_with_paths(read_only)
                    LOG.debug("ExecuteNode: enriched read_only mapping keys=%s", list(read_only_arg.keys()))
            except Exception:
                LOG.debug("ExecuteNode: failed to enrich read_only mapping", exc_info=True)

            # Try preferred Tools.execute_sql
            try:
                if hasattr(self._tools, "execute_sql"):
                    result = self._tools.execute_sql(sql, read_only=read_only_arg, limit=effective_limit, as_dataframe=as_dataframe)
                    runtime = time.time() - start
                    LOG.info("ExecuteNode: executed via Tools.execute_sql in %.3fs", runtime)
                    return self._format_result(result, as_dataframe=as_dataframe)

                # Fallback to Tools._executor object
                executor = getattr(self._tools, "_executor", None)
                if executor:
                    # If executor provides execute_sql
                    if hasattr(executor, "execute_sql"):
                        result = executor.execute_sql(sql, read_only=read_only_arg, limit=effective_limit, as_dataframe=as_dataframe)
                        runtime = time.time() - start
                        LOG.info("ExecuteNode: executed via executor.execute_sql in %.3fs", runtime)
                        return self._format_result(result, as_dataframe=as_dataframe)

                    # If executor is a function/callable
                    if callable(executor):
                        result = executor(sql, read_only=read_only_arg, limit=effective_limit, as_dataframe=as_dataframe)
                        runtime = time.time() - start
                        LOG.info("ExecuteNode: executed via executor callable in %.3fs", runtime)
                        return self._format_result(result, as_dataframe=as_dataframe)

            except CustomException:
                raise
            except Exception as e:
                LOG.exception("ExecuteNode: execution raised an exception for SQL: %s", sql)
                raise

            # No executor available
            LOG.error("ExecuteNode: no executor available to run SQL")
            raise CustomException("No executor available to run SQL", sys)

        except CustomException:
            raise
        except Exception as e:
            logger.exception("ExecuteNode.run failed")
            raise CustomException(e, sys)
