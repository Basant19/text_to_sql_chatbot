# File: app/graph/nodes/validate_node.py
"""
ValidateNode
============

Responsibilities:
  - Enforce SQL safety rules (SELECT-only, forbidden tables)
  - Validate referenced tables against known schemas
  - Apply optional LIMIT enforcement
  - Produce structured validation output for downstream nodes
"""

from __future__ import annotations

import sys
import logging
import os
import difflib
from typing import Dict, Any, List, Optional, Set, Tuple

from app.logger import get_logger
from app.exception import CustomException
import app.utils as utils

logger = get_logger("validate_node")
LOG = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Optional SQL parser detection
# ------------------------------------------------------------------
def _has_sqlglot() -> bool:
    try:
        import sqlglot  # type: ignore
        return True
    except Exception:
        return False


# ------------------------------------------------------------------
# Normalize safety rules
# ------------------------------------------------------------------
def _normalize_safety_rules(obj: Any) -> Dict[str, Any]:
    """
    Accepts:
      - dict
      - Tools instance
      - None

    Always returns a dict to avoid attribute errors.
    """
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    try:
        from app.tools import Tools
        if isinstance(obj, Tools):
            rules = obj.get_safety_rules()
            return rules if isinstance(rules, dict) else {}
    except Exception:
        pass

    LOG.warning(
        "ValidateNode received unsupported safety_rules (%s); ignoring.",
        type(obj).__name__,
    )
    return {}


# ------------------------------------------------------------------
# ValidateNode
# ------------------------------------------------------------------
class ValidateNode:
    """
    ValidateNode
    ------------

    Enforces SQL safety and validates table references against known schemas.
    Produces structured output for downstream nodes.
    """

    def __init__(self, safety_rules: Optional[Any] = None):
        try:
            self.safety_rules = _normalize_safety_rules(safety_rules)
            self.max_row_limit: Optional[int] = self.safety_rules.get("max_row_limit")

            forbidden = self.safety_rules.get("forbidden_tables") or []
            self.forbidden_tables = {utils._canonicalize_name(t) for t in forbidden if t}

            self._use_sqlglot = _has_sqlglot()

            LOG.info(
                "ValidateNode initialized | sqlglot=%s | max_row_limit=%s",
                self._use_sqlglot,
                self.max_row_limit,
            )

        except Exception as e:
            logger.exception("ValidateNode initialization failed")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # SQL parsing with sqlglot
    # ------------------------------------------------------------------
    def _extract_with_sqlglot(self, sql: str) -> Tuple[Set[str], Set[str]]:
        try:
            import sqlglot  # type: ignore
            from sqlglot.expressions import Table, Column  # type: ignore

            parsed = sqlglot.parse_one(sql)
            tables: Set[str] = set()
            columns: Set[str] = set()

            for t in parsed.find_all(Table):
                if t.name:
                    tables.add(str(t.name))

            for c in parsed.find_all(Column):
                if c.name:
                    columns.add(str(c.name))
                if c.table:
                    tables.add(str(c.table))

            return tables, columns
        except Exception:
            return set(), set()

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------
    def _build_table_index(self, schemas: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Builds lookup:
          normalized_name -> canonical_table_name
        """
        index: Dict[str, str] = {}
        for canonical, meta in (schemas or {}).items():
            try:
                key = utils._canonicalize_name(canonical)
                if key:
                    index[key] = canonical

                if isinstance(meta, dict):
                    for alias in meta.get("aliases") or []:
                        akey = utils._canonicalize_name(alias)
                        if akey:
                            index[akey] = canonical

                    path = meta.get("path") or meta.get("csv_path")
                    if path:
                        base = os.path.splitext(os.path.basename(path))[0]
                        bkey = utils._canonicalize_name(base)
                        if bkey:
                            index[bkey] = canonical
            except Exception:
                continue
        return index

    def _fuzzy_match(self, name: str, candidates: List[str]) -> Optional[str]:
        match = difflib.get_close_matches(name, candidates, n=1, cutoff=0.75)
        return match[0] if match else None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, sql: str, schemas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate SQL and return structured results:

        Returns:
        {
            "sql": str,
            "valid": bool,
            "errors": List[str],
            "fixes": List[str],
            "suggested_sql": Optional[str],
            "meta": dict
        }
        """
        try:
            sql = (sql or "").strip()
            errors: List[str] = []
            fixes: List[str] = []
            suggested_sql: Optional[str] = None

            if not sql:
                return {
                    "sql": "",
                    "valid": False,
                    "errors": ["SQL query is empty."],
                    "fixes": [],
                    "suggested_sql": None,
                    "meta": {},
                }

            # --------------------------------------------------
            # Enforce SELECT-only queries
            # --------------------------------------------------
            if not utils.is_select_query(sql):
                errors.append("Only SELECT queries are allowed.")

            # --------------------------------------------------
            # Extract tables / columns
            # --------------------------------------------------
            if self._use_sqlglot:
                tables, columns = self._extract_with_sqlglot(sql)
            else:
                tables = set(utils.extract_table_names_from_sql(sql) or [])
                columns = set(utils.extract_column_names_from_sql(sql) or [])

            # --------------------------------------------------
            # Validate tables against known schemas
            # --------------------------------------------------
            table_index = self._build_table_index(schemas)
            known_tables = set(table_index.keys())

            for table in tables:
                norm = utils._canonicalize_name(table)
                if norm in self.forbidden_tables:
                    errors.append(f"Access to table '{table}' is forbidden.")
                    continue
                if norm not in known_tables:
                    guess = self._fuzzy_match(norm, list(known_tables))
                    if guess:
                        fixes.append(f"Interpreted table '{table}' as '{table_index[guess]}'.")
                    else:
                        errors.append(f"Unknown table referenced: '{table}'.")

            # --------------------------------------------------
            # LIMIT enforcement
            # --------------------------------------------------
            if self.max_row_limit and utils.is_select_query(sql):
                try:
                    if utils.exceeds_row_limit(sql, self.max_row_limit):
                        suggested_sql = utils.limit_sql_rows(sql, self.max_row_limit)
                        if suggested_sql:
                            fixes.append(
                                f"Applied LIMIT {self.max_row_limit} to prevent large scans."
                            )
                            sql = suggested_sql
                except Exception:
                    LOG.debug("LIMIT enforcement failed", exc_info=True)

            valid = not errors

            return {
                "sql": sql,
                "valid": valid,
                "errors": errors,
                "fixes": fixes,
                "suggested_sql": suggested_sql,
                "meta": {
                    "tables_found": list(tables),
                    "columns_found": list(columns),
                    "forbidden_tables": list(self.forbidden_tables),
                },
            }

        except CustomException:
            raise
        except Exception as e:
            logger.exception("ValidateNode.run failed")
            raise CustomException(e, sys)
