# D:\text_to_sql_bot\app\graph\nodes\validate_node.py
"""
ValidateNode
============

FINAL SQL CANONICALIZATION & ENFORCEMENT LAYER

Responsibilities
----------------
- Enforce read-only SQL (SELECT / WITH / EXPLAIN)
- Validate referenced tables against SchemaStore
- Rewrite canonical / alias table names → runtime table names
- Validate and rewrite column identifiers → EXACT CSV headers
- Block unknown or forbidden tables / columns
- Produce UI-friendly validation output (errors + fixes)

Design Principles
-----------------
- NEVER execute SQL
- NEVER guess silently
- NEVER allow non-runtime tables to reach execution
- All identifier rewriting happens HERE (tables + columns)
"""

from __future__ import annotations

import sys
import logging
import os
import difflib
import re
from typing import Dict, Any, List, Optional, Set, Tuple

from app.logger import get_logger
from app.exception import CustomException
import app.utils as utils

logger = get_logger("validate_node")
LOG = logging.getLogger(__name__)


# ==============================================================
# Optional SQL parser detection
# ==============================================================
def _has_sqlglot() -> bool:
    try:
        import sqlglot  # type: ignore
        return True
    except Exception:
        return False


# ==============================================================
# Safety rule normalization
# ==============================================================
def _normalize_safety_rules(obj: Any) -> Dict[str, Any]:
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
        "Unsupported safety_rules input (%s); ignoring",
        type(obj).__name__,
    )
    return {}


# ==============================================================
# Helper: identifier normalization
# ==============================================================
def _normalize_identifier(name: str) -> str:
    """
    Normalize identifiers for matching only.
    NEVER used for execution.
    """
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_"))


def _quote_identifier(name: str) -> str:
    """
    Quote identifiers for DuckDB-safe execution.
    """
    if name.startswith('"') and name.endswith('"'):
        return name
    return f'"{name}"'


# ==============================================================
# ValidateNode
# ==============================================================
class ValidateNode:
    """
    Enforces schema correctness and final SQL rewriting.
    """

    def __init__(self, safety_rules: Optional[Any] = None):
        try:
            self.safety_rules = _normalize_safety_rules(safety_rules)

            self.max_row_limit: Optional[int] = self.safety_rules.get(
                "max_row_limit"
            )

            forbidden = self.safety_rules.get("forbidden_tables") or []
            self.forbidden_tables: Set[str] = {
                _normalize_identifier(t)
                for t in forbidden
                if t
            }

            self._use_sqlglot = _has_sqlglot()

            LOG.info(
                "ValidateNode initialized | sqlglot=%s | max_row_limit=%s | forbidden_tables=%d",
                self._use_sqlglot,
                self.max_row_limit,
                len(self.forbidden_tables),
            )

        except Exception as e:
            logger.exception("ValidateNode initialization failed")
            raise CustomException(e, sys)

    # ==========================================================
    # SQL parsing
    # ==========================================================
    def _extract_with_sqlglot(
        self, sql: str
    ) -> Tuple[Set[str], Set[str]]:
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
            LOG.debug("sqlglot parsing failed; falling back")
            return set(), set()

    # ==========================================================
    # Schema helpers
    # ==========================================================
    def _build_table_maps(
        self, schemas: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Returns:
        - normalized_name → canonical_table
        - canonical_table → runtime_table
        """
        name_to_canonical: Dict[str, str] = {}
        canonical_to_runtime: Dict[str, str] = {}

        for canonical, meta in (schemas or {}).items():
            if not isinstance(meta, dict):
                continue

            runtime = meta.get("runtime_table")
            if not runtime:
                continue

            canonical_key = _normalize_identifier(canonical)
            name_to_canonical[canonical_key] = canonical
            canonical_to_runtime[canonical] = runtime

            display = meta.get("display_name")
            if display:
                name_to_canonical[
                    _normalize_identifier(display)
                ] = canonical

            for alias in meta.get("aliases") or []:
                name_to_canonical[
                    _normalize_identifier(alias)
                ] = canonical

            path = meta.get("path") or meta.get("csv_path")
            if path:
                base = os.path.splitext(os.path.basename(path))[0]
                name_to_canonical[
                    _normalize_identifier(base)
                ] = canonical

        return name_to_canonical, canonical_to_runtime

    def _build_column_map(
        self, schemas: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Build:
        normalized_column → EXACT CSV column name
        """
        col_map: Dict[str, str] = {}

        for meta in (schemas or {}).values():
            cols = meta.get("columns") or []
            for col in cols:
                norm = _normalize_identifier(col)
                col_map[norm] = col

        return col_map

    def _fuzzy_match(
        self, name: str, candidates: List[str]
    ) -> Optional[str]:
        match = difflib.get_close_matches(
            name, candidates, n=1, cutoff=0.75
        )
        return match[0] if match else None

    # ==========================================================
    # Graph entrypoint
    # ==========================================================
    def run(
        self,
        *,
        sql: Optional[str] = None,
        schemas: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        try:
            sql = (sql or "").strip()
            schemas = schemas or {}

            errors: List[str] = []
            fixes: List[str] = []
            rewritten_sql = sql

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
            # Read-only enforcement
            # --------------------------------------------------
            if not utils.is_select_query(sql):
                errors.append(
                    "Only SELECT / WITH / EXPLAIN queries are allowed."
                )

            # --------------------------------------------------
            # Extract identifiers
            # --------------------------------------------------
            if self._use_sqlglot:
                tables, columns = self._extract_with_sqlglot(sql)
            else:
                tables = set(
                    utils.extract_table_names_from_sql(sql) or []
                )
                columns = set(
                    utils.extract_column_names_from_sql(sql) or []
                )

            # --------------------------------------------------
            # Table enforcement + rewrite
            # --------------------------------------------------
            name_to_canonical, canonical_to_runtime = (
                self._build_table_maps(schemas)
            )

            for table in tables:
                norm = _normalize_identifier(table)

                if norm in self.forbidden_tables:
                    errors.append(
                        f"Access to table '{table}' is forbidden."
                    )
                    continue

                if norm not in name_to_canonical:
                    errors.append(
                        f"Unknown table referenced: '{table}'."
                    )
                    continue

                canonical = name_to_canonical[norm]
                runtime = canonical_to_runtime.get(canonical)

                if table != runtime:
                    rewritten_sql = utils.replace_table_name(
                        rewritten_sql,
                        table,
                        runtime,
                    )
                    fixes.append(
                        f"Rewrote table '{table}' → '{runtime}'."
                    )

            # --------------------------------------------------
            # Column enforcement + rewrite (CRITICAL FIX)
            # --------------------------------------------------
            column_map = self._build_column_map(schemas)

            for col in columns:
                norm = _normalize_identifier(col)

                if norm not in column_map:
                    errors.append(
                        f"Unknown column referenced: '{col}'."
                    )
                    continue

                exact = column_map[norm]
                quoted = _quote_identifier(exact)

                if col != quoted and col != exact:
                    rewritten_sql = utils.replace_column_name(
                        rewritten_sql,
                        col,
                        quoted,
                    )
                    fixes.append(
                        f"Rewrote column '{col}' → {quoted}."
                    )

            valid = not errors

            return {
                "sql": rewritten_sql,
                "valid": valid,
                "errors": errors,
                "fixes": fixes,
                "suggested_sql": None,
                "meta": {
                    "tables_found": list(tables),
                    "columns_found": list(columns),
                },
            }

        except CustomException:
            raise
        except Exception as e:
            logger.exception("ValidateNode.run failed")
            raise CustomException(e, sys)
