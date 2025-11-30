# app/graph/nodes/validate_node.py
import sys
import logging
from typing import Dict, Any, List, Optional, Set

from app.logger import get_logger
from app.exception import CustomException
from app import utils

logger = get_logger("validate_node")
LOG = logging.getLogger(__name__)


def _use_sqlglot():
    try:
        import sqlglot  # type: ignore
        return True
    except Exception:
        return False


class ValidateNode:
    """
    Node to validate SQL queries against schema and safety rules.

    Improvements:
    - Uses sqlglot (if available) for robust parsing and extraction of tables/columns.
    - Falls back to helpers in app.utils if sqlglot unavailable.
    - Normalizes names case-insensitively and attempts canonical matching.
    - Returns:
        {"sql": <possibly modified sql>,
         "valid": bool,
         "errors": [...],
         "fixes": [...],
         "suggested_sql": <sql if we applied LIMIT>}
    """
    def __init__(self, safety_rules: Optional[Dict[str, Any]] = None):
        try:
            self.safety_rules = safety_rules or {}
            self.max_row_limit = self.safety_rules.get("max_row_limit")
            self.forbidden_tables = [t.lower() for t in self.safety_rules.get("forbidden_tables", [])]
            self._have_sqlglot = _use_sqlglot()
            if self._have_sqlglot:
                LOG.debug("ValidateNode: sqlglot available — using it for parsing.")
            else:
                LOG.debug("ValidateNode: sqlglot not available — falling back to utils functions.")
            logger.info("ValidateNode initialized with safety_rules: %s", self.safety_rules)
        except Exception as e:
            logger.exception("Failed to initialize ValidateNode")
            raise CustomException(e, sys)

    # --------------------------
    # Helpers
    # --------------------------
    def _extract_tables_and_columns_sqlglot(self, sql: str) -> (Set[str], Set[str]):
        """
        Use sqlglot to parse and return referenced table names and column tokens (best-effort).
        Returns (tables_set, columns_set) as lowercase strings.
        """
        try:
            import sqlglot  # type: ignore
            from sqlglot.expressions import Column, Table  # type: ignore

            parsed = sqlglot.parse_one(sql, read=None)  # let sqlglot autodetect dialect
            tables = set()
            cols = set()

            # find Table nodes
            for t in parsed.find_all(Table):
                name = t.name or ""
                if name:
                    tables.add(name.lower())

            # find Column nodes
            for c in parsed.find_all(Column):
                # Column may be like table.column or just column
                col_name = c.name
                if col_name:
                    cols.add(col_name.lower())

            return tables, cols
        except Exception as e:
            LOG.debug("sqlglot parsing failed: %s", e)
            # fallback to empty sets so the outer code uses utils fallback
            return set(), set()

    def _canonical_schema_lookup(self, ref_names: Set[str], schemas: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Given a set of referenced names (lowercased), map them to actual schema keys.
        Returns mapping: ref_name -> matched_store_key
        """
        mapping = {}
        lower_to_key = {k.lower(): k for k in schemas.keys()}
        # try direct lower match first
        for r in ref_names:
            if r in lower_to_key:
                mapping[r] = lower_to_key[r]
        # try more permissive matching using utils._find_matching_key if available
        try:
            find = getattr(utils, "find_matching_schema_key", None)
            if callable(find):
                for r in ref_names:
                    if r in mapping:
                        continue
                    candidate = find(r, schemas)  # expected to return store key or None
                    if candidate:
                        mapping[r] = candidate
        except Exception:
            # ignore - keep mapping as-is
            pass
        return mapping

    # --------------------------
    # Main run
    # --------------------------
    def run(self, sql: str, schemas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        try:
            errors: List[str] = []
            fixes: List[str] = []
            suggested_sql: Optional[str] = None

            orig_sql = (sql or "").strip()
            sql = orig_sql

            # 1) empty query
            if not sql:
                errors.append("SQL query is empty.")
                return {"sql": sql, "valid": False, "errors": errors, "fixes": fixes, "suggested_sql": suggested_sql}

            # 2) Use sqlglot if available to detect statement type and extract refs
            tables_referenced: Set[str] = set()
            columns_referenced: Set[str] = set()

            if self._have_sqlglot:
                try:
                    tset, cset = self._extract_tables_and_columns_sqlglot(sql)
                    tables_referenced = tset
                    columns_referenced = cset
                    LOG.debug("ValidateNode(sqlglot): tables=%s cols=%s", tables_referenced, columns_referenced)
                    # detect non-SELECT: use sqlglot top-level expression type
                    try:
                        import sqlglot as _sqlglot  # type: ignore
                        parsed = _sqlglot.parse_one(sql)
                        root_type = parsed.key or parsed.__class__.__name__
                        # root_type might be 'Select' or other; safer to call utils.is_select_query as fallback
                        if not utils.is_select_query(sql):
                            errors.append("Only SELECT statements are allowed.")
                    except Exception:
                        # fallback to utils.is_select_query
                        if not utils.is_select_query(sql):
                            errors.append("Only SELECT statements are allowed.")
                except Exception:
                    # parsing failed; fallback to utils
                    LOG.debug("sqlglot extraction failed — falling back to utils-based validation")
                    tables_referenced = set()
                    columns_referenced = set()
            else:
                # fallback: rely on utils helpers (they may return lists or sets)
                try:
                    t_missing = utils.extract_table_names_from_sql(sql) if hasattr(utils, "extract_table_names_from_sql") else []
                    # ensure we have a collection
                    tables_referenced = {t.lower() for t in (t_missing or [])}
                except Exception:
                    tables_referenced = set()
                try:
                    c_missing = utils.extract_column_names_from_sql(sql) if hasattr(utils, "extract_column_names_from_sql") else []
                    columns_referenced = {c.lower() for c in (c_missing or [])}
                except Exception:
                    columns_referenced = set()

                # still ensure it's a SELECT
                if not utils.is_select_query(sql):
                    errors.append("Only SELECT statements are allowed.")

            # 3) Validate referenced tables exist (with canonical matching)
            missing_tables = []
            found_table_map = {}  # maps referenced lower name -> store key
            if tables_referenced:
                # map referenced to store keys (case-insensitive / canonical)
                mapping = self._canonical_schema_lookup(tables_referenced, schemas)
                for ref in tables_referenced:
                    if ref in mapping:
                        found_table_map[ref] = mapping[ref]
                    else:
                        # try case-insensitive contains match: look for schema keys that contain the ref
                        matched = None
                        for store_key in schemas.keys():
                            if ref == store_key.lower() or ref in store_key.lower() or store_key.lower() in ref:
                                matched = store_key
                                break
                        if matched:
                            found_table_map[ref] = matched
                        else:
                            missing_tables.append(ref)
            else:
                # If no tables were extracted by parser, attempt a conservative detection via utils.validate_tables_in_sql
                try:
                    miss = utils.validate_tables_in_sql(sql, schemas) or []
                    if miss:
                        missing_tables.extend([m.lower() for m in miss])
                except Exception:
                    # ignore
                    pass

            if missing_tables:
                errors.append(f"Tables not found in schema: {', '.join(sorted(set(missing_tables)))}")
                fixes.append("Ensure the table name in the query matches the uploaded CSV canonical name. Use the sidebar to check 'table names' or the Schema viewer.")

            # 4) Validate referenced columns exist (case-insensitive)
            missing_columns = []
            if columns_referenced:
                # build an aggregate set of valid columns from matched tables
                valid_cols = set()
                # If table mapping found, restrict column set to those tables; otherwise union all schema columns
                if found_table_map:
                    for store_key in set(found_table_map.values()):
                        cols = schemas.get(store_key, {}).get("columns") or []
                        valid_cols.update([c.lower() for c in cols if c])
                else:
                    # union all columns from all schemas
                    for store_key in schemas.keys():
                        cols = schemas.get(store_key, {}).get("columns") or []
                        valid_cols.update([c.lower() for c in cols if c])

                for col in columns_referenced:
                    # allow wildcard like * or function names - skip columns that look like functions
                    if col == "*" or "(" in col or col.endswith(")"):
                        continue
                    if col not in valid_cols:
                        missing_columns.append(col)

            else:
                # fallback: call utils.validate_columns_in_sql to get a list of problem columns (if implemented)
                try:
                    miss = utils.validate_columns_in_sql(sql, schemas) or []
                    if miss:
                        missing_columns.extend([m.lower() for m in miss])
                except Exception:
                    pass

            if missing_columns:
                errors.append(f"Columns not found in schema: {', '.join(sorted(set(missing_columns)))}")
                fixes.append("Check column spellings and that the column exists in the selected CSV. Consider renaming ambiguous columns in the CSV or qualify with table name.")

            # 5) Forbidden tables check
            forbidden_used = []
            if self.forbidden_tables:
                # compare against both found_table_map values and extracted table names
                used_table_names = {v.lower() for v in found_table_map.values()} | {t.lower() for t in tables_referenced}
                for forb in self.forbidden_tables:
                    if forb.lower() in used_table_names:
                        forbidden_used.append(forb)
                if forbidden_used:
                    errors.append(f"Query uses forbidden tables: {', '.join(forbidden_used)}")
                    fixes.append("Remove references to forbidden tables from the query.")

            # 6) Max row limit enforcement (apply LIMIT automatically when configured)
            if self.max_row_limit and utils.is_select_query(sql):
                try:
                    if utils.exceeds_row_limit(sql, self.max_row_limit):
                        # apply automatic limit and offer suggested_sql
                        suggested_sql = utils.limit_sql_rows(sql, self.max_row_limit)
                        fixes.append(f"Applied automatic LIMIT {self.max_row_limit} to avoid large scans.")
                        sql = suggested_sql or sql
                except Exception:
                    # If utils helper not available or failed, skip automatic limiting but warn
                    LOG.debug("Could not evaluate/enforce row limit via utils.exceeds_row_limit", exc_info=True)

            valid = len(errors) == 0

            if valid:
                LOG.info("ValidateNode: SQL validation passed")
            else:
                LOG.warning("ValidateNode: validation failed: %s", errors)

            return {
                "sql": sql,
                "valid": valid,
                "errors": errors,
                "fixes": fixes,
                "suggested_sql": suggested_sql,
            }

        except CustomException:
            raise
        except Exception as e:
            logger.exception("ValidateNode.run failed")
            raise CustomException(e, sys)
