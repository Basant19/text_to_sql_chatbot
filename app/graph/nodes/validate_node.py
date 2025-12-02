# app/graph/nodes/validate_node.py
import sys
import logging
import os
from typing import Dict, Any, List, Optional, Set, Tuple
import difflib

from app.logger import get_logger
from app.exception import CustomException
from app import utils

logger = get_logger("validate_node")
LOG = logging.getLogger(__name__)


def _use_sqlglot() -> bool:
    try:
        import sqlglot  # type: ignore
        return True
    except Exception:
        return False


class ValidateNode:
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

    def _extract_tables_and_columns_sqlglot(self, sql: str) -> Tuple[Set[str], Set[str]]:
        try:
            import sqlglot  # type: ignore
            from sqlglot.expressions import Table, Column, Alias  # type: ignore

            parsed = sqlglot.parse_one(sql, read=None)
            tables: Set[str] = set()
            cols: Set[str] = set()

            for t in parsed.find_all(Table):
                try:
                    name = getattr(t, "name", None) or getattr(t, "this", None)
                    if name is None:
                        name = t.sql() if hasattr(t, "sql") else str(t)
                    tables.add(str(name).lower())
                except Exception:
                    continue

            for a in parsed.find_all(Alias):
                try:
                    alias_name = None
                    if hasattr(a, "alias"):
                        alias_name = a.alias
                    elif "alias" in a.args and a.args["alias"] is not None:
                        alias_name = a.args["alias"].name if hasattr(a.args["alias"], "name") else str(a.args["alias"])
                    if alias_name:
                        tables.add(str(alias_name).lower())
                except Exception:
                    pass

            for c in parsed.find_all(Column):
                try:
                    col_name = getattr(c, "name", None)
                    table_qual = getattr(c, "table", None)
                    if col_name:
                        cols.add(col_name.lower())
                    else:
                        token_text = c.sql() if hasattr(c, "sql") else str(c)
                        if "." in token_text:
                            parts = token_text.split(".")
                            cols.add(parts[-1].strip().lower())
                        else:
                            cols.add(token_text.strip().lower())
                    if table_qual:
                        try:
                            tables.add(table_qual.lower() if isinstance(table_qual, str) else str(table_qual).lower())
                        except Exception:
                            pass
                except Exception:
                    continue

            return tables, cols
        except Exception as e:
            LOG.debug("sqlglot parsing failed: %s", e, exc_info=True)
            return set(), set()

    def _build_name_index(self, schemas: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        name_index: Dict[str, str] = {}
        for store_key, meta in schemas.items():
            try:
                store_key_l = store_key.lower()
                name_index[store_key_l] = store_key
                canonical = None
                aliases = []
                if isinstance(meta, dict):
                    canonical = meta.get("canonical") or meta.get("canonical_name") or meta.get("table_name")
                    aliases = meta.get("aliases") or meta.get("alias") or []
                if canonical:
                    name_index[str(canonical).lower()] = store_key
                if isinstance(aliases, (list, tuple)):
                    for a in aliases:
                        if a:
                            name_index[str(a).lower()] = store_key
                path = meta.get("path") if isinstance(meta, dict) else None
                if path:
                    base = os.path.splitext(os.path.basename(path))[0].lower()
                    if base and base not in name_index:
                        name_index[base] = store_key
            except Exception:
                continue
        return name_index

    def _fuzzy_match_name(self, name: str, candidates: List[str], cutoff: float = 0.75) -> Optional[str]:
        if not name or not candidates:
            return None
        matches = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
        return matches[0] if matches else None

    def _canonical_schema_lookup(self, ref_names: Set[str], schemas: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if not schemas:
            return mapping

        name_index = self._build_name_index(schemas)
        candidates = list(name_index.keys())

        for r in ref_names:
            r_low = r.lower()
            if r_low in name_index:
                mapping[r_low] = name_index[r_low]
                continue
            found = None
            for cand in candidates:
                if r_low == cand or r_low in cand or cand in r_low:
                    found = cand
                    break
            if found:
                mapping[r_low] = name_index[found]
                continue
            fuzzy = self._fuzzy_match_name(r_low, candidates, cutoff=0.7)
            if fuzzy:
                mapping[r_low] = name_index[fuzzy]
                LOG.debug("ValidateNode: fuzzy-matched table '%s' -> '%s' (store_key=%s)", r_low, fuzzy, name_index[fuzzy])
                continue
        return mapping

    def run(self, sql: str, schemas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        try:
            errors: List[str] = []
            fixes: List[str] = []
            suggested_sql: Optional[str] = None

            orig_sql = (sql or "").strip()
            sql = orig_sql

            if not sql:
                errors.append("SQL query is empty.")
                return {"sql": sql, "valid": False, "errors": errors, "fixes": fixes, "suggested_sql": suggested_sql}

            tables_referenced: Set[str] = set()
            columns_referenced: Set[str] = set()

            if self._have_sqlglot:
                try:
                    tset, cset = self._extract_tables_and_columns_sqlglot(sql)
                    tables_referenced = set([t.lower() for t in tset if t])
                    columns_referenced = set([c.lower() for c in cset if c])
                    LOG.debug("ValidateNode(sqlglot): tables=%s cols=%s", tables_referenced, columns_referenced)

                    try:
                        import sqlglot as _sqlglot  # type: ignore
                        parsed = _sqlglot.parse_one(sql)
                        if not utils.is_select_query(sql):
                            errors.append("Only SELECT statements are allowed.")
                    except Exception:
                        if not utils.is_select_query(sql):
                            errors.append("Only SELECT statements are allowed.")
                except Exception:
                    LOG.debug("sqlglot extraction failed — falling back to utils-based validation", exc_info=True)
                    tables_referenced = set()
                    columns_referenced = set()
            else:
                try:
                    t_missing = utils.extract_table_names_from_sql(sql) if hasattr(utils, "extract_table_names_from_sql") else []
                    tables_referenced = {t.lower() for t in (t_missing or [])}
                except Exception:
                    tables_referenced = set()
                try:
                    c_missing = utils.extract_column_names_from_sql(sql) if hasattr(utils, "extract_column_names_from_sql") else []
                    columns_referenced = {c.lower() for c in (c_missing or [])}
                except Exception:
                    columns_referenced = set()
                if not utils.is_select_query(sql):
                    errors.append("Only SELECT statements are allowed.")

            missing_tables: List[str] = []
            found_table_map: Dict[str, str] = {}
            if tables_referenced:
                mapping = self._canonical_schema_lookup(tables_referenced, schemas)
                for ref in tables_referenced:
                    if ref in mapping:
                        found_table_map[ref] = mapping[ref]
                    else:
                        matched = None
                        for store_key in schemas.keys():
                            if ref == store_key.lower() or ref in store_key.lower() or store_key.lower() in ref:
                                matched = store_key
                                break
                        if matched:
                            found_table_map[ref] = matched
                        else:
                            candidates = [k.lower() for k in schemas.keys()]
                            fuzzy = self._fuzzy_match_name(ref, candidates, cutoff=0.7)
                            if fuzzy:
                                # fuzzy is lowercase cand; return original key
                                for k in schemas.keys():
                                    if k.lower() == fuzzy:
                                        found_table_map[ref] = k
                                        break
                            else:
                                missing_tables.append(ref)
            else:
                try:
                    miss = utils.validate_tables_in_sql(sql, schemas) or []
                    if miss:
                        missing_tables.extend([m.lower() for m in miss])
                except Exception:
                    pass

            if missing_tables:
                errors.append(f"Tables not found in schema: {', '.join(sorted(set(missing_tables)))}")
                fixes.append("Ensure the table name in the query matches the uploaded CSV canonical name. Open the schema viewer in the sidebar to check canonical/alias names.")

            missing_columns: List[str] = []
            fuzzy_column_mappings: Dict[str, Tuple[str, str]] = {}
            if columns_referenced:
                valid_cols: Set[str] = set()
                cols_by_store: Dict[str, Set[str]] = {}
                if found_table_map:
                    for store_key in set(found_table_map.values()):
                        rec = schemas.get(store_key, {}) or {}
                        cols = rec.get("columns") or []
                        # allow columns_normalized where present
                        norm = rec.get("columns_normalized") or []
                        lowered = {str(c).lower() for c in cols if c is not None}
                        # include normalized names too
                        lowered.update({str(c).lower() for c in norm if c})
                        cols_by_store[store_key] = lowered
                        valid_cols.update(lowered)
                else:
                    for store_key in schemas.keys():
                        rec = schemas.get(store_key, {}) or {}
                        cols = rec.get("columns") or []
                        norm = rec.get("columns_normalized") or []
                        lowered = {str(c).lower() for c in cols if c is not None}
                        lowered.update({str(c).lower() for c in norm if c})
                        cols_by_store[store_key] = lowered
                        valid_cols.update(lowered)

                for col in columns_referenced:
                    if not col or col == "*" or "(" in col or col.endswith(")"):
                        continue
                    if col in valid_cols:
                        continue
                    best = self._fuzzy_match_name(col, list(valid_cols), cutoff=0.8)
                    if best:
                        store_for_best = None
                        for sk, scols in cols_by_store.items():
                            if best in scols:
                                store_for_best = sk
                                break
                        if store_for_best:
                            fuzzy_column_mappings[col] = (best, store_for_best)
                            fixes.append(f"Interpreting column '{col}' as '{best}' (table={store_for_best}).")
                            continue
                    missing_columns.append(col)
            else:
                try:
                    miss = utils.validate_columns_in_sql(sql, schemas) or []
                    if miss:
                        missing_columns.extend([m.lower() for m in miss])
                except Exception:
                    pass

            if missing_columns:
                errors.append(f"Columns not found in schema: {', '.join(sorted(set(missing_columns)))}")
                fixes.append("Check column spellings and that the column exists in the selected CSV. Consider qualifying the column with the table name.")

            forbidden_used: List[str] = []
            if self.forbidden_tables:
                used_table_names = {v.lower() for v in found_table_map.values()} | {t.lower() for t in tables_referenced}
                for forb in self.forbidden_tables:
                    if forb.lower() in used_table_names:
                        forbidden_used.append(forb)
                if forbidden_used:
                    errors.append(f"Query uses forbidden tables: {', '.join(forbidden_used)}")
                    fixes.append("Remove references to forbidden tables from the query.")

            if self.max_row_limit and utils.is_select_query(sql):
                try:
                    if utils.exceeds_row_limit(sql, self.max_row_limit):
                        suggested_sql = utils.limit_sql_rows(sql, self.max_row_limit)
                        fixes.append(f"Applied automatic LIMIT {self.max_row_limit} to avoid large scans.")
                        sql = suggested_sql or sql
                except Exception:
                    LOG.debug("Could not evaluate/enforce row limit via utils.exceeds_row_limit", exc_info=True)

            valid = len(errors) == 0

            if valid:
                LOG.info("ValidateNode: SQL validation passed")
            else:
                LOG.warning("ValidateNode: validation failed: %s", errors)
                try:
                    if found_table_map:
                        for ref, sk in found_table_map.items():
                            if ref != sk.lower():
                                fixes.append(f"Interpreting table '{ref}' as '{sk}'.")
                except Exception:
                    pass

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
            logger.exception("ValidateNode.run failed", exc_info=True)
            raise CustomException(e, sys)
