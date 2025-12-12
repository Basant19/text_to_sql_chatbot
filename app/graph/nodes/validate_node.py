#D:\text_to_sql_bot\app\graph\nodes\validate_node.py
"""
ValidateNode:
  - Validates incoming SQL (basic safety rules)
  - Ensures referenced tables exist in provided schemas (with fuzzy / canonical lookup)
  - Ensures referenced columns exist (uses columns_normalized where available)
  - Produces helpful 'fixes' and 'suggested_sql' (e.g., inject LIMIT)
"""
from __future__ import annotations
import sys
import logging
import os
import difflib
from typing import Dict, Any, List, Optional, Set, Tuple

from app.logger import get_logger
from app.exception import CustomException
import app.utils as utils  # local utilities used for canonicalization, simple SQL helpers

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
            self.forbidden_tables = [t for t in (self.safety_rules.get("forbidden_tables") or []) if t]
            # normalize forbidden table tokens early for comparison
            try:
                self.forbidden_tables = [utils._canonicalize_name(t) for t in self.forbidden_tables]
            except Exception:
                self.forbidden_tables = [str(t).lower() for t in self.forbidden_tables]

            self._have_sqlglot = _use_sqlglot()
            if self._have_sqlglot:
                LOG.debug("ValidateNode: sqlglot available — using it for parsing.")
            else:
                LOG.debug("ValidateNode: sqlglot not available — falling back to utils functions.")
            logger.info("ValidateNode initialized with safety_rules: %s", self.safety_rules)
        except Exception as e:
            logger.exception("Failed to initialize ValidateNode")
            raise CustomException(e, sys)

    # ---------- SQL parsing helpers ----------
    def _extract_tables_and_columns_sqlglot(self, sql: str) -> Tuple[Set[str], Set[str]]:
        """
        Use sqlglot to extract table & column tokens. Return raw tokens (not normalized).
        """
        try:
            import sqlglot  # type: ignore
            from sqlglot.expressions import Table, Column, Alias  # type: ignore

            parsed = sqlglot.parse_one(sql)
            tables: Set[str] = set()
            cols: Set[str] = set()

            for t in parsed.find_all(Table):
                try:
                    name = getattr(t, "name", None) or getattr(t, "this", None) or str(t)
                    if name:
                        tables.add(str(name))
                except Exception:
                    continue

            # collect column expressions
            for c in parsed.find_all(Column):
                try:
                    # Column can be qualified (table.col) or only col
                    col_name = getattr(c, "name", None) or getattr(c, "this", None) or str(c)
                    if col_name:
                        cols.add(str(col_name))
                    # if column has table qualifier, include it as referenced table
                    table_qual = getattr(c, "table", None)
                    if table_qual:
                        q = table_qual if isinstance(table_qual, str) else getattr(table_qual, "name", None) or str(table_qual)
                        if q:
                            tables.add(str(q))
                except Exception:
                    continue

            # collect aliases that might hide table names
            for a in parsed.find_all(Alias):
                try:
                    alias_name = None
                    if hasattr(a, "alias") and a.alias is not None:
                        alias_name = getattr(a.alias, "name", None) or str(a.alias)
                    elif "alias" in getattr(a, "args", {}):
                        alias_arg = a.args.get("alias")
                        alias_name = getattr(alias_arg, "name", None) if hasattr(alias_arg, "name") else str(alias_arg)
                    if alias_name:
                        tables.add(str(alias_name))
                except Exception:
                    pass

            return tables, cols
        except Exception as e:
            LOG.debug("sqlglot parsing failed: %s", e, exc_info=True)
            return set(), set()

    # ---------- name indexing & fuzzy helpers ----------
    def _build_name_index(self, schemas: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Map many possible normalized tokens -> schema store key
        """
        name_index: Dict[str, str] = {}
        for store_key, meta in (schemas or {}).items():
            try:
                sk_norm = utils._canonicalize_name(store_key)
                if sk_norm:
                    name_index[sk_norm] = store_key

                if isinstance(meta, dict):
                    for candidate_field in ("canonical", "canonical_name", "table_name", "original_name"):
                        val = meta.get(candidate_field)
                        if val:
                            tok = utils._canonicalize_name(str(val))
                            if tok:
                                name_index[tok] = store_key

                    aliases = meta.get("aliases") or []
                    if isinstance(aliases, (list, tuple)):
                        for a in aliases:
                            if a:
                                a_tok = utils._canonicalize_name(str(a))
                                if a_tok:
                                    name_index[a_tok] = store_key

                    path = meta.get("path")
                    if path:
                        base = os.path.splitext(os.path.basename(path))[0]
                        base_tok = utils._canonicalize_name(base)
                        if base_tok:
                            name_index[base_tok] = store_key
            except Exception:
                continue
        return name_index

    def _fuzzy_match_name(self, name: str, candidates: List[str], cutoff: float = 0.75) -> Optional[str]:
        if not name or not candidates:
            return None
        matches = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
        return matches[0] if matches else None

    def _canonical_schema_lookup(self, ref_names: Set[str], schemas: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Map referenced tokens (raw or normalized) -> schema store_key
        """
        mapping: Dict[str, str] = {}
        if not ref_names or not schemas:
            return mapping

        name_index = self._build_name_index(schemas)
        candidates = list(name_index.keys())

        for r in ref_names:
            try:
                r_tok = utils._canonicalize_name(r)
            except Exception:
                r_tok = (r or "").strip().lower()
            if not r_tok:
                continue
            # exact
            if r_tok in name_index:
                mapping[r_tok] = name_index[r_tok]
                continue
            # containment (short-circuit)
            found = None
            for cand in candidates:
                if not cand:
                    continue
                if r_tok == cand or r_tok in cand or cand in r_tok:
                    found = cand
                    break
            if found:
                mapping[r_tok] = name_index[found]
                continue
            # fuzzy
            fuzzy = self._fuzzy_match_name(r_tok, candidates, cutoff=0.7)
            if fuzzy:
                mapping[r_tok] = name_index[fuzzy]
                LOG.debug("ValidateNode: fuzzy-matched table '%s' -> '%s'", r_tok, name_index[fuzzy])
                continue
        return mapping

    # ---------- helpers ----------
    def _strip_qualifier_from_column(self, col: str) -> str:
        """
        Remove any table/qualifier prefix from a column token like 'table.col' or 'schema.table.col'.
        Returns the rightmost token.
        """
        if not col:
            return col
        if "." in col:
            return col.split(".")[-1]
        return col

    # ---------- run ----------
    def run(self, sql: str, schemas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate SQL against provided schemas.

        Returns dict:
          {
            "sql": possibly_modified_sql,
            "valid": bool,
            "errors": [..],
            "fixes": [..],
            "suggested_sql": <optional>
          }
        """
        try:
            errors: List[str] = []
            fixes: List[str] = []
            suggested_sql: Optional[str] = None

            orig_sql = (sql or "").strip()
            sql = orig_sql

            if not sql:
                errors.append("SQL query is empty.")
                return {"sql": sql, "valid": False, "errors": errors, "fixes": fixes, "suggested_sql": suggested_sql}

            # extract refs
            tables_referenced: Set[str] = set()
            columns_referenced: Set[str] = set()

            if self._have_sqlglot:
                try:
                    tset, cset = self._extract_tables_and_columns_sqlglot(sql)
                    tables_referenced = {str(t) for t in tset if t}
                    columns_referenced = {str(c) for c in cset if c}
                except Exception:
                    tables_referenced = set()
                    columns_referenced = set()
            else:
                # fallback to utils helpers if available
                try:
                    tlist = utils.extract_table_names_from_sql(sql) if hasattr(utils, "extract_table_names_from_sql") else []
                    if isinstance(tlist, (list, tuple)):
                        tables_referenced = set(tlist)
                    elif isinstance(tlist, str):
                        tables_referenced = {tlist}
                except Exception:
                    tables_referenced = set()
                try:
                    clist = utils.extract_column_names_from_sql(sql) if hasattr(utils, "extract_column_names_from_sql") else []
                    if isinstance(clist, (list, tuple)):
                        columns_referenced = set(clist)
                    elif isinstance(clist, str):
                        columns_referenced = {clist}
                except Exception:
                    columns_referenced = set()

            # Basic SELECT enforcement
            if not utils.is_select_query(sql):
                errors.append("Only SELECT statements are allowed.")
                # we still continue to gather helpful messages

            # Validate tables
            missing_tables: List[str] = []
            found_table_map: Dict[str, str] = {}

            if tables_referenced:
                mapping = self._canonical_schema_lookup(tables_referenced, schemas)
                for ref in tables_referenced:
                    try:
                        ref_tok = utils._canonicalize_name(ref)
                    except Exception:
                        ref_tok = (ref or "").strip().lower()
                    if not ref_tok:
                        continue
                    if ref_tok in mapping:
                        found_table_map[ref_tok] = mapping[ref_tok]
                    else:
                        # try containment against store keys
                        matched = None
                        for store_key, meta in (schemas or {}).items():
                            try:
                                sk_norm = utils._canonicalize_name(store_key)
                                canonical = (meta.get("canonical") if isinstance(meta, dict) else None) or ""
                                can_norm = utils._canonicalize_name(canonical) if canonical else ""
                                path = meta.get("path") if isinstance(meta, dict) else None
                                base_norm = utils._canonicalize_name(os.path.splitext(os.path.basename(path))[0]) if path else ""
                                candidates = [sk_norm, can_norm, base_norm]
                                for cand in candidates:
                                    if not cand:
                                        continue
                                    if ref_tok == cand or ref_tok in cand or cand in ref_tok:
                                        matched = store_key
                                        break
                                if matched:
                                    break
                            except Exception:
                                continue
                        if matched:
                            found_table_map[ref_tok] = matched
                        else:
                            # fuzzy over store keys
                            candidates = [utils._canonicalize_name(k) for k in (schemas or {}).keys() if k]
                            fuzzy = self._fuzzy_match_name(ref_tok, candidates, cutoff=0.7)
                            if fuzzy:
                                for k in (schemas or {}).keys():
                                    if utils._canonicalize_name(k) == fuzzy:
                                        found_table_map[ref_tok] = k
                                        break
                            else:
                                missing_tables.append(ref_tok)
            else:
                # try utils.validate_tables_in_sql if available
                try:
                    miss = utils.validate_tables_in_sql(sql, schemas) if hasattr(utils, "validate_tables_in_sql") else []
                    if miss:
                        if isinstance(miss, (list, tuple)):
                            missing_tables.extend([utils._canonicalize_name(m) for m in miss])
                        else:
                            missing_tables.append(utils._canonicalize_name(miss))
                except Exception:
                    pass

            if missing_tables:
                errors.append(f"Tables not found in schema: {', '.join(sorted(set(missing_tables)))}")
                fixes.append(
                    "Ensure the table name in the query matches the uploaded CSV canonical name/alias (check Schema viewer)."
                )

            # Validate columns
            missing_columns: List[str] = []
            fuzzy_column_mappings: Dict[str, Tuple[str, str]] = {}
            if columns_referenced:
                valid_cols: Set[str] = set()
                cols_by_store: Dict[str, Set[str]] = {}
                # narrow to found stores if possible
                target_stores = set(found_table_map.values()) if found_table_map else set((schemas or {}).keys())
                for store_key in target_stores:
                    rec = (schemas or {}).get(store_key, {}) or {}
                    raw_cols = rec.get("columns") or []
                    norm_cols = rec.get("columns_normalized") or []
                    lowered = {utils._canonicalize_name(str(c)) for c in raw_cols if c is not None}
                    lowered.update({utils._canonicalize_name(str(c)) for c in norm_cols if c})
                    lowered = {c for c in lowered if c}
                    cols_by_store[store_key] = lowered
                    valid_cols.update(lowered)

                for col in columns_referenced:
                    # strip qualifiers like table.col
                    col_simple = self._strip_qualifier_from_column(col)
                    if not col_simple or col_simple == "*" or "(" in col_simple or col_simple.endswith(")"):
                        continue
                    try:
                        col_tok = utils._canonicalize_name(col_simple)
                    except Exception:
                        col_tok = (col_simple or "").strip().lower()
                    if not col_tok:
                        continue
                    if col_tok in valid_cols:
                        continue
                    # fuzzy match
                    best = self._fuzzy_match_name(col_tok, list(valid_cols), cutoff=0.8)
                    if best:
                        store_for_best = None
                        for sk, scols in cols_by_store.items():
                            if best in scols:
                                store_for_best = sk
                                break
                        if store_for_best:
                            fuzzy_column_mappings[col_tok] = (best, store_for_best)
                            hint = f"Interpreting column '{col}' as '{best}' (table={store_for_best})."
                            if hint not in fixes:
                                fixes.append(hint)
                            continue
                    missing_columns.append(col_tok)
            else:
                try:
                    miss = utils.validate_columns_in_sql(sql, schemas) if hasattr(utils, "validate_columns_in_sql") else []
                    if miss:
                        if isinstance(miss, (list, tuple)):
                            missing_columns.extend([utils._canonicalize_name(m) for m in miss])
                        else:
                            missing_columns.append(utils._canonicalize_name(miss))
                except Exception:
                    pass

            if missing_columns:
                errors.append(f"Columns not found in schema: {', '.join(sorted(set(missing_columns)))}")
                fixes.append("Check column spellings and that the column exists in the selected CSV. Qualify with table name if needed.")

            # forbidden tables (use canonicalized tokens)
            forbidden_used: List[str] = []
            if self.forbidden_tables:
                # Build a set of referenced store_keys and associated canonical/aliases for checking
                used_store_keys = set(found_table_map.values())
                # If nothing was mapped but tables were referenced, attempt to map for the purpose of forbidden check
                if not used_store_keys and tables_referenced:
                    temp_map = self._canonical_schema_lookup(tables_referenced, schemas)
                    used_store_keys.update(temp_map.values())

                # Now check each used store_key's canonical/aliases/store_key tokens for matches
                for sk in used_store_keys:
                    try:
                        meta = (schemas or {}).get(sk, {}) or {}
                        tokens_to_check = set()
                        tokens_to_check.add(utils._canonicalize_name(sk))
                        canonical = meta.get("canonical") or meta.get("canonical_name") or ""
                        if canonical:
                            tokens_to_check.add(utils._canonicalize_name(canonical))
                        aliases = meta.get("aliases") or []
                        if isinstance(aliases, (list, tuple)):
                            for a in aliases:
                                if a:
                                    tokens_to_check.add(utils._canonicalize_name(str(a)))
                        # check if any forbidden token matches these
                        for forb_tok in self.forbidden_tables:
                            if not forb_tok:
                                continue
                            if forb_tok in tokens_to_check:
                                forbidden_used.append(forb_tok)
                    except Exception:
                        continue

                # Also check raw referenced table tokens (in case someone wrote the forbidden token directly)
                for t in tables_referenced:
                    try:
                        ttok = utils._canonicalize_name(t)
                        for forb_tok in self.forbidden_tables:
                            if forb_tok == ttok:
                                forbidden_used.append(forb_tok)
                    except Exception:
                        continue

                if forbidden_used:
                    errors.append(f"Query uses forbidden tables: {', '.join(sorted(set(forbidden_used)))}")
                    fixes.append("Remove references to forbidden tables from the query.")

            # max row limit enforcement
            if self.max_row_limit and utils.is_select_query(sql):
                try:
                    if utils.exceeds_row_limit(sql, self.max_row_limit):
                        # produce suggested_sql and apply to returned sql for safety
                        suggested = utils.limit_sql_rows(sql, self.max_row_limit)
                        if suggested and suggested != sql:
                            suggested_sql = suggested
                            fixes.append(f"Applied automatic LIMIT {self.max_row_limit} to avoid large scans.")
                            sql = suggested_sql
                except Exception:
                    LOG.debug("Could not evaluate/enforce row limit via utils.exceeds_row_limit", exc_info=True)

            # Add interpretation hints for any non-trivial found_table_map matches
            try:
                if found_table_map:
                    for ref, sk in found_table_map.items():
                        try:
                            sk_canonical = (schemas.get(sk, {}) or {}).get("canonical") or sk
                            if ref and sk and ref != utils._canonicalize_name(sk):
                                # always add interpreting hint (helpful even when valid)
                                hint = f"Interpreting table '{ref}' as store key '{sk}'. Use canonical/alias '{sk_canonical}' in your query."
                                if hint not in fixes:
                                    fixes.append(hint)
                        except Exception:
                            continue
            except Exception:
                LOG.debug("Could not append interpretation hints", exc_info=True)

            # dedupe fixes preserving order
            seen = set()
            deduped_fixes = []
            for f in fixes:
                if f not in seen:
                    deduped_fixes.append(f)
                    seen.add(f)

            valid = len(errors) == 0

            if not valid:
                LOG.warning("ValidateNode: validation failed: %s", errors)
            else:
                LOG.info("ValidateNode: SQL validation passed")

            return {
                "sql": sql,
                "valid": valid,
                "errors": errors,
                "fixes": deduped_fixes,
                "suggested_sql": suggested_sql,
            }

        except CustomException:
            raise
        except Exception as e:
            logger.exception("ValidateNode.run failed", exc_info=True)
            raise CustomException(e, sys)
