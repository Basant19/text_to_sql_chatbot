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
    """
    ValidateNode:
      - Validates incoming SQL (basic safety rules)
      - Ensures referenced tables exist in provided schemas (with fuzzy / canonical lookup)
      - Ensures referenced columns exist (uses columns_normalized where available)
      - Produces helpful 'fixes' and 'suggested_sql' (e.g., inject LIMIT)
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

    # -------------------------
    # SQL extraction (sqlglot-backed)
    # -------------------------
    def _extract_tables_and_columns_sqlglot(self, sql: str) -> Tuple[Set[str], Set[str]]:
        """
        Use sqlglot to robustly extract table and column tokens.
        Returned tokens are canonicalized using utils._canonicalize_name where possible.
        """
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
                    token = utils._canonicalize_name(str(name))
                    if token:
                        tables.add(token)
                except Exception:
                    continue

            # collect alias names (may act as table refs)
            for a in parsed.find_all(Alias):
                try:
                    alias_name = None
                    if hasattr(a, "alias") and a.alias is not None:
                        # sqlglot alias object handling
                        alias_name = getattr(a.alias, "name", None) or str(a.alias)
                    elif "alias" in getattr(a, "args", {}):
                        alias_arg = a.args.get("alias")
                        alias_name = getattr(alias_arg, "name", None) if hasattr(alias_arg, "name") else str(alias_arg)
                    if alias_name:
                        tok = utils._canonicalize_name(str(alias_name))
                        if tok:
                            tables.add(tok)
                except Exception:
                    pass

            for c in parsed.find_all(Column):
                try:
                    # prefer explicit column name property
                    col_name = getattr(c, "name", None) or getattr(c, "this", None)
                    table_qual = getattr(c, "table", None)
                    if col_name:
                        token = utils._canonicalize_name(str(col_name))
                        if token:
                            cols.add(token)
                    else:
                        token_text = c.sql() if hasattr(c, "sql") else str(c)
                        if "." in token_text:
                            token = utils._canonicalize_name(token_text.split(".")[-1])
                        else:
                            token = utils._canonicalize_name(token_text)
                        if token:
                            cols.add(token)
                    # qualifier
                    if table_qual:
                        try:
                            qname = table_qual if isinstance(table_qual, str) else getattr(table_qual, "name", None) or str(table_qual)
                            qtok = utils._canonicalize_name(str(qname))
                            if qtok:
                                tables.add(qtok)
                        except Exception:
                            pass
                except Exception:
                    continue

            return tables, cols
        except Exception as e:
            LOG.debug("sqlglot parsing failed: %s", e, exc_info=True)
            return set(), set()

    # -------------------------
    # Schema name indexing & fuzzy helpers
    # -------------------------
    def _build_name_index(self, schemas: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Build an index mapping many possible normalized tokens -> store_key.

        Tokens included:
          - store_key (lower)
          - canonical (normalized)
          - friendly/table_name (normalized)
          - aliases (normalized)
          - basename of path (normalized)
        """
        name_index: Dict[str, str] = {}
        for store_key, meta in schemas.items():
            try:
                # normalized store_key
                sk_norm = utils._canonicalize_name(store_key)
                if sk_norm:
                    name_index[sk_norm] = store_key

                # meta fields
                if isinstance(meta, dict):
                    canonical = meta.get("canonical") or meta.get("canonical_name") or meta.get("table_name")
                    friendly = meta.get("friendly") or meta.get("table_name")
                    aliases = meta.get("aliases") or meta.get("alias") or []
                    path = meta.get("path")
                else:
                    canonical = None
                    friendly = None
                    aliases = []
                    path = None

                for candidate in (canonical, friendly):
                    if candidate:
                        tok = utils._canonicalize_name(str(candidate))
                        if tok:
                            name_index[tok] = store_key

                if isinstance(aliases, (list, tuple)):
                    for a in aliases:
                        if a:
                            tok = utils._canonicalize_name(str(a))
                            if tok:
                                name_index[tok] = store_key

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
        Map referenced tokens (already canonicalized via utils._canonicalize_name) to
        SchemaStore store_keys. Returns mapping: ref_token -> store_key
        """
        mapping: Dict[str, str] = {}
        if not schemas:
            return mapping

        name_index = self._build_name_index(schemas)
        candidates = list(name_index.keys())

        for r in ref_names:
            r_tok = utils._canonicalize_name(r)
            if not r_tok:
                continue
            # direct exact
            if r_tok in name_index:
                mapping[r_tok] = name_index[r_tok]
                continue
            # substring / containment
            found = None
            for cand in candidates:
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
                LOG.debug("ValidateNode: fuzzy-matched table '%s' -> '%s' (store_key=%s)", r_tok, fuzzy, name_index[fuzzy])
                continue
        return mapping

    # -------------------------
    # Main validation run
    # -------------------------
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

            # Extract referenced tables & columns (canonicalized tokens)
            tables_referenced: Set[str] = set()
            columns_referenced: Set[str] = set()

            if self._have_sqlglot:
                try:
                    tset, cset = self._extract_tables_and_columns_sqlglot(sql)
                    tables_referenced = {utils._canonicalize_name(t) for t in tset if t}
                    columns_referenced = {utils._canonicalize_name(c) for c in cset if c}
                    LOG.debug("ValidateNode(sqlglot): tables=%s cols=%s", tables_referenced, columns_referenced)
                    # enforce top-level SELECT
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
                    tlist = utils.extract_table_names_from_sql(sql) if hasattr(utils, "extract_table_names_from_sql") else []
                    tables_referenced = {utils._canonicalize_name(t) for t in (tlist or [])}
                except Exception:
                    tables_referenced = set()
                try:
                    clist = utils.extract_column_names_from_sql(sql) if hasattr(utils, "extract_column_names_from_sql") else []
                    columns_referenced = {utils._canonicalize_name(c) for c in (clist or [])}
                except Exception:
                    columns_referenced = set()
                if not utils.is_select_query(sql):
                    errors.append("Only SELECT statements are allowed.")

            # Validate tables
            missing_tables: List[str] = []
            found_table_map: Dict[str, str] = {}
            if tables_referenced:
                mapping = self._canonical_schema_lookup(tables_referenced, schemas)
                for ref in tables_referenced:
                    ref_tok = utils._canonicalize_name(ref)
                    if not ref_tok:
                        continue
                    if ref_tok in mapping:
                        found_table_map[ref_tok] = mapping[ref_tok]
                    else:
                        # attempt additional heuristics: match against store_key normalized or canonical tokens
                        matched = None
                        for store_key, meta in schemas.items():
                            try:
                                sk_norm = utils._canonicalize_name(store_key)
                                canonical = (meta.get("canonical") if isinstance(meta, dict) else None) or ""
                                can_norm = utils._canonicalize_name(canonical)
                                path = meta.get("path") if isinstance(meta, dict) else None
                                base_norm = utils._canonicalize_name(os.path.splitext(os.path.basename(path))[0]) if path else ""
                                # direct containment checks across normalized tokens
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
                            # fuzzy across store keys
                            candidates = [utils._canonicalize_name(k) for k in schemas.keys() if k]
                            fuzzy = self._fuzzy_match_name(ref_tok, candidates, cutoff=0.7)
                            if fuzzy:
                                for k in schemas.keys():
                                    if utils._canonicalize_name(k) == fuzzy:
                                        found_table_map[ref_tok] = k
                                        break
                            else:
                                missing_tables.append(ref_tok)
            else:
                try:
                    miss = utils.validate_tables_in_sql(sql, schemas) or []
                    if miss:
                        missing_tables.extend([utils._canonicalize_name(m) for m in miss])
                except Exception:
                    pass

            if missing_tables:
                errors.append(f"Tables not found in schema: {', '.join(sorted(set(missing_tables)))}")
                fixes.append(
                    "Ensure the table name in the query matches the uploaded CSV canonical name/alias (check Schema viewer). "
                    "If your CSV had a generated key (e.g. 'googleplaystore_abc123'), use its canonical/alias token instead (e.g. 'googleplaystore')."
                )

            # Validate columns
            missing_columns: List[str] = []
            fuzzy_column_mappings: Dict[str, Tuple[str, str]] = {}
            if columns_referenced:
                valid_cols: Set[str] = set()
                cols_by_store: Dict[str, Set[str]] = {}
                # If specific tables were found, narrow columns search to them; otherwise use all
                target_stores = set(found_table_map.values()) if found_table_map else set(schemas.keys())
                for store_key in target_stores:
                    rec = schemas.get(store_key, {}) or {}
                    raw_cols = rec.get("columns") or []
                    norm_cols = rec.get("columns_normalized") or []
                    # unify normalized representation
                    lowered = {utils._canonicalize_name(str(c)) for c in raw_cols if c is not None}
                    lowered.update({utils._canonicalize_name(str(c)) for c in norm_cols if c})
                    # remove empty tokens
                    lowered = {c for c in lowered if c}
                    cols_by_store[store_key] = lowered
                    valid_cols.update(lowered)

                for col in columns_referenced:
                    if not col or col == "*" or "(" in col or col.endswith(")"):
                        continue
                    col_tok = utils._canonicalize_name(col)
                    if not col_tok:
                        continue
                    if col_tok in valid_cols:
                        continue
                    # fuzzy match against union of columns
                    best = self._fuzzy_match_name(col_tok, list(valid_cols), cutoff=0.8)
                    if best:
                        store_for_best = None
                        for sk, scols in cols_by_store.items():
                            if best in scols:
                                store_for_best = sk
                                break
                        if store_for_best:
                            fuzzy_column_mappings[col_tok] = (best, store_for_best)
                            fixes.append(f"Interpreting column '{col}' as '{best}' (table={store_for_best}).")
                            continue
                    missing_columns.append(col_tok)
            else:
                try:
                    miss = utils.validate_columns_in_sql(sql, schemas) or []
                    if miss:
                        missing_columns.extend([utils._canonicalize_name(m) for m in miss])
                except Exception:
                    pass

            if missing_columns:
                errors.append(f"Columns not found in schema: {', '.join(sorted(set(missing_columns)))}")
                fixes.append("Check column spellings and that the column exists in the selected CSV. Consider qualifying the column with the table name.")

            # Forbidden table usage
            forbidden_used: List[str] = []
            if self.forbidden_tables:
                used_table_names = {v.lower() for v in found_table_map.values()} | {t.lower() for t in tables_referenced}
                for forb in self.forbidden_tables:
                    if forb.lower() in used_table_names:
                        forbidden_used.append(forb)
                if forbidden_used:
                    errors.append(f"Query uses forbidden tables: {', '.join(forbidden_used)}")
                    fixes.append("Remove references to forbidden tables from the query.")

            # Max row limit enforcement
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
                # append table interpretation hints
                try:
                    if found_table_map:
                        for ref, sk in found_table_map.items():
                            # add only if interpretation is non-trivial
                            if ref and sk and ref != utils._canonicalize_name(sk):
                                fixes.append(f"Interpreting table '{ref}' as store key '{sk}'. Use canonical/alias '{schemas.get(sk,{}).get('canonical') or sk}' in your query.")
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
