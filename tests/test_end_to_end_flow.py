# tests/test_end_to_end_flow.py
import os
import csv
import sqlite3
from typing import Any, Dict, List, Optional

import pytest

# project imports
from app.csv_loader import CSVLoader, load_csv_metadata
from app.schema_store import SchemaStore
from app.vector_search import get_vector_search, VectorSearch
from app.tools import Tools
from app.graph.nodes.validate_node import ValidateNode
from app.graph.nodes.format_node import FormatNode
from app.graph.nodes.context_node import ContextNode
from app.graph.nodes.error_node import ErrorNode

import app.graph.nodes.validate_node as validate_node_module


# -------------------------
# Small SQLiteExecutor used by Tools in tests
# -------------------------
class SQLiteExecutor:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or ":memory:"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def load_csv_table(self, csv_path: str, table_name: str, force_reload: bool = False) -> None:
        if force_reload:
            with self.conn:
                self.conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []

            cols = [c.strip().replace('"', '""') or f"col{i}" for i, c in enumerate(header, start=1)]
            col_defs = ", ".join(f'"{c}" TEXT' for c in cols)

            with self.conn:
                self.conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})')

                # --- Correct insert_sql generation ---
                col_names = ", ".join(f'"{c}"' for c in cols)
                placeholders = ", ".join("?" for _ in cols)
                insert_sql = f'INSERT INTO "{table_name}" ({col_names}) VALUES ({placeholders})'

                batch = []
                for row in reader:
                    row_norm = list(row) + [None] * (len(cols) - len(row))
                    row_norm = row_norm[: len(cols)]
                    batch.append(tuple(str(x) if x is not None else None for x in row_norm))
                if batch:
                    self.conn.executemany(insert_sql, batch)

    def load_table_from_memory(self, name: str, rows: List[Dict[str, Any]]):
        if not rows:
            return
        cols = list(rows[0].keys())
        col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
        with self.conn:
            self.conn.execute(f'DROP TABLE IF EXISTS "{name}"')
            self.conn.execute(f'CREATE TABLE "{name}" ({col_defs})')

            # --- Correct insert_sql generation ---
            col_names = ", ".join(f'"{c}"' for c in cols)
            placeholders = ", ".join("?" for _ in cols)
            insert_sql = f'INSERT INTO "{name}" ({col_names}) VALUES ({placeholders})'

            batch = []
            for r in rows:
                batch.append(tuple(str(r.get(c)) for c in cols))
            if batch:
                self.conn.executemany(insert_sql, batch)

    def list_tables(self) -> List[str]:
        cur = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [r[0] for r in cur.fetchall()]

    def execute_sql(self, sql: str, read_only: bool = True, limit: Optional[int] = None, as_dataframe: bool = False) -> Dict[str, Any]:
        q = sql.strip()
        if limit and "limit" not in q.lower():
            q = q.rstrip(";") + f" LIMIT {limit}"
        cur = self.conn.cursor()
        try:
            cur.execute(q)
            rows = [dict(r) for r in cur.fetchall()]
            cols = [c[0] for c in cur.description] if cur.description else []
            return {"rows": rows, "columns": cols, "rowcount": len(rows)}
        finally:
            cur.close()
