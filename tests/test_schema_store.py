# tests/test_schema_store.py
import os
import shutil
import pytest
from app.schema_store import SchemaStore
from app.csv_loader import save_uploaded_csv

TEST_DIR = "./tests/schema_store_test"
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR, exist_ok=True)

@pytest.fixture(scope="module")
def store():
    # Clean test dir
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)
    store_path = os.path.join(TEST_DIR, "schema_store.json")
    s = SchemaStore(store_path)
    return s

def test_register_from_csv_and_lookup(store):
    csv_content = "App,Rating\nTikTok,4.2\nSignal,4.5\n"
    csv_path = os.path.join(TEST_DIR, "apps.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    canonical = store.register_from_csv(csv_path)
    assert canonical is not None
    # canonical should map from alias 'apps' or filename
    assert store.has_table(canonical)
    assert store.get_table_canonical("apps") == canonical or store.get_table_canonical("apps") is not None
    cols = store.get_columns(canonical)
    assert "app" in cols or "rating" in cols  # normalized names

def test_has_column_and_validate(store):
    # columns: App, Rating
    ok = store.has_column("apps", "App")
    assert ok
    ok2 = store.has_column("apps", "rating")
    assert ok2

    valid, missing_tables, missing_columns = store.validate_table_and_columns("apps", ["App", "unknown_col"])
    assert not valid
    assert missing_tables == []
    assert "unknown_col" in missing_columns

def test_persistence(store):
    store_path = os.path.join(TEST_DIR, "schema_store.json")
    # create a new instance pointing to same store_path
    s2 = SchemaStore(store_path)
    # tables should exist
    assert len(s2.list_tables()) >= 1
    # cleanup
    s2.clear()
    shutil.rmtree(TEST_DIR)
