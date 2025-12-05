import os
import io
import shutil
import pytest
from app.csv_loader import CSVLoader, save_uploaded_csv, load_csv_metadata

UPLOAD_DIR = "./tests/uploads"

@pytest.fixture(scope="module")
def csv_loader():
    # Ensure a clean test upload directory
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    loader = CSVLoader(upload_dir=UPLOAD_DIR, chunk_size=10, chunk_overlap=2)
    return loader

def test_save_uploaded_csv_bytes(csv_loader):
    content = b"name,description\nAlice,Hello world\nBob,Goodbye world"
    filename = "test_bytes.csv"
    path = save_uploaded_csv(content, filename, upload_dir=UPLOAD_DIR)
    assert os.path.exists(path)
    assert path.lower().endswith(".csv")

def test_save_uploaded_csv_fileobj(csv_loader):
    content = io.StringIO("name,description\nAlice,Hello world\nBob,Goodbye world")
    filename = "test_fileobj.csv"
    path = save_uploaded_csv(content, filename, upload_dir=UPLOAD_DIR)
    assert os.path.exists(path)
    assert path.lower().endswith(".csv")

def test_list_uploaded_csvs(csv_loader):
    # Ensure at least one CSV exists
    files = csv_loader.list_uploaded_csvs()
    assert isinstance(files, list)
    assert all(f.endswith(".csv") for f in files)

def test_load_csv_metadata_basic(csv_loader):
    test_csv_path = os.path.join(UPLOAD_DIR, "meta_test.csv")
    with open(test_csv_path, "w", encoding="utf-8") as f:
        f.write("col1,col2\nval1,val2\nval3,val4")

    meta = load_csv_metadata(test_csv_path)
    assert meta["table_name"]
    # loader normalizes column names; test the normalized names.
    assert meta["columns_normalized"] == ["col_1", "col_2"]
    assert meta["row_count"] == 2
    assert len(meta["sample_rows"]) <= 2


def test_load_and_chunk_csv(csv_loader):
    test_csv_path = os.path.join(UPLOAD_DIR, "chunk_test.csv")
    with open(test_csv_path, "w", encoding="utf-8") as f:
        f.write("text\n" + "A" * 25 + "\n" + "B" * 15 + "\n")

    chunks = csv_loader.load_and_chunk_csv(test_csv_path)
    assert isinstance(chunks, list)
    # Should produce multiple chunks because of chunk_size=10
    assert len(chunks) > 2
    for chunk in chunks:
        assert "text" in chunk
        assert "meta" in chunk
        assert "row" in chunk["meta"]
        assert "column" in chunk["meta"]
        assert "original" in chunk["meta"]
