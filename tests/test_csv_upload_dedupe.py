# tests/test_csv_upload_dedupe.py
import os
import shutil
from app.csv_loader import CSVLoader
from io import BytesIO
import pytest

def test_csvloader_content_addressing_and_dedupe(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    loader = CSVLoader(upload_dir=str(upload_dir))

    content = b"col1,col2\n1,2\n3,4\n"
    bio1 = BytesIO(content)
    bio1.name = "mydata.csv"

    path1 = loader.save_csv(bio1)
    assert os.path.exists(path1)
    files_after_first = os.listdir(str(upload_dir))
    assert len(files_after_first) == 1

    # Save identical content again (new BytesIO). Should return same path (content-addressed)
    bio2 = BytesIO(content)
    bio2.name = "mydata.csv"
    path2 = loader.save_csv(bio2)
    assert os.path.exists(path2)
    # path must be equal (or at least file count must still be 1)
    assert path1 == path2 or len(os.listdir(str(upload_dir))) == 1

    # Save different content -> new file should appear
    content2 = b"colA,colB\nx,y\n"
    bio3 = BytesIO(content2)
    bio3.name = "mydata.csv"
    path3 = loader.save_csv(bio3)
    assert os.path.exists(path3)
    assert path3 != path1
    assert len(os.listdir(str(upload_dir))) == 2
