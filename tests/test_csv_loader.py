import os
import io
import sys
import tempfile
import traceback

# Ensure project root (one level up) is on sys.path so "from app import ..." works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest  # noqa: E402
from app import config  # noqa: E402
from app.csv_loader import save_uploaded_csv, load_csv_metadata  # noqa: E402
from app.logger import get_logger  # noqa: E402
from app.exception import CustomException  # noqa: E402

logger = get_logger("test_csv_loader")

SAMPLE_CSV = """id,name,age
1,Alice,30
2,Bob,25
3,Charlie,40
4,Dana,28
"""


# ---------- Helper functions (used by both pytest and script-run) ----------

def _run_save_and_load(upload_dir: str):
    """
    Helper that saves a sample CSV into upload_dir (by patching config.UPLOAD_DIR),
    then loads metadata and returns it.
    """
    # Patch config path (works for both pytest monkeypatch and standalone)
    original_upload_dir = getattr(config, "UPLOAD_DIR", None)
    config.UPLOAD_DIR = upload_dir

    try:
        file_obj = io.StringIO(SAMPLE_CSV)
        saved_path = save_uploaded_csv(file_obj, "people.csv")
        metadata = load_csv_metadata(saved_path, sample_rows=2)
        return saved_path, metadata
    finally:
        # restore original UPLOAD_DIR
        if original_upload_dir is None:
            delattr(config, "UPLOAD_DIR")
        else:
            config.UPLOAD_DIR = original_upload_dir


# -------------------- pytest tests --------------------

def test_save_uploaded_csv_and_load_metadata(tmp_path, monkeypatch):
    # Use pytest's tmp_path and monkeypatch to keep things isolated
    monkeypatch.setattr(config, "UPLOAD_DIR", str(tmp_path))

    file_obj = io.StringIO(SAMPLE_CSV)
    saved_path = save_uploaded_csv(file_obj, "people.csv")

    assert os.path.isfile(saved_path), "Saved CSV file should exist"

    metadata = load_csv_metadata(saved_path, sample_rows=2)
    assert metadata["table_name"].startswith("people")
    assert metadata["path"] == saved_path
    assert metadata["columns"] == ["id", "name", "age"]
    assert isinstance(metadata["sample_rows"], list)
    assert len(metadata["sample_rows"]) == 2
    assert metadata["row_count"] == 4  # data rows count


def test_duplicate_filename_creates_unique_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", str(tmp_path))

    # Save first file
    first = io.StringIO(SAMPLE_CSV)
    path1 = save_uploaded_csv(first, "dup.csv")
    assert os.path.exists(path1)

    # Save second file with same name; should not overwrite
    second = io.StringIO(SAMPLE_CSV)
    path2 = save_uploaded_csv(second, "dup.csv")
    assert os.path.exists(path2)

    assert path1 != path2, "Saving a duplicate filename should produce a unique path"

    # Load metadata from both to ensure both are valid CSVs
    meta1 = load_csv_metadata(path1)
    meta2 = load_csv_metadata(path2)
    assert meta1["columns"] == meta2["columns"]


def test_load_nonexistent_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", str(tmp_path))
    nonexist = os.path.join(str(tmp_path), "nope.csv")
    # Expect CustomException (the loader wraps errors in CustomException)
    with pytest.raises(CustomException):
        load_csv_metadata(nonexist)


# -------------------- Standalone script runner --------------------
def _run_as_script():
    """
    Run a small sequence of checks without pytest:
    - Use a TemporaryDirectory
    - Call helper to save and load CSV
    - Verify duplicate filename behavior
    - Verify nonexistent file raises CustomException
    Print friendly output messages for success/failure.
    """
    print("Running tests in standalone mode (no pytest).")
    successes = 0
    failures = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            saved_path, metadata = _run_save_and_load(tmpdir)
            assert os.path.isfile(saved_path)
            assert metadata["columns"] == ["id", "name", "age"]
            print("✔ save & load metadata: PASSED")
            successes += 1
        except Exception:
            print("✖ save & load metadata: FAILED")
            traceback.print_exc()
            failures += 1

        try:
            # duplicate filename
            path1 = save_uploaded_csv(io.StringIO(SAMPLE_CSV), "dup.csv")
            path2 = save_uploaded_csv(io.StringIO(SAMPLE_CSV), "dup.csv")
            assert os.path.exists(path1) and os.path.exists(path2) and (path1 != path2)
            print("✔ duplicate filename uniqueness: PASSED")
            successes += 1
        except Exception:
            print("✖ duplicate filename uniqueness: FAILED")
            traceback.print_exc()
            failures += 1

        try:
            nonexist = os.path.join(tmpdir, "nope.csv")
            try:
                load_csv_metadata(nonexist)
                # if no exception, it's a failure
                print("✖ load nonexistent file: FAILED (no exception raised)")
                failures += 1
            except CustomException:
                print("✔ load nonexistent file raises CustomException: PASSED")
                successes += 1
            except Exception:
                print("✖ load nonexistent file: FAILED (wrong exception type)")
                traceback.print_exc()
                failures += 1
        except Exception:
            print("✖ load nonexistent file test infrastructure failed")
            traceback.print_exc()
            failures += 1

    print(f"\nStandalone run complete. successes={successes}, failures={failures}")
    return failures == 0


# Allow both pytest discovery and direct execution:
if __name__ == "__main__":
    ok = _run_as_script()
    if not ok:
        sys.exit(1)
