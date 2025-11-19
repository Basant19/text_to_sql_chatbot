import os
import io
import csv
import time
import sys
import uuid
from typing import Dict, List, Any, Union, TextIO

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("csv_loader")


def _sanitize_filename(name: str) -> str:
    """Make a filename safe: replace spaces and remove troublesome chars."""
    keepchars = (" ", ".", "_", "-")
    safe = "".join(c if c.isalnum() or c in keepchars else "_" for c in name)
    safe = safe.replace(" ", "_")
    return safe


def _unique_path(dest_dir: str, filename: str) -> str:
    """
    Return a unique path under dest_dir for filename.
    If filename exists, append a UUID to guarantee uniqueness.
    """
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    if not os.path.exists(candidate):
        return candidate

    # Use UUID to ensure uniqueness even when called multiple times quickly
    unique_name = f"{base}_{uuid.uuid4().hex}{ext}"
    return os.path.join(dest_dir, unique_name)


def save_uploaded_csv(fileobj: Union[bytes, TextIO, io.StringIO, io.BytesIO], filename: str) -> str:
    """
    Save an uploaded CSV-like object to config.UPLOAD_DIR.
    Accepts bytes or file-like objects. Returns the saved file path.
    """
    try:
        upload_dir = getattr(config, "UPLOAD_DIR", None)
        if not upload_dir:
            # fallback to cwd/uploads
            upload_dir = os.path.join(os.getcwd(), "uploads")

        os.makedirs(upload_dir, exist_ok=True)
        safe_name = _sanitize_filename(filename)
        dest_path = _unique_path(upload_dir, safe_name)

        # If bytes provided
        if isinstance(fileobj, (bytes, bytearray)):
            with open(dest_path, "wb") as f:
                f.write(fileobj)
        else:
            # For BytesIO
            if isinstance(fileobj, io.BytesIO):
                with open(dest_path, "wb") as f:
                    f.write(fileobj.getvalue())
            else:
                # Text stream or StringIO
                if hasattr(fileobj, "read"):
                    with open(dest_path, "w", newline="", encoding="utf-8") as f:
                        content = fileobj.read()
                        if isinstance(content, bytes):
                            content = content.decode("utf-8", errors="replace")
                        f.write(content)
                else:
                    raise ValueError("Unsupported file object provided to save_uploaded_csv")

        logger.info(f"Saved uploaded CSV to {dest_path}")
        return dest_path
    except Exception as e:
        logger.exception("Failed to save uploaded CSV")
        raise CustomException(e, sys)


def load_csv_metadata(path: str, sample_rows: int = 5) -> Dict[str, Any]:
    """
    Read a CSV file and return basic metadata:
    {
      "table_name": <safe base filename>,
      "path": <path>,
      "columns": [col1, col2, ...],
      "sample_rows": [ [row values], ... ],
      "row_count": int
    }
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                # empty file
                header = []
                samples = []
                row_count = 0
                table_name = _sanitize_filename(os.path.basename(path))
                return {
                    "table_name": table_name,
                    "path": path,
                    "columns": header,
                    "sample_rows": samples,
                    "row_count": row_count,
                }

            samples: List[List[str]] = []
            row_count = 0
            for row in reader:
                row_count += 1
                if len(samples) < sample_rows:
                    samples.append(row)

        table_name = _sanitize_filename(os.path.splitext(os.path.basename(path))[0])
        metadata = {
            "table_name": table_name,
            "path": path,
            "columns": header,
            "sample_rows": samples,
            "row_count": row_count,
        }
        logger.info(f"Loaded metadata for {path}: columns={len(header)}, rows={row_count}")
        return metadata
    except Exception as e:
        logger.exception("Failed to load CSV metadata")
        raise CustomException(e, sys)
