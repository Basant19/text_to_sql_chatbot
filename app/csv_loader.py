# app/csv_loader.py
import os
import io
import csv
import uuid
from typing import Dict, List, Any, Union, TextIO, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("csv_loader")


def _sanitize_filename(name: str) -> str:
    """
    Make a filename safe by replacing unsafe characters.
    Keeps alphanumeric, dot, underscore, dash, and replaces spaces with underscores.
    """
    keepchars = (".", "_", "-")
    safe = "".join(c if c.isalnum() or c in keepchars else "_" for c in name)
    return safe.strip().replace(" ", "_")


def _unique_path(dest_dir: str, filename: str) -> str:
    """
    Ensure a filename is unique in the destination directory.
    Appends a UUID if file exists.
    """
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    if not os.path.exists(candidate):
        return candidate
    unique_name = f"{base}_{uuid.uuid4().hex}{ext}"
    return os.path.join(dest_dir, unique_name)


def save_uploaded_csv(fileobj: Union[bytes, TextIO, io.StringIO, io.BytesIO], filename: str) -> str:
    """
    Save an uploaded CSV to disk safely.
    Supports bytes, file-like objects, and StringIO/BytesIO.
    Returns the final path of the saved file.
    """
    try:
        upload_dir = getattr(config, "UPLOAD_DIR", os.path.join(os.getcwd(), "uploads"))
        os.makedirs(upload_dir, exist_ok=True)

        safe_name = _sanitize_filename(filename)
        if not safe_name.lower().endswith(".csv"):
            # ensure extension for clarity
            safe_name = safe_name + ".csv"

        dest_path = _unique_path(upload_dir, safe_name)

        # handle bytes-like
        if isinstance(fileobj, (bytes, bytearray)):
            with open(dest_path, "wb") as f:
                f.write(fileobj)
        # BytesIO
        elif isinstance(fileobj, io.BytesIO):
            with open(dest_path, "wb") as f:
                f.write(fileobj.getvalue())
        # TextIO / file-like (e.g., Streamlit UploadedFile object has .read())
        elif hasattr(fileobj, "read"):
            # Read content; if bytes, decode
            content = fileobj.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            # ensure newline handling is normalized
            with open(dest_path, "w", newline="", encoding="utf-8") as f:
                f.write(content)
        else:
            raise ValueError("Unsupported file object provided to save_uploaded_csv")

        logger.info("Saved uploaded CSV to %s", dest_path)
        return dest_path
    except Exception as e:
        logger.exception("Failed to save uploaded CSV")
        raise CustomException(e)


def _canonical_table_name_from_path(path: str) -> str:
    """Return sanitized canonical table name (filename without extension)."""
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    return _sanitize_filename(name)


def load_csv_metadata(path: str, sample_rows: int = 5) -> Dict[str, Any]:
    """
    Load CSV metadata: columns, sample rows, and total row count.
    Returns a dict suitable for schema display or ingestion into SchemaStore.

    Returned structure:
      {
        "table_name": "<sanitized basename>",
        "path": "<path>",
        "columns": [list of header names],
        "sample_rows": [list of sample row dicts or lists],
        "row_count": <int>
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
                # Empty CSV
                return {
                    "table_name": _canonical_table_name_from_path(path),
                    "path": path,
                    "columns": [],
                    "sample_rows": [],
                    "row_count": 0,
                }

            samples: List[List[str]] = []
            row_count = 0
            for row in reader:
                row_count += 1
                if len(samples) < sample_rows:
                    samples.append(row)

        table_name = _canonical_table_name_from_path(path)
        metadata = {
            "table_name": table_name,
            "path": path,
            "columns": header,
            "sample_rows": samples,
            "row_count": row_count,
        }
        logger.info("Loaded metadata for %s: columns=%d, rows=%d", path, len(header), row_count)
        return metadata
    except Exception as e:
        logger.exception("Failed to load CSV metadata for %s", path)
        raise CustomException(e)


class CSVLoader:
    """
    Centralized CSV management class.
    Tracks uploaded files and exposes metadata extraction.

    Notes:
      - upload directory is controlled by config.UPLOAD_DIR (defaults to ./uploads)
      - list_uploaded_csvs() scans the upload dir so results persist across restarts
    """

    def __init__(self, upload_dir: Optional[str] = None):
        self.upload_dir = upload_dir or getattr(config, "UPLOAD_DIR", os.path.join(os.getcwd(), "uploads"))
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_csv(self, file) -> str:
        """
        Save a file object to disk and return its path.

        Accepts:
          - Streamlit UploadedFile (has .name and .read())
          - bytes / BytesIO / StringIO / file-like
        """
        try:
            filename = getattr(file, "name", None) or f"upload_{uuid.uuid4().hex}.csv"
            path = save_uploaded_csv(file, filename)
            logger.info("CSVLoader.save_csv -> %s", path)
            return path
        except Exception as e:
            logger.exception("CSVLoader.save_csv failed")
            raise CustomException(e)

    def list_uploaded_csvs(self) -> List[str]:
        """Return the list of saved CSV paths by scanning the upload directory."""
        try:
            out: List[str] = []
            for entry in os.listdir(self.upload_dir):
                p = os.path.join(self.upload_dir, entry)
                if os.path.isfile(p) and entry.lower().endswith(".csv"):
                    out.append(p)
            # sort by mtime (most recent first)
            out.sort(key=lambda p: os.path.getmtime(p), reverse=False)
            return out
        except Exception as e:
            logger.exception("Failed to list uploaded CSVs from %s", self.upload_dir)
            raise CustomException(e)

    def load_and_extract(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Load metadata for multiple CSV files.
        Returns list of dicts containing columns, samples, row counts, and table_name.
        """
        schemas: List[Dict[str, Any]] = []
        for path in file_paths:
            try:
                metadata = load_csv_metadata(path)
                schemas.append(metadata)
            except Exception as e:
                logger.warning("Skipping CSV %s due to load failure: %s", path, e)
        return schemas
