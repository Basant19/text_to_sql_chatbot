import os
import io
import csv
import uuid
from typing import Dict, List, Any, Union, TextIO

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("csv_loader")


def _sanitize_filename(name: str) -> str:
    """
    Make a filename safe by replacing unsafe characters.
    Keeps alphanumeric, dot, underscore, dash, and replaces spaces with underscores.
    """
    keepchars = (" ", ".", "_", "-")
    safe = "".join(c if c.isalnum() or c in keepchars else "_" for c in name)
    return safe.replace(" ", "_")


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
        dest_path = _unique_path(upload_dir, safe_name)

        if isinstance(fileobj, (bytes, bytearray)):
            with open(dest_path, "wb") as f:
                f.write(fileobj)
        elif isinstance(fileobj, io.BytesIO):
            with open(dest_path, "wb") as f:
                f.write(fileobj.getvalue())
        elif hasattr(fileobj, "read"):
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
        raise CustomException(e)


def load_csv_metadata(path: str, sample_rows: int = 5) -> Dict[str, Any]:
    """
    Load CSV metadata: columns, sample rows, and total row count.
    Returns a dict suitable for schema display or ingestion into SchemaStore.
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
                    "table_name": _sanitize_filename(os.path.basename(path)),
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
        raise CustomException(e)


class CSVLoader:
    """
    Centralized CSV management class.
    Tracks uploaded files and exposes metadata extraction.
    """

    def __init__(self):
        self.uploaded_files: List[str] = []

    def save_csv(self, file) -> str:
        """
        Save a file object to disk and track its path.
        Returns the saved file path.
        """
        try:
            path = save_uploaded_csv(file, file.name)
            self.uploaded_files.append(path)
            return path
        except Exception as e:
            logger.exception("CSVLoader.save_csv failed")
            raise CustomException(e)

    def list_uploaded_csvs(self) -> List[str]:
        """Return the list of saved CSV paths."""
        return self.uploaded_files

    def load_and_extract(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Load metadata for multiple CSV files.
        Returns list of dicts containing columns, samples, and row counts.
        """
        schemas: List[Dict[str, Any]] = []
        for path in file_paths:
            try:
                metadata = load_csv_metadata(path)
                schemas.append(metadata)
            except Exception as e:
                logger.warning(f"Skipping CSV {path} due to load failure: {e}")
        return schemas
