# app/csv_loader.py
import os
import io
import csv
import re
import uuid
from typing import Dict, List, Any, Union, TextIO, Optional

from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("csv_loader")


def _sanitize_filename(name: str) -> str:
    keepchars = (".", "_", "-")
    safe = "".join(c if c.isalnum() or c in keepchars else "_" for c in name)
    return safe.strip().replace(" ", "_")


def _unique_path(dest_dir: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    if not os.path.exists(candidate):
        return candidate
    unique_name = f"{base}_{uuid.uuid4().hex}{ext}"
    return os.path.join(dest_dir, unique_name)


def _normalize_table_name_from_filename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"^(.+?)_[0-9a-fA-F]{8,32}$", stem)
    if m:
        candidate = m.group(1)
    else:
        candidate = stem
    candidate = _sanitize_filename(candidate)
    if candidate.isdigit():
        candidate = f"t_{candidate}"
    return candidate.lower()


def save_uploaded_csv(fileobj: Union[bytes, TextIO, io.StringIO, io.BytesIO], filename: str) -> str:
    try:
        upload_dir = getattr(config, "UPLOAD_DIR", os.path.join(os.getcwd(), "uploads"))
        os.makedirs(upload_dir, exist_ok=True)

        safe_name = _sanitize_filename(filename)
        if not safe_name.lower().endswith(".csv"):
            safe_name = safe_name + ".csv"

        dest_path = _unique_path(upload_dir, safe_name)

        if isinstance(fileobj, (bytes, bytearray)):
            with open(dest_path, "wb") as f:
                f.write(fileobj)
        elif isinstance(fileobj, io.BytesIO):
            with open(dest_path, "wb") as f:
                f.write(fileobj.getvalue())
        elif hasattr(fileobj, "read"):
            content = fileobj.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
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
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    return _sanitize_filename(name).lower()


def _strip_bom_and_normalize(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    for bom in ("\ufeff", "\ufffe"):
        s = s.replace(bom, "")
    return s.strip()


def _to_snake_case(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s.lower()


def _detect_dialect_and_header(sample: str) -> Dict[str, Any]:
    result = {"delimiter": ",", "has_header": True}
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        result["delimiter"] = dialect.delimiter
        try:
            result["has_header"] = sniffer.has_header(sample)
        except Exception:
            result["has_header"] = True
    except Exception:
        result["delimiter"] = ","
        result["has_header"] = True
    return result


def load_csv_metadata(path: str, sample_rows: int = 5) -> Dict[str, Any]:
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        with open(path, "rb") as bf:
            raw_start = bf.read(8192)
            try:
                start_text = raw_start.decode("utf-8", errors="replace")
            except Exception:
                start_text = raw_start.decode("latin-1", errors="replace")

        sniff = _detect_dialect_and_header(start_text)
        delimiter = sniff.get("delimiter", ",")
        has_header = sniff.get("has_header", True)

        with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            try:
                first_row = next(reader)
            except StopIteration:
                table_name = _normalize_table_name_from_filename(os.path.basename(path))
                canonical = _canonical_table_name_from_path(path)
                aliases = list({canonical, table_name})
                return {
                    "table_name": table_name,
                    "canonical_name": canonical,
                    "aliases": aliases,
                    "path": path,
                    "original_name": os.path.basename(path),
                    "columns": [],
                    "columns_normalized": [],
                    "sample_rows": [],
                    "row_count": 0,
                    "file_size_bytes": os.path.getsize(path),
                }

            if has_header:
                raw_headers = [_strip_bom_and_normalize(h) for h in first_row]
            else:
                raw_headers = [f"col_{i+1}" for i in range(len(first_row))]
                reader = iter([first_row] + list(reader))

            headers = [h if isinstance(h, str) else str(h) for h in raw_headers]
            headers = [h.strip() for h in headers]
            columns_normalized = [_to_snake_case(h) for h in headers]

            samples: List[Dict[str, Any]] = []
            row_count = 0
            for row in reader:
                row_count += 1
                if len(samples) < sample_rows:
                    values = [v for v in row]
                    if len(values) < len(headers):
                        values = values + [None] * (len(headers) - len(values))
                    elif len(values) > len(headers):
                        values = values[:len(headers)]
                    row_dict = {headers[i]: values[i] for i in range(len(headers))}
                    samples.append(row_dict)

            table_name = _normalize_table_name_from_filename(os.path.basename(path))
            canonical = _canonical_table_name_from_path(path)
            orig_stem = os.path.splitext(os.path.basename(path))[0]
            aliases = []
            for a in (table_name, orig_stem, canonical):
                if a and a not in aliases:
                    aliases.append(a)

            metadata = {
                "table_name": table_name,
                "canonical_name": canonical,
                "aliases": aliases,
                "path": path,
                "original_name": os.path.basename(path),
                "columns": headers,
                "columns_normalized": columns_normalized,
                "sample_rows": samples,
                "row_count": row_count,
                "file_size_bytes": os.path.getsize(path),
            }

        logger.info("Loaded metadata for %s: columns=%d, rows=%d", path, len(headers), row_count)
        return metadata
    except Exception as e:
        logger.exception("Failed to load CSV metadata for %s", path)
        raise CustomException(e)


class CSVLoader:
    def __init__(self, upload_dir: Optional[str] = None):
        self.upload_dir = upload_dir or getattr(config, "UPLOAD_DIR", os.path.join(os.getcwd(), "uploads"))
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_csv(self, file) -> str:
        try:
            filename = getattr(file, "name", None) or f"upload_{uuid.uuid4().hex}.csv"
            path = save_uploaded_csv(file, filename)
            logger.info("CSVLoader.save_csv -> %s", path)
            return path
        except Exception as e:
            logger.exception("CSVLoader.save_csv failed")
            raise CustomException(e)

    def list_uploaded_csvs(self) -> List[str]:
        try:
            out: List[str] = []
            for entry in os.listdir(self.upload_dir):
                p = os.path.join(self.upload_dir, entry)
                if os.path.isfile(p) and entry.lower().endswith(".csv"):
                    out.append(p)
            out.sort(key=lambda p: os.path.getmtime(p), reverse=False)
            return out
        except Exception as e:
            logger.exception("Failed to list uploaded CSVs from %s", self.upload_dir)
            raise CustomException(e)

    def load_and_extract(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        for path in file_paths:
            try:
                metadata = load_csv_metadata(path)
                schemas.append(metadata)
            except Exception as e:
                logger.warning("Skipping CSV %s due to load failure: %s", path, e)
        return schemas
