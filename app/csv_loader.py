# app/csv_loader.py
import os
import io
import sys
import uuid
import re
import csv
import hashlib
from typing import Union, List, Dict, Any, Optional
from app.logger import get_logger
from app.exception import CustomException
from app import config

logger = get_logger("csv_loader")

# Text splitter — REQUIRE the modern package and fail fast with a clear message if missing.
try:
    # Preferred import per LangChain docs:
    #   from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _USE_TEXT_SPLITTER = True
    logger.info("Using RecursiveCharacterTextSplitter from langchain_text_splitters")
except Exception as e:
    _USE_TEXT_SPLITTER = False
    logger.exception("RecursiveCharacterTextSplitter import failed: %s", e)
    raise ImportError(
        "RecursiveCharacterTextSplitter (langchain-text-splitters) is required. "
        "Install it with `pip install langchain-text-splitters` and ensure you are "
        "in the correct virtualenv. Original error: " + str(e)
    ) from e


def _sanitize_filename(name: str) -> str:
    keepchars = (".", "_", "-")
    safe = "".join(c if c.isalnum() or c in keepchars else "_" for c in name)
    return safe.strip().replace(" ", "_")


def _compute_sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def save_uploaded_csv(
    fileobj: Union[bytes, io.BytesIO, io.StringIO, str],
    filename: str,
    upload_dir: Optional[str] = None,
    *,
    make_content_addressed: bool = True,
) -> str:
    """
    Save uploaded CSV content to upload_dir and return the saved path.

    Behavior:
    - If make_content_addressed is True: compute SHA256 of the file bytes and
      prefix filename with the short hash. If a file with the same hash+name
      already exists and sizes match, reuse it (no re-write).
    - Accepts:
        - bytes / bytearray
        - file-like object with .read()
        - path string (file copy)
    - Returns absolute path to saved file.
    """
    upload_dir = upload_dir or getattr(config, "UPLOAD_DIR", "./uploads")
    os.makedirs(upload_dir, exist_ok=True)

    safe_name = _sanitize_filename(filename)
    if not safe_name.lower().endswith(".csv"):
        safe_name += ".csv"

    try:
        # Normalize into bytes
        data_bytes: Optional[bytes] = None

        # If fileobj is a path string, read bytes from that file
        if isinstance(fileobj, str) and os.path.exists(fileobj):
            with open(fileobj, "rb") as f:
                data_bytes = f.read()

        # Bytes-like
        elif isinstance(fileobj, (bytes, bytearray)):
            data_bytes = bytes(fileobj)

        # File-like object with read()
        elif hasattr(fileobj, "read"):
            # Try to rewind if possible
            try:
                fileobj.seek(0)
            except Exception:
                pass

            content = fileobj.read()
            # If the read returned text, encode; if bytes, use directly
            if isinstance(content, str):
                data_bytes = content.encode("utf-8")
            elif isinstance(content, (bytes, bytearray)):
                data_bytes = bytes(content)
            else:
                # Fallback to str conversion
                data_bytes = str(content).encode("utf-8")

        else:
            raise ValueError("Unsupported file object provided to save_uploaded_csv")

        if data_bytes is None:
            raise ValueError("No data read from file object")

        # Compute short sha prefix for filename if requested
        if make_content_addressed:
            full_hash = _compute_sha256_bytes(data_bytes)
            short_hash = full_hash[:32]
            out_name = f"{short_hash}_{safe_name}"
        else:
            out_name = f"{uuid.uuid4().hex}_{safe_name}"

        dest_path = os.path.join(upload_dir, out_name)

        # If file already exists with same size, assume identical and reuse
        if os.path.exists(dest_path):
            try:
                if os.path.getsize(dest_path) == len(data_bytes):
                    logger.debug("save_uploaded_csv: reusing existing file %s", dest_path)
                    return os.path.abspath(dest_path)
            except Exception:
                # fall through to re-write
                logger.debug("save_uploaded_csv: failed to stat existing file; will overwrite", exc_info=True)

        # Write atomically
        tmp = dest_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data_bytes)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, dest_path)

        logger.info("Saved uploaded CSV to %s", dest_path)
        return os.path.abspath(dest_path)
    except Exception as e:
        logger.exception("Failed to save uploaded CSV")
        raise CustomException(e, sys)


def _normalize_table_name_from_filename(filename: str) -> str:
    """
    Returns a DuckDB-safe physical table name.

    RULES:
    - Always lowercase
    - Always prefix with `t_`
    - Never start with a digit
    - Deterministic for same filename
    """
    stem = os.path.splitext(os.path.basename(filename))[0]

    # Strip hash suffix added by save_uploaded_csv
    m = re.match(r"^(.+?)_[0-9a-fA-F]{8,64}$", stem)
    base = m.group(1) if m else stem

    base = _sanitize_filename(base).lower()

    # Collapse multiple underscores
    base = re.sub(r"_+", "_", base).strip("_")

    # 🚨 ALWAYS prefix with t_
    return f"t_{base}"



def load_csv_metadata(path: str, sample_rows: int = 5) -> Dict[str, Any]:
    """
    Reads CSV header & a few sample rows. Also detects delimiter/header heuristically.
    Returns columns (raw), columns_normalized, aliases, etc.

    Empty rows (all cells empty/whitespace) are ignored for row_count and samples.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        # Read initial bytes for sniffing and to detect encoding quirks
        with open(path, "rb") as bf:
            raw_start = bf.read(8192)
            try:
                start_text = raw_start.decode("utf-8", errors="replace")
            except Exception:
                start_text = raw_start.decode("latin-1", errors="replace")

        # Attempt to sniff delimiter and header. Fall back to sensible defaults.
        from csv import Sniffer
        try:
            sniffer = Sniffer()
            dialect = sniffer.sniff(start_text)
            delimiter = dialect.delimiter
            has_header = sniffer.has_header(start_text)
        except Exception:
            delimiter = ","
            has_header = True

        # Open as text for CSV reader
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            try:
                first_row = next(reader)
            except StopIteration:
                # file empty
                table_name = _normalize_table_name_from_filename(os.path.basename(path))
                canonical = _sanitize_filename(os.path.splitext(os.path.basename(path))[0]).lower()
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
                    "delimiter": delimiter,
                    "has_header": has_header,
                }

            # If Sniffer says no header, apply heuristic: if the first row contains alphabetic
            # characters — treat it as a header to avoid mis-detections.
            if not has_header and any(re.search(r"[A-Za-z]", str(cell or "")) for cell in first_row):
                has_header = True

            # Handle BOM in first header cell
            def _clean_cell(h: str) -> str:
                return h.strip().replace("\ufeff", "")

            if has_header:
                headers = [_clean_cell(h) for h in first_row]
                # reader continues from second line
                data_iter = reader
            else:
                # no header: generate header names, but include first_row as data
                headers = [f"col_{i+1}" for i in range(len(first_row))]
                # create an iterator with first_row as the first data row
                data_iter = iter([first_row] + list(reader))

            def normalize_col_name(h: str) -> str:
                # insert underscore between letter->digit and digit->letter boundaries:
                s = re.sub(r"(?<=[A-Za-z])(?=\d)", "_", h)
                s = re.sub(r"(?<=\d)(?=[A-Za-z])", "_", s)
                # replace any non-word char with underscore
                s = re.sub(r"[^\w]", "_", s)
                # collapse multiple underscores
                s = re.sub(r"_+", "_", s)
                s = s.strip("_").lower()
                if not s:
                    s = "col"
                return s

            columns_normalized = [normalize_col_name(h) for h in headers]

            samples: List[Dict[str, Any]] = []
            row_count = 0
            for row in data_iter:
                # skip rows that are entirely empty (all cells empty or whitespace)
                if not any((cell or "").strip() for cell in row):
                    continue
                row_count += 1
                if len(samples) < sample_rows:
                    # pad or truncate to header length
                    values = list(row) + [None] * (len(headers) - len(row))
                    values = values[: len(headers)]
                    row_dict = {headers[i]: values[i] for i in range(len(headers))}
                    samples.append(row_dict)

            table_name = _normalize_table_name_from_filename(os.path.basename(path))
            canonical = _sanitize_filename(os.path.splitext(os.path.basename(path))[0]).lower()
            aliases = []
            for a in (table_name, canonical, os.path.splitext(os.path.basename(path))[0]):
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
                "delimiter": delimiter,
                "has_header": has_header,
            }

        logger.info("Loaded metadata for %s: columns=%d, rows=%d", path, len(headers), row_count)
        return metadata
    except Exception as e:
        logger.exception("Failed to load CSV metadata for %s", path)
        raise CustomException(e, sys)


class CSVLoader:
    def __init__(self, upload_dir: Optional[str] = None, chunk_size: int = 500, chunk_overlap: int = 50):
        self.upload_dir = upload_dir or getattr(config, "UPLOAD_DIR", os.path.join(os.getcwd(), "uploads"))
        os.makedirs(self.upload_dir, exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = None
        if _USE_TEXT_SPLITTER:
            # Instantiate the splitter per LangChain docs.
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def save_csv(self, file) -> str:
        """
        Save an uploaded file-like / bytes / path. Will return a content-addressed
        path (SHA-prefixed) to avoid duplicate copies on disk.
        """
        filename = getattr(file, "name", None) or f"upload_{uuid.uuid4().hex}.csv"

        # If file is a file-like, read bytes then call save_uploaded_csv to ensure
        # consistent hashing. For streamlit UploadedFile, .read() returns bytes.
        try:
            # If file is a path string, hand off directly (save_uploaded_csv will read it)
            if isinstance(file, str) and os.path.exists(file):
                return save_uploaded_csv(file, filename, upload_dir=self.upload_dir, make_content_addressed=True)

            # If file has read(), use it but ensure we don't consume original if caller still needs it.
            if hasattr(file, "read"):
                try:
                    file.seek(0)
                except Exception:
                    pass
                content = file.read()
                # If the object returned text instead of bytes, encode:
                if isinstance(content, str):
                    content = content.encode("utf-8")
                # create BytesIO to pass to save_uploaded_csv (so we don't depend on original file object lifetime)
                bio = io.BytesIO(content)
                try:
                    bio.name = filename
                except Exception:
                    pass
                return save_uploaded_csv(bio, filename, upload_dir=self.upload_dir, make_content_addressed=True)

            # Bytes-like
            if isinstance(file, (bytes, bytearray)):
                return save_uploaded_csv(bytes(file), filename, upload_dir=self.upload_dir, make_content_addressed=True)

            # Fallback: try to stringify
            return save_uploaded_csv(str(file).encode("utf-8"), filename, upload_dir=self.upload_dir, make_content_addressed=True)
        except Exception as e:
            logger.exception("CSVLoader.save_csv failed")
            raise CustomException(e, sys)

    def list_uploaded_csvs(self) -> List[str]:
        try:
            out: List[str] = []
            for entry in os.listdir(self.upload_dir):
                p = os.path.join(self.upload_dir, entry)
                if os.path.isfile(p) and entry.lower().endswith(".csv"):
                    out.append(p)
            out.sort(key=lambda p: os.path.getmtime(p))
            return out
        except Exception as e:
            logger.exception("Failed to list uploaded CSVs from %s", self.upload_dir)
            raise CustomException(e, sys)

    def load_and_extract(self, paths: List[str]) -> List[Dict[str, Any]]:
        """
        Backwards-compatible convenience used by app.py.

        Accepts a list of file paths (to CSV files), calls load_csv_metadata on each,
        and returns a list of metadata dicts (one per input path).
        """
        out_meta: List[Dict[str, Any]] = []
        for p in paths or []:
            try:
                if not p or not os.path.exists(p):
                    logger.warning("load_and_extract: path missing or not found: %s", p)
                    continue
                m = load_csv_metadata(p)
                # ensure 'path' set and canonical/table fields present
                m.setdefault("path", p)
                m.setdefault("canonical_name", m.get("canonical_name") or _sanitize_filename(os.path.splitext(os.path.basename(p))[0]).lower())
                m.setdefault("table_name", m.get("table_name") or _normalize_table_name_from_filename(os.path.basename(p)))
                out_meta.append(m)
            except Exception:
                logger.exception("load_and_extract: failed to extract metadata for %s", p)
        return out_meta

    def load_and_chunk_csv(self, path: str) -> List[Dict[str, Any]]:
        """
        Reads CSV using delimiter/header detected by load_csv_metadata, splits long text cells into chunks,
        returns list of chunks with metadata.
        """
        try:
            metadata = load_csv_metadata(path)
            delimiter = metadata.get("delimiter", ",")
            table = metadata.get("table_name")
            canonical = metadata.get("canonical_name")

            chunks: List[Dict[str, Any]] = []

            # Use DictReader when we have headers, otherwise regular reader.
            with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
                if metadata.get("has_header", True):
                    reader = csv.DictReader(f, delimiter=delimiter)
                    for row_idx, row in enumerate(reader):
                        for col_name, cell in row.items():
                            text = str(cell or "")
                            if self.splitter and text:
                                split_texts = self.splitter.split_text(text)
                            else:
                                split_texts = [text]
                            for chunk_text in split_texts:
                                chunks.append({
                                    "text": chunk_text,
                                    "meta": {
                                        "table": table,
                                        "canonical": canonical,
                                        "row": row_idx,
                                        "column": col_name,
                                        "original": cell
                                    }
                                })
                else:
                    reader = csv.reader(f, delimiter=delimiter)
                    for row_idx, row in enumerate(reader):
                        for i, cell in enumerate(row):
                            col_name = f"col_{i+1}"
                            text = str(cell or "")
                            if self.splitter and text:
                                split_texts = self.splitter.split_text(text)
                            else:
                                split_texts = [text]
                            for chunk_text in split_texts:
                                chunks.append({
                                    "text": chunk_text,
                                    "meta": {
                                        "table": table,
                                        "canonical": canonical,
                                        "row": row_idx,
                                        "column": col_name,
                                        "original": cell
                                    }
                                })

            logger.info("CSV %s chunked into %d pieces (table=%s)", path, len(chunks), table)
            return chunks
        except Exception as e:
            logger.exception("Failed to load and chunk CSV %s", path)
            raise CustomException(e, sys)

    def chunk_and_index(self, path: str, vector_search_client=None, id_prefix: Optional[str] = None) -> List[str]:
        """
        Convenience helper: chunk CSV and optionally upsert chunks into a VectorSearch client.
        """
        chunks = self.load_and_chunk_csv(path)
        docs = []
        ids = []
        for i, c in enumerate(chunks):
            doc_id = f"{id_prefix + '_' if id_prefix else ''}{uuid.uuid4().hex}"
            docs.append({"id": doc_id, "text": c["text"], "meta": c["meta"]})
            ids.append(doc_id)

        if vector_search_client:
            # Upsert in batches (if client supports batch upsert)
            try:
                # Try single call
                vector_search_client.upsert_documents(docs)
            except Exception:
                # try smaller batches if large
                batch_size = int(getattr(config, "VECTOR_UPSERT_BATCH", 64))
                for start in range(0, len(docs), batch_size):
                    batch = docs[start:start + batch_size]
                    vector_search_client.upsert_documents(batch)

        return ids
