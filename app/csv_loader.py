# app/csv_loader.py
import os
import io
import uuid
import re
import csv
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
    # Fail fast — user explicitly requested only this splitter.
    # Raise a helpful ImportError so CI/devs see what's missing immediately.
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


def save_uploaded_csv(fileobj: Union[bytes, io.BytesIO, io.StringIO], filename: str, upload_dir: Optional[str] = None) -> str:
    upload_dir = upload_dir or getattr(config, "UPLOAD_DIR", "./uploads")
    os.makedirs(upload_dir, exist_ok=True)

    safe_name = _sanitize_filename(filename)
    if not safe_name.lower().endswith(".csv"):
        safe_name += ".csv"

    dest_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}_{safe_name}")

    try:
        if isinstance(fileobj, (bytes, bytearray)):
            with open(dest_path, "wb") as f:
                f.write(fileobj)
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


def _normalize_table_name_from_filename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"^(.+?)_[0-9a-fA-F]{8,32}$", stem)
    candidate = m.group(1) if m else stem
    candidate = _sanitize_filename(candidate)
    if candidate.isdigit():
        candidate = f"t_{candidate}"
    return candidate.lower()

def load_csv_metadata(path: str, sample_rows: int = 5) -> Dict[str, Any]:
    """
    Reads CSV header & a few sample rows. Also detects delimiter/header heuristically.
    Returns columns (raw), columns_normalized, aliases, etc.
    Empty rows (all cells empty/whitespace) are ignored for row_count and samples.

    Heuristics:
    - If Sniffer says no header but the first row contains alphabetic characters,
      treat the first row as header (fixes Sniffer mis-detections).
    - Normalized column names insert underscores between letter<->digit boundaries
      and replace non-word chars with underscores, collapsing repeated underscores.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        # Peek first bytes for sniffing
        with open(path, "rb") as bf:
            raw_start = bf.read(8192)
            try:
                start_text = raw_start.decode("utf-8", errors="replace")
            except Exception:
                start_text = raw_start.decode("latin-1", errors="replace")

        from csv import Sniffer
        try:
            sniffer = Sniffer()
            dialect = sniffer.sniff(start_text)
            delimiter = dialect.delimiter
            has_header = sniffer.has_header(start_text)
        except Exception:
            delimiter = ","
            has_header = True

        with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            try:
                first_row = next(reader)
            except StopIteration:
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
            # characters (e.g. "col1", "name") — treat it as a header to avoid miscounting.
            if not has_header and any(re.search(r"[A-Za-z]", str(cell or "")) for cell in first_row):
                has_header = True

            if has_header:
                headers = [h.strip().replace("\ufeff", "") for h in first_row]
            else:
                headers = [f"col_{i+1}" for i in range(len(first_row))]
                # put first_row back into iteration as data row
                reader = iter([first_row] + list(reader))

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
            for row in reader:
                # skip rows that are entirely empty (all cells empty or whitespace)
                if not any((cell or "").strip() for cell in row):
                    continue
                row_count += 1
                if len(samples) < sample_rows:
                    # normalize length of row to headers length
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
        raise CustomException(e)



class CSVLoader:
    def __init__(self, upload_dir: Optional[str] = None, chunk_size: int = 500, chunk_overlap: int = 50):
        self.upload_dir = upload_dir or getattr(config, "UPLOAD_DIR", os.path.join(os.getcwd(), "uploads"))
        os.makedirs(self.upload_dir, exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = None
        if _USE_TEXT_SPLITTER:
            # Instantiate the splitter per LangChain docs.
            # Example usage:
            #   text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
            #   texts = text_splitter.split_text(document)
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def save_csv(self, file) -> str:
        filename = getattr(file, "name", None) or f"upload_{uuid.uuid4().hex}.csv"
        return save_uploaded_csv(file, filename, upload_dir=self.upload_dir)

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
            raise CustomException(e)

    def load_and_chunk_csv(self, path: str) -> List[Dict[str, Any]]:
        """
        Reads CSV using delimiter/header detected by load_csv_metadata, splits long text cells into chunks,
        returns list of chunks with metadata.

        Each chunk: {
            "text": <chunk_text>,
            "meta": {
                "table": <table_name>,
                "canonical": <canonical_name>,
                "row": <row_index>,
                "column": <column_name>,
                "original": <original_cell_value>
            }
        }
        """
        try:
            metadata = load_csv_metadata(path)
            delimiter = metadata.get("delimiter", ",")
            table = metadata.get("table_name")
            canonical = metadata.get("canonical_name")

            chunks: List[Dict[str, Any]] = []

            with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter=delimiter) if metadata.get("has_header", True) else csv.reader(f, delimiter=delimiter)
                if metadata.get("has_header", True):
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
                    # no header: treat columns as col_1, col_2...
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
            raise CustomException(e)

    def chunk_and_index(self, path: str, vector_search_client=None, id_prefix: Optional[str] = None) -> List[str]:
        """
        Convenience helper: chunk CSV and optionally upsert chunks into a VectorSearch client.

        - vector_search_client: an object exposing upsert_documents(list_of_docs) where each doc is:
            {"id": <id>, "text": <text>, "meta": {...}}
          If None, this method returns chunks but does not index.

        - id_prefix: optional prefix to use for doc ids.
        Returns list of document ids that were upserted (or an empty list if vector_search_client is None).
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
                vector_search_client.upsert_documents(docs)
            except Exception:
                # try smaller batches if large
                batch_size = int(getattr(config, "VECTOR_UPSERT_BATCH", 64))
                for start in range(0, len(docs), batch_size):
                    batch = docs[start:start + batch_size]
                    vector_search_client.upsert_documents(batch)

        return ids
