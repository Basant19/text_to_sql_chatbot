# D:\text_to_sql_bot\app\config.py
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

from app.logger import get_logger, LOGS_DIR
from app.exception import CustomException

logger = get_logger("config")

# Load .env
try:
    load_dotenv()
    logger.info("Loaded environment variables from .env file (if present)")
except Exception as e:
    logger.warning("Could not load .env file: %s", e)


# -----------------------
# Helper utilities
# -----------------------
def parse_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def getenv(key: str, default=None):
    v = os.environ.get(key)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip()


# -----------------------
# Module-level defaults
# -----------------------
GEMINI_API_KEY = getenv("GEMINI_API_KEY") or getenv("GOOGLE_API_KEY")
GEMINI_ENDPOINT = getenv("GEMINI_ENDPOINT", "")
GEMINI_MODEL = getenv("GEMINI_MODEL", "gemini-2.5-flash")

LANGSMITH_API_KEY = getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = parse_bool(getenv("LANGSMITH_TRACING"), False)
LANGSMITH_ENDPOINT = getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_PROJECT = getenv("LANGSMITH_PROJECT", "default-project")
USE_LANGSMITH_FOR_GEN = parse_bool(getenv("USE_LANGSMITH_FOR_GEN"), False)

DATA_DIR = getenv("DATA_DIR", "./data")
DATABASE_PATH = getenv("DATABASE_PATH", os.path.join(DATA_DIR, "text_to_sql.db"))
UPLOAD_DIR = getenv("UPLOAD_DIR", "./uploads")
VECTOR_INDEX_PATH = getenv("VECTOR_INDEX_PATH", "./faiss/index.faiss")
HISTORY_BACKEND = getenv("HISTORY_BACKEND", "json")
LANGGRAPH_ENABLED = parse_bool(getenv("LANGGRAPH_ENABLED"), False)

VECTOR_CHUNK_SIZE = int(getenv("VECTOR_CHUNK_SIZE", 1000))
VECTOR_CHUNK_OVERLAP = int(getenv("VECTOR_CHUNK_OVERLAP", 200))

GEN_RETRIES = int(getenv("GEN_RETRIES", 2))
GEN_TIMEOUT = int(getenv("GEN_TIMEOUT", 30))
GEN_BACKOFF = float(getenv("GEN_BACKOFF", 1.0))

STORE_FULL_LLM_BLOBS = parse_bool(getenv("STORE_FULL_LLM_BLOBS"), False)
MAX_RAW_SUMMARY_CHARS = int(getenv("MAX_RAW_SUMMARY_CHARS", 800))


class Config:
    GEMINI_API_KEY: Optional[str]
    GEMINI_ENDPOINT: str
    GEMINI_MODEL: str

    LANGSMITH_API_KEY: Optional[str]
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_PROJECT: str
    USE_LANGSMITH_FOR_GEN: bool

    DATA_DIR: str
    DATABASE_PATH: str
    DATABASE_DIR: str
    UPLOAD_DIR: str
    VECTOR_INDEX_PATH: str
    VECTOR_INDEX_DIR: str
    HISTORY_BACKEND: str

    LANGGRAPH_ENABLED: bool
    VECTOR_CHUNK_SIZE: int
    VECTOR_CHUNK_OVERLAP: int

    GEN_RETRIES: int
    GEN_TIMEOUT: int
    GEN_BACKOFF: float

    STORE_FULL_LLM_BLOBS: bool
    MAX_RAW_SUMMARY_CHARS: int

    def __init__(self, require_keys: bool = True):
        if require_keys:
            gemini_key = getenv("GEMINI_API_KEY") or getenv("GOOGLE_API_KEY")
            use_langsmith_for_gen = parse_bool(getenv("USE_LANGSMITH_FOR_GEN"), default=USE_LANGSMITH_FOR_GEN)
            langsmith_key = getenv("LANGSMITH_API_KEY")

            if not gemini_key and use_langsmith_for_gen and not langsmith_key:
                raise CustomException(
                    "USE_LANGSMITH_FOR_GEN=true but LANGSMITH_API_KEY is missing. "
                    "Provide LANGSMITH_API_KEY or set USE_LANGSMITH_FOR_GEN=false and provide GEMINI_API_KEY/GOOGLE_API_KEY."
                )
            if not gemini_key and not use_langsmith_for_gen:
                raise CustomException(
                    "Missing required environment variable: GEMINI_API_KEY or GOOGLE_API_KEY."
                )

        # Load environment variables with fallbacks
        self.GEMINI_API_KEY = getenv("GEMINI_API_KEY") or getenv("GOOGLE_API_KEY") or GEMINI_API_KEY
        self.GEMINI_ENDPOINT = getenv("GEMINI_ENDPOINT", GEMINI_ENDPOINT)
        self.GEMINI_MODEL = getenv("GEMINI_MODEL", GEMINI_MODEL)

        self.LANGSMITH_API_KEY = getenv("LANGSMITH_API_KEY", LANGSMITH_API_KEY)
        self.LANGSMITH_TRACING = parse_bool(getenv("LANGSMITH_TRACING"), LANGSMITH_TRACING)
        self.LANGSMITH_ENDPOINT = getenv("LANGSMITH_ENDPOINT", LANGSMITH_ENDPOINT)
        self.LANGSMITH_PROJECT = getenv("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
        self.USE_LANGSMITH_FOR_GEN = parse_bool(getenv("USE_LANGSMITH_FOR_GEN"), USE_LANGSMITH_FOR_GEN)

        self.DATA_DIR = getenv("DATA_DIR", DATA_DIR)
        self.DATABASE_PATH = getenv("DATABASE_PATH", DATABASE_PATH)
        self.UPLOAD_DIR = getenv("UPLOAD_DIR", UPLOAD_DIR)
        self.VECTOR_INDEX_PATH = getenv("VECTOR_INDEX_PATH", VECTOR_INDEX_PATH)
        self.HISTORY_BACKEND = getenv("HISTORY_BACKEND", HISTORY_BACKEND)
        self.LANGGRAPH_ENABLED = parse_bool(getenv("LANGGRAPH_ENABLED"), LANGGRAPH_ENABLED)

        self.VECTOR_CHUNK_SIZE = int(getenv("VECTOR_CHUNK_SIZE", VECTOR_CHUNK_SIZE))
        self.VECTOR_CHUNK_OVERLAP = int(getenv("VECTOR_CHUNK_OVERLAP", VECTOR_CHUNK_OVERLAP))

        self.GEN_RETRIES = int(getenv("GEN_RETRIES", GEN_RETRIES))
        self.GEN_TIMEOUT = int(getenv("GEN_TIMEOUT", GEN_TIMEOUT))
        self.GEN_BACKOFF = float(getenv("GEN_BACKOFF", GEN_BACKOFF))

        self.STORE_FULL_LLM_BLOBS = parse_bool(getenv("STORE_FULL_LLM_BLOBS"), STORE_FULL_LLM_BLOBS)
        self.MAX_RAW_SUMMARY_CHARS = int(getenv("MAX_RAW_SUMMARY_CHARS", MAX_RAW_SUMMARY_CHARS))

        # Derived directories
        self.DATABASE_DIR = str(Path(self.DATABASE_PATH).parent)
        self.VECTOR_INDEX_DIR = str(Path(self.VECTOR_INDEX_PATH).parent)

        # Ensure directories exist
        for dir_path in [self.DATA_DIR, self.DATABASE_DIR, self.UPLOAD_DIR, self.VECTOR_INDEX_DIR, LOGS_DIR]:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.info("Ensured directory exists: %s", dir_path)
            except Exception as e:
                logger.exception("Failed to create directory %s: %s", dir_path, e)
                raise CustomException(f"Failed to create directory {dir_path}: {e}")

        logger.info(
            "Config initialized: GEMINI_MODEL=%s, LANGGRAPH_ENABLED=%s, LANGSMITH_TRACING=%s, USE_LANGSMITH_FOR_GEN=%s",
            self.GEMINI_MODEL,
            self.LANGGRAPH_ENABLED,
            self.LANGSMITH_TRACING,
            self.USE_LANGSMITH_FOR_GEN,
        )

    def get(self, name: str, default=None):
        return getattr(self, name, default)

    def __repr__(self):
        return (
            f"Config(GEMINI_MODEL={self.GEMINI_MODEL}, "
            f"LANGGRAPH_ENABLED={self.LANGGRAPH_ENABLED}, "
            f"LANGSMITH_TRACING={self.LANGSMITH_TRACING}, "
            f"USE_LANGSMITH_FOR_GEN={self.USE_LANGSMITH_FOR_GEN})"
        )
