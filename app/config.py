import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

from app.logger import get_logger, LOGS_DIR
from app.exception import CustomException

logger = get_logger("config")

# Load .env (best-effort)
try:
    load_dotenv()
    logger.info("Loaded environment variables from .env file (if present)")
except Exception as e:
    logger.warning("Could not load .env file (or none present): %s", e)


# -----------------------
# Helper utilities
# -----------------------
def parse_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def getenv(key: str, default=None):
    """
    Read environment variable and treat empty / whitespace-only values as missing.
    This avoids subtle bugs where a variable exists but is blank.
    """
    v = os.environ.get(key)
    if v is None:
        return default
    v_str = str(v).strip()
    return v_str if v_str != "" else default


# -----------------------
# Module-level defaults
# -----------------------

# API keys (support both GEMINI_API_KEY and GOOGLE_API_KEY for backward compatibility)
GEMINI_API_KEY = getenv("GEMINI_API_KEY") or getenv("GOOGLE_API_KEY")

# Optional endpoint override (most SDKs won't require this)
GEMINI_ENDPOINT = getenv("GEMINI_ENDPOINT", "")

# Default Gemini model (configurable via .env)
GEMINI_MODEL = getenv("GEMINI_MODEL", "gemini-2.5-flash")

# LangSmith (observability/tracing) - optional for runtime (not required for generation by default)
LANGSMITH_API_KEY = getenv("LANGSMITH_API_KEY")
# Default to False to avoid noisy 404s unless explicitly enabled
LANGSMITH_TRACING = parse_bool(getenv("LANGSMITH_TRACING"), default=False)
LANGSMITH_ENDPOINT = getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_PROJECT = getenv("LANGSMITH_PROJECT", "default-project")

# Safety guard: prevent accidental use of LangSmith for generation unless explicit opt-in
USE_LANGSMITH_FOR_GEN = parse_bool(getenv("USE_LANGSMITH_FOR_GEN"), default=False)

# Storage / app paths
DATA_DIR = getenv("DATA_DIR", "./data")
DATABASE_PATH = getenv("DATABASE_PATH", os.path.join(DATA_DIR, "text_to_sql.db"))
UPLOAD_DIR = getenv("UPLOAD_DIR", "./uploads")
VECTOR_INDEX_PATH = getenv("VECTOR_INDEX_PATH", "./faiss/index.faiss")
HISTORY_BACKEND = getenv("HISTORY_BACKEND", "json")  # used by app.py

# LangGraph toggle
LANGGRAPH_ENABLED = parse_bool(getenv("LANGGRAPH_ENABLED"), default=False)

# Vector chunking defaults (configurable)
try:
    VECTOR_CHUNK_SIZE = int(getenv("VECTOR_CHUNK_SIZE", 1000))
except Exception:
    VECTOR_CHUNK_SIZE = 1000

try:
    VECTOR_CHUNK_OVERLAP = int(getenv("VECTOR_CHUNK_OVERLAP", 200))
except Exception:
    VECTOR_CHUNK_OVERLAP = 200

# Generation defaults
GEN_RETRIES = int(getenv("GEN_RETRIES", 2))
GEN_TIMEOUT = int(getenv("GEN_TIMEOUT", 30))
GEN_BACKOFF = float(getenv("GEN_BACKOFF", 1.0))

# LLM blob storage & preview limits
STORE_FULL_LLM_BLOBS = parse_bool(getenv("STORE_FULL_LLM_BLOBS"), default=False)
MAX_RAW_SUMMARY_CHARS = int(getenv("MAX_RAW_SUMMARY_CHARS", 800))


class Config:
    """
    Encapsulate configuration and ensure required environment variables and directories.

    Usage:
        cfg = Config()          # validate required keys (default)
        cfg = Config(False)     # skip required-key check (useful for tests)

    Validation logic:
      - By default (require_keys=True) we *require* GEMINI_API_KEY or GOOGLE_API_KEY.
      - If USE_LANGSMITH_FOR_GEN=true then generation via LangSmith is explicitly opted-in,
        and LANGSMITH_API_KEY must be present for generation to work.
      - LANGSMITH_API_KEY is otherwise optional (used only for tracing).
    """

    def __init__(self, require_keys: Optional[bool] = True):
        # Validate required keys based on opt-in flags
        if require_keys:
            gemini_key = getenv("GEMINI_API_KEY") or getenv("GOOGLE_API_KEY")
            use_langsmith_for_gen = parse_bool(getenv("USE_LANGSMITH_FOR_GEN"), default=USE_LANGSMITH_FOR_GEN)
            langsmith_key = getenv("LANGSMITH_API_KEY")

            if not gemini_key:
                if use_langsmith_for_gen:
                    if not langsmith_key:
                        raise CustomException(
                            "USE_LANGSMITH_FOR_GEN=true but LANGSMITH_API_KEY is missing. "
                            "Provide LANGSMITH_API_KEY or set USE_LANGSMITH_FOR_GEN=false and provide GEMINI_API_KEY/GOOGLE_API_KEY."
                        )
                    else:
                        logger.warning(
                            "No GEMINI_API_KEY/GOOGLE_API_KEY found; USE_LANGSMITH_FOR_GEN is true and LANGSMITH_API_KEY is present. "
                            "LangSmith will be used for generation — ensure this is intentional."
                        )
                else:
                    raise CustomException(
                        "Missing required environment variable: GEMINI_API_KEY or GOOGLE_API_KEY. "
                        "Set GEMINI_API_KEY (or set GOOGLE_API_KEY) for generation."
                    )

        # Populate instance attributes (module-level defaults used as fallbacks)
        self.GEMINI_API_KEY = getenv("GEMINI_API_KEY") or getenv("GOOGLE_API_KEY") or GEMINI_API_KEY
        self.GEMINI_ENDPOINT = getenv("GEMINI_ENDPOINT", GEMINI_ENDPOINT)
        self.GEMINI_MODEL = getenv("GEMINI_MODEL", GEMINI_MODEL)

        self.LANGSMITH_API_KEY = getenv("LANGSMITH_API_KEY", LANGSMITH_API_KEY)
        self.LANGSMITH_TRACING = parse_bool(getenv("LANGSMITH_TRACING"), default=LANGSMITH_TRACING)
        self.LANGSMITH_ENDPOINT = getenv("LANGSMITH_ENDPOINT", LANGSMITH_ENDPOINT)
        self.LANGSMITH_PROJECT = getenv("LANGSMITH_PROJECT", LANGSMITH_PROJECT)

        self.USE_LANGSMITH_FOR_GEN = parse_bool(getenv("USE_LANGSMITH_FOR_GEN"), default=USE_LANGSMITH_FOR_GEN)

        # Storage / paths
        self.DATA_DIR = getenv("DATA_DIR", DATA_DIR)
        self.DATABASE_PATH = getenv("DATABASE_PATH", DATABASE_PATH)
        self.UPLOAD_DIR = getenv("UPLOAD_DIR", UPLOAD_DIR)
        self.VECTOR_INDEX_PATH = getenv("VECTOR_INDEX_PATH", VECTOR_INDEX_PATH)
        self.HISTORY_BACKEND = getenv("HISTORY_BACKEND", HISTORY_BACKEND)

        self.LANGGRAPH_ENABLED = parse_bool(getenv("LANGGRAPH_ENABLED"), default=LANGGRAPH_ENABLED)

        # Vector chunking
        try:
            self.VECTOR_CHUNK_SIZE = int(getenv("VECTOR_CHUNK_SIZE", VECTOR_CHUNK_SIZE))
        except Exception:
            logger.warning("Invalid VECTOR_CHUNK_SIZE, falling back to default %s", VECTOR_CHUNK_SIZE)
            self.VECTOR_CHUNK_SIZE = VECTOR_CHUNK_SIZE

        try:
            self.VECTOR_CHUNK_OVERLAP = int(getenv("VECTOR_CHUNK_OVERLAP", VECTOR_CHUNK_OVERLAP))
        except Exception:
            logger.warning("Invalid VECTOR_CHUNK_OVERLAP, falling back to default %s", VECTOR_CHUNK_OVERLAP)
            self.VECTOR_CHUNK_OVERLAP = VECTOR_CHUNK_OVERLAP

        # Generation defaults
        try:
            self.GEN_RETRIES = int(getenv("GEN_RETRIES", GEN_RETRIES))
        except Exception:
            self.GEN_RETRIES = GEN_RETRIES

        try:
            self.GEN_TIMEOUT = int(getenv("GEN_TIMEOUT", GEN_TIMEOUT))
        except Exception:
            self.GEN_TIMEOUT = GEN_TIMEOUT

        try:
            self.GEN_BACKOFF = float(getenv("GEN_BACKOFF", GEN_BACKOFF))
        except Exception:
            self.GEN_BACKOFF = GEN_BACKOFF

        # LLM blob options
        self.STORE_FULL_LLM_BLOBS = parse_bool(getenv("STORE_FULL_LLM_BLOBS"), default=STORE_FULL_LLM_BLOBS)
        try:
            self.MAX_RAW_SUMMARY_CHARS = int(getenv("MAX_RAW_SUMMARY_CHARS", MAX_RAW_SUMMARY_CHARS))
        except Exception:
            self.MAX_RAW_SUMMARY_CHARS = MAX_RAW_SUMMARY_CHARS

        # Derived directories
        self.DATABASE_DIR = str(Path(self.DATABASE_PATH).parent) if self.DATABASE_PATH else ""
        self.VECTOR_INDEX_DIR = str(Path(self.VECTOR_INDEX_PATH).parent) if self.VECTOR_INDEX_PATH else ""
        self.DATA_DIR = self.DATA_DIR or "./data"

        # Ensure important directories exist (best-effort)
        for dir_path in [self.DATA_DIR, self.DATABASE_DIR, self.UPLOAD_DIR, self.VECTOR_INDEX_DIR, LOGS_DIR]:
            if not dir_path:
                continue
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.info("Ensured directory exists: %s", dir_path)
            except Exception as e:
                logger.exception("Failed to ensure directory %s: %s", dir_path, e)
                raise CustomException(f"Failed to create directory {dir_path}: {e}")

        # Log a safe summary of configuration (do not echo secrets)
        logger.info(
            "Config initialized: GEMINI_MODEL=%s, LANGGRAPH_ENABLED=%s, LANGSMITH_TRACING=%s, USE_LANGSMITH_FOR_GEN=%s",
            self.GEMINI_MODEL,
            self.LANGGRAPH_ENABLED,
            self.LANGSMITH_TRACING,
            self.USE_LANGSMITH_FOR_GEN,
        )

    # convenience accessor
    def get(self, name: str, default=None):
        return getattr(self, name, default)
