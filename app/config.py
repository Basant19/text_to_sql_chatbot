# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

from app.logger import get_logger, LOGS_DIR
from app.exception import CustomException

# Initialize logger for config
logger = get_logger("config")

# Load .env (best-effort)
try:
    load_dotenv()
    logger.info("Loaded environment variables from .env file (if present)")
except Exception as e:
    # dotenv failing is non-fatal here; we validate required keys below when needed.
    logger.warning("Could not load .env file (or none present): %s", e)


# -----------------------
# Module-level defaults
# -----------------------

# GEMINI_API_KEY supports both GEMINI_API_KEY and GOOGLE_API_KEY envvars (backwards compat)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# Optional endpoint override (most SDKs won't require this)
GEMINI_ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "")

# Default Gemini model (configurable via .env)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# LangSmith (observability/tracing) - optional for runtime (not required for generation by default)
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "default-project")

# Safety guard: prevent accidental use of LangSmith for generation unless explicit opt-in
USE_LANGSMITH_FOR_GEN = os.environ.get("USE_LANGSMITH_FOR_GEN", "false").lower() == "true"

# Storage / app paths
DATABASE_PATH = os.environ.get("DATABASE_PATH", "./data/text_to_sql.db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
VECTOR_INDEX_PATH = os.environ.get("VECTOR_INDEX_PATH", "./faiss/index.faiss")
HISTORY_BACKEND = os.environ.get("HISTORY_BACKEND", "json")  # used by app.py

# LangGraph toggle
LANGGRAPH_ENABLED = os.environ.get("LANGGRAPH_ENABLED", "false").lower() == "true"

# Vector chunking defaults (configurable)
VECTOR_CHUNK_SIZE = int(os.environ.get("VECTOR_CHUNK_SIZE", 1000))
VECTOR_CHUNK_OVERLAP = int(os.environ.get("VECTOR_CHUNK_OVERLAP", 200))


class Config:
    """
    Encapsulate configuration and ensure required environment variables and directories.

    Usage:
        cfg = Config()          # validate required keys (default)
        cfg = Config(False)     # skip required-key check (useful for some tests)

    Validation logic:
      - By default (require_keys=True) we *require* GEMINI_API_KEY or GOOGLE_API_KEY.
      - If USE_LANGSMITH_FOR_GEN=true then generation via LangSmith is explicitly opted-in,
        and LANGSMITH_API_KEY must be present for generation to work. This is allowed but
        discouraged unless you explicitly want LangSmith as a provider.
      - LANGSMITH_API_KEY is otherwise optional (used only for tracing).
    """

    def __init__(self, require_keys: Optional[bool] = True):
        # Validate required keys based on policy and opt-in flags.
        if require_keys:
            # Primary expected path: Gemini (or GOOGLE_API_KEY)
            gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            use_langsmith_for_gen = os.environ.get("USE_LANGSMITH_FOR_GEN", str(USE_LANGSMITH_FOR_GEN)).lower() == "true"
            langsmith_key = os.environ.get("LANGSMITH_API_KEY")

            if not gemini_key:
                # If user explicitly opted to use LangSmith for generation, allow it only if LANGSMITH_API_KEY exists.
                if use_langsmith_for_gen:
                    if not langsmith_key:
                        raise CustomException(
                            "USE_LANGSMITH_FOR_GEN=true but LANGSMITH_API_KEY is missing. "
                            "Provide LANGSMITH_API_KEY or set USE_LANGSMITH_FOR_GEN=false and provide GEMINI_API_KEY/GOOGLE_API_KEY."
                        )
                    else:
                        # Allowed: LangSmith gen opt-in with key present (explicit opt-in)
                        logger.warning(
                            "No GEMINI_API_KEY/GOOGLE_API_KEY found; USE_LANGSMITH_FOR_GEN is true and LANGSMITH_API_KEY is present. "
                            "This means LangSmith will be used for generation — ensure this is intentional."
                        )
                else:
                    # Default safe behavior: require Gemini/Google API key
                    raise CustomException(
                        "Missing required environment variable: GEMINI_API_KEY or GOOGLE_API_KEY. "
                        "Set GEMINI_API_KEY (or set GOOGLE_API_KEY) for generation."
                    )

        # Populate instance attributes from environment (fall back to module-level defaults)
        self.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.GEMINI_ENDPOINT = os.environ.get("GEMINI_ENDPOINT", GEMINI_ENDPOINT)
        self.GEMINI_MODEL = os.environ.get("GEMINI_MODEL", GEMINI_MODEL)

        self.LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", LANGSMITH_API_KEY)
        self.LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", str(LANGSMITH_TRACING)).lower() == "true"
        self.LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT", LANGSMITH_ENDPOINT)
        self.LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", LANGSMITH_PROJECT)

        self.USE_LANGSMITH_FOR_GEN = os.environ.get("USE_LANGSMITH_FOR_GEN", str(USE_LANGSMITH_FOR_GEN)).lower() == "true"

        self.DATABASE_PATH = os.environ.get("DATABASE_PATH", DATABASE_PATH)
        self.UPLOAD_DIR = os.environ.get("UPLOAD_DIR", UPLOAD_DIR)
        self.VECTOR_INDEX_PATH = os.environ.get("VECTOR_INDEX_PATH", VECTOR_INDEX_PATH)
        self.HISTORY_BACKEND = os.environ.get("HISTORY_BACKEND", HISTORY_BACKEND)

        self.LANGGRAPH_ENABLED = os.environ.get("LANGGRAPH_ENABLED", str(LANGGRAPH_ENABLED)).lower() == "true"

        # Vector chunking
        try:
            self.VECTOR_CHUNK_SIZE = int(os.environ.get("VECTOR_CHUNK_SIZE", VECTOR_CHUNK_SIZE))
        except Exception:
            logger.warning("Invalid VECTOR_CHUNK_SIZE, falling back to default %s", VECTOR_CHUNK_SIZE)
            self.VECTOR_CHUNK_SIZE = VECTOR_CHUNK_SIZE

        try:
            self.VECTOR_CHUNK_OVERLAP = int(os.environ.get("VECTOR_CHUNK_OVERLAP", VECTOR_CHUNK_OVERLAP))
        except Exception:
            logger.warning("Invalid VECTOR_CHUNK_OVERLAP, falling back to default %s", VECTOR_CHUNK_OVERLAP)
            self.VECTOR_CHUNK_OVERLAP = VECTOR_CHUNK_OVERLAP

        # Derived directories
        self.DATABASE_DIR = str(Path(self.DATABASE_PATH).parent) if self.DATABASE_PATH else ""
        self.VECTOR_INDEX_DIR = str(Path(self.VECTOR_INDEX_PATH).parent) if self.VECTOR_INDEX_PATH else ""

        # Ensure important directories exist (best-effort). Skip empty strings.
        for dir_path in [self.DATABASE_DIR, self.UPLOAD_DIR, self.VECTOR_INDEX_DIR, LOGS_DIR]:
            if not dir_path:
                continue
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.info("Ensured directory exists: %s", dir_path)
            except Exception as e:
                logger.exception("Failed to ensure directory %s: %s", dir_path, e)
                # Directory creation failures are likely environmental; raise a clear exception
                raise CustomException(f"Failed to create directory {dir_path}: {e}")

        # Log current high-level configuration (do not print secrets)
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
