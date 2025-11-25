# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

from app.logger import get_logger, LOGS_DIR
from app.exception import CustomException

# Initialize logger for config
logger = get_logger("config")

# Load .env
try:
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except Exception as e:
    raise CustomException(e)


# Module-level defaults (also useful if other modules import these directly)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "default-project")

DATABASE_PATH = os.environ.get("DATABASE_PATH", "./data/text_to_sql.db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
VECTOR_INDEX_PATH = os.environ.get("VECTOR_INDEX_PATH", "./faiss/index.faiss")
HISTORY_BACKEND = os.environ.get("HISTORY_BACKEND", "json")  # used by app.py


class Config:
    """
    Encapsulate configuration and ensure required environment variables and directories.
    Instantiate with `cfg = Config()` (app.py already expects this pattern).
    """

    def __init__(self, require_keys: Optional[bool] = True):
        # required keys
        if require_keys:
            try:
                # raise if missing
                self.GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
                self.LANGSMITH_API_KEY = os.environ["LANGSMITH_API_KEY"]
            except KeyError as e:
                raise CustomException(f"Missing required environment variable: {str(e)}")

        # optional / defaults
        self.LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
        self.LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT", LANGSMITH_ENDPOINT)
        self.LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", LANGSMITH_PROJECT)

        self.DATABASE_PATH = os.environ.get("DATABASE_PATH", DATABASE_PATH)
        self.UPLOAD_DIR = os.environ.get("UPLOAD_DIR", UPLOAD_DIR)
        self.VECTOR_INDEX_PATH = os.environ.get("VECTOR_INDEX_PATH", VECTOR_INDEX_PATH)
        self.HISTORY_BACKEND = os.environ.get("HISTORY_BACKEND", HISTORY_BACKEND)

        # derived paths
        self.DATABASE_DIR = str(Path(self.DATABASE_PATH).parent) if self.DATABASE_PATH else ""
        self.VECTOR_INDEX_DIR = str(Path(self.VECTOR_INDEX_PATH).parent) if self.VECTOR_INDEX_PATH else ""

        # ensure directories exist, but guard against empty strings
        for dir_path in [self.DATABASE_DIR, self.UPLOAD_DIR, self.VECTOR_INDEX_DIR, LOGS_DIR]:
            if not dir_path:
                # if path is empty (e.g., database path in current dir), skip creating ''
                continue
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured directory exists: {dir_path}")
            except Exception as e:
                logger.exception(f"Failed to ensure directory {dir_path}: {e}")
                raise CustomException(e)

        logger.info("Configuration (Config) initialized successfully")

    # convenience: allow attribute-style access for keys
    def get(self, name: str, default=None):
        return getattr(self, name, default)



