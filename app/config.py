import os
from dotenv import load_dotenv
from app.logger import get_logger, LOGS_DIR
from app.exception import CustomException

# Initialize logger for config
logger = get_logger("config")

# Load .env file
try:
    load_dotenv()  # automatically loads .env from project root
    logger.info("Loaded environment variables from .env file")
except Exception as e:
    raise CustomException(e)

# Required Environment Variables

try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    LANGSMITH_API_KEY = os.environ["LANGSMITH_API_KEY"]
except KeyError as e:
    raise CustomException(f"Missing required environment variable: {str(e)}")


#  Default Variables

LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "default-project")

DATABASE_PATH = os.environ.get("DATABASE_PATH", "./data/text_to_sql.db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
VECTOR_INDEX_PATH = os.environ.get("VECTOR_INDEX_PATH", "./faiss/index.faiss")

# Ensure directories exist
for dir_path in [os.path.dirname(DATABASE_PATH), UPLOAD_DIR, os.path.dirname(VECTOR_INDEX_PATH), LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Ensured directory exists: {dir_path}")

logger.info("Configuration loaded successfully")
