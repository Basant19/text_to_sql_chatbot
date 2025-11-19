import os
import pytest
from app import config
from app.logger import LOGS_DIR

def test_required_env_variables_loaded():
    """Test that required environment variables are loaded."""
    assert hasattr(config, "GOOGLE_API_KEY")
    assert config.GOOGLE_API_KEY != ""
    assert hasattr(config, "LANGSMITH_API_KEY")
    assert config.LANGSMITH_API_KEY != ""

def test_optional_env_variables_defaults():
    """Test that optional variables have correct defaults."""
    assert isinstance(config.LANGSMITH_TRACING, bool)
    assert config.LANGSMITH_ENDPOINT.startswith("http")
    assert isinstance(config.LANGSMITH_PROJECT, str)
    assert config.DATABASE_PATH.endswith(".db")
    assert os.path.isdir(config.UPLOAD_DIR)
    assert os.path.isdir(os.path.dirname(config.VECTOR_INDEX_PATH))
    assert os.path.isdir(LOGS_DIR)

def test_logger_initialized():
    """Test that logger is initialized and can log messages."""
    logger = config.logger
    try:
        logger.info("Test log message from test_config")
    except Exception as e:
        pytest.fail(f"Logger failed to log message: {e}")
