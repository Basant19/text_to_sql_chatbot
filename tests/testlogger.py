import os
import logging
from app.logger import get_logger, LOGS_DIR

def test_get_logger_returns_logger_instance():
    lg = get_logger("test_logger")
    assert isinstance(lg, logging.Logger)
    # ensure we can call logging methods without error
    lg.info("logger test message")

def test_logs_directory_and_file_created():
    # call get_logger to ensure basic configuration executed
    _ = get_logger("test_logger_2")
    assert os.path.isdir(LOGS_DIR)
    files = [f for f in os.listdir(LOGS_DIR) if f.endswith(".log")]
    assert len(files) >= 1
