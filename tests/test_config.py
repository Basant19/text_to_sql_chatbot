# tests/test_config.py
import unittest
import os
from app.config import Config

class TestConfig(unittest.TestCase):
    def test_config_loads(self):
        cfg = Config(require_keys=False)
        # Check main attributes
        self.assertIsNotNone(cfg.GEMINI_MODEL)
        self.assertIsInstance(cfg.LANGGRAPH_ENABLED, bool)
        self.assertIsInstance(cfg.LANGSMITH_TRACING, bool)
        self.assertIsInstance(cfg.USE_LANGSMITH_FOR_GEN, bool)
        print(cfg)  # Print summary for debug

    def test_directories_exist(self):
        cfg = Config(require_keys=False)
        dirs = [cfg.DATA_DIR, cfg.DATABASE_DIR, cfg.UPLOAD_DIR, cfg.VECTOR_INDEX_DIR]
        for d in dirs:
            self.assertTrue(os.path.exists(d), f"Directory does not exist: {d}")

if __name__ == "__main__":
    unittest.main()
