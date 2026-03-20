import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import ConfigError, Settings


class TestSettings:
    def test_defaults_loaded(self):
        s = Settings()
        assert s.local_llm_model == os.getenv("LOCAL_LLM_MODEL", "llama3.2").strip()
        assert s.qdrant_collection == os.getenv("QDRANT_COLLECTION", "knowledge_base").strip()

    def test_project_root_points_to_repo(self):
        s = Settings()
        assert (s.project_root / "src").exists()
        assert (s.project_root / "pyproject.toml").exists()

    def test_prompt_path_combines_dir_and_file(self):
        s = Settings()
        assert s.prompt_path == s.prompts_dir / s.active_prompt_file

    def test_data_dir_exists(self):
        s = Settings()
        assert s.data_dir.exists()

    def test_validate_raises_on_missing_qdrant_url(self):
        s = Settings(qdrant_url="", qdrant_api_key="test")
        with pytest.raises(ConfigError, match="QDRANT_URL"):
            s.validate()

    def test_validate_raises_on_missing_qdrant_key(self):
        s = Settings(qdrant_url="http://localhost", qdrant_api_key="")
        with pytest.raises(ConfigError, match="QDRANT_API_KEY"):
            s.validate()
