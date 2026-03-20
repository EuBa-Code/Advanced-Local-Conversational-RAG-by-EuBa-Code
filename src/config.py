import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parent.parent

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "").strip()
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "knowledge_base").strip()

    # Google Gemini
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "").strip()
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp").strip()
    gemini_embeddings_model: str = os.getenv("GEMINI_EMBEDDINGS_MODEL", "models/text-embedding-004").strip()

    # OpenRouter
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "google/gemma-3-27b-it:free").strip()

    # Local (Ollama & HuggingFace)
    local_llm_model: str = os.getenv("LOCAL_LLM_MODEL", "llama3.2").strip()
    eval_llm_model: str = os.getenv("EVAL_LLM_MODEL", "llama3.1:8b").strip()
    local_embeddings_model: str = os.getenv("HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    hf_models_cache: str = os.getenv("HF_MODELS_CACHE", "").strip()

    # Paths
    data_dir: Path = project_root / os.getenv("DATA_DIR", "data")
    prompts_dir: Path = project_root / "prompts"
    active_prompt_file: str = os.getenv("PROMPT_FILE", "prompts.txt").strip()

    def validate(self):
        errors = []

        if not self.qdrant_url:
            errors.append("QDRANT_URL not set in .env file")
        if not self.qdrant_api_key:
            errors.append("QDRANT_API_KEY not set in .env file")

        has_any_llm = bool(self.google_api_key or self.openrouter_api_key or self.local_llm_model)
        if not has_any_llm:
            errors.append("No LLM provider configured (Gemini, OpenRouter or Ollama).")

        if not self.data_dir.exists():
            errors.append(f"Data folder does not exist: {self.data_dir}")
        if not self.prompts_dir.exists():
            errors.append(f"Prompt folder does not exist: {self.prompts_dir}")
        elif not self.prompt_path.exists():
            errors.append(f"Prompt file '{self.active_prompt_file}' not found in {self.prompts_dir}")

        if errors:
            raise ConfigError("Invalid Configuration:\n" + "\n".join(f"- {e}" for e in errors))

    @property
    def prompt_path(self) -> Path:
        return self.prompts_dir / self.active_prompt_file

    def setup_environment(self):
        if self.hf_models_cache:
            hf_path = Path(self.hf_models_cache).resolve()
            hf_path.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(hf_path)


def get_settings() -> Settings:
    s = Settings()
    s.validate()
    s.setup_environment()
    return s
