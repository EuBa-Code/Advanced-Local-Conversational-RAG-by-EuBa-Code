import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

class ConfigError(Exception):
    """
    Custom exception: used to differentiate setup errors 
    (e.g. missing keys) from actual code bugs.
    """
    pass

@dataclass(frozen=True)
class Settings:
    """
    Class for centralized configuration management.
    The frozen=True decorator ensures settings don't change during execution.
    """
    # Dynamically determine the project root to handle absolute paths
    project_root: Path = Path(__file__).resolve().parent.parent

    # --- Qdrant Settings ---
    qdrant_url: str = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "").strip()
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "knowledge_base").strip()

    # --- Google Gemini ---
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "").strip()
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp").strip()
    gemini_embeddings_model: str = os.getenv("GEMINI_EMBEDDINGS_MODEL", "models/text-embedding-004").strip()

    # --- OpenRouter ---
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "google/gemma-3-27b-it:free").strip()

    # --- Local (Ollama & HF) ---
    local_llm_model: str = os.getenv("LOCAL_LLM_MODEL", "llama3.2").strip()
    eval_llm_model: str = os.getenv("EVAL_LLM_MODEL", "llama3.1:8b").strip()
    local_embeddings_model: str = os.getenv("HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip() 
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()

    # Path for HuggingFace cache
    hf_models_cache: str = os.getenv("HF_MODELS_CACHE", "").strip()

    # --- Path Management ---
    data_dir: Path = project_root / os.getenv("DATA_DIR", "data")
    prompts_dir: Path = project_root / "prompts"

    # --- Prompt Selection ---
    active_prompt_file: str = os.getenv("PROMPT_FILE", "prompts.txt").strip() 

    def validate(self):
        """
        SYSTEM CORE: Check that everything is ready before starting.
        """
        errors = []
        
        # 1. Qdrant check
        if not self.qdrant_url:
            errors.append("QDRANT_URL not set in .env file")
        if not self.qdrant_api_key:
            errors.append("QDRANT_API_KEY not set in .env file")
            
        # 2. Provider Check: Relax constraint if Ollama is active
        # But warn if all providers are missing.
        has_gemini = bool(self.google_api_key)
        has_openrouter = bool(self.openrouter_api_key)
        has_ollama = bool(self.local_llm_model) # Assume Ollama is present if model is defined
        
        if not (has_gemini or has_openrouter or has_ollama):
            errors.append("No LLM provider configured (Gemini, OpenRouter or Ollama).")

        # 3. Path Recognition
        if not self.data_dir.exists():
            errors.append(f"Data folder does not exist: {self.data_dir}")
        
        if not self.prompts_dir.exists():
            errors.append(f"Prompt folder does not exist: {self.prompts_dir}")
        elif not self.prompt_path.exists():
            errors.append(f"Prompt file '{self.active_prompt_file}' was not found in {self.prompts_dir}")
            
        if errors:
            error_msg = "\n".join([f"- {err}" for err in errors])
            raise ConfigError(f"Invalid Configuration:\n{error_msg}")

    @property 
    def prompt_path(self) -> Path:
        """Helper to get full path to instructions file (System Prompt)"""
        return self.prompts_dir / self.active_prompt_file

    def setup_environment(self):
        """Configure environment variables for HF cache."""
        if self.hf_models_cache:
            hf_path = Path(self.hf_models_cache).resolve()
            hf_path.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(hf_path)

def get_settings() -> Settings:
    s = Settings()
    
    # Run full validation
    s.validate()
    
    # Apply HF configuration
    s.setup_environment()
    return s