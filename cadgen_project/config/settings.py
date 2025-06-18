# ===============================
# 2. cadgen_project/config/settings.py
# ===============================

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings with validation."""

    # Application
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    timeout_seconds: int = Field(default=30, env="TIMEOUT_SECONDS")

    # LLM Configuration
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field(default="llama3-8b-8192", env="GROQ_MODEL")

    # Database Configuration
    database_url: str = Field(default="sqlite:///./cadgen.db", env="DATABASE_URL")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")

    # RAG Configuration
    chroma_persist_directory: str = Field(
        default="./storage/chroma_db", env="CHROMA_PERSIST_DIRECTORY"
    )
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    # Directory Paths
    models_dir: str = Field(default="./storage/models", env="MODELS_DIR")
    cache_dir: str = Field(default="./storage/cache", env="CACHE_DIR")
    docs_dir: str = Field(default="./storage/docs", env="DOCS_DIR")

    # UI Configuration
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    streamlit_host: str = Field(default="localhost", env="STREAMLIT_HOST")

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        dirs_to_create = [
            self.models_dir,
            self.cache_dir,
            self.docs_dir,
            self.chroma_persist_directory,
        ]

        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get_database_path(self) -> Path:
        """Get database file path."""
        if self.database_url.startswith("sqlite:///"):
            db_path = self.database_url.replace("sqlite:///", "")
            return Path(db_path)
        return None

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
