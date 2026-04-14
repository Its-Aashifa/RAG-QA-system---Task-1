"""
Configuration management using Pydantic BaseSettings.
All settings are read from environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # LLM
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "llama3-70b-8192"

    # Embedding
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # local, no API key needed

    # Chunking
    CHUNK_SIZE: int = 512        # tokens/chars — see design_decisions.md for rationale
    CHUNK_OVERLAP: int = 64      # ~12.5% overlap to preserve context across boundaries

    # Storage
    UPLOAD_DIR: str = "uploads"
    FAISS_INDEX_PATH: str = "faiss_store/index"

    # Rate limiting
    RATE_LIMIT_UPLOAD: str = "10/minute"
    RATE_LIMIT_QUERY: str = "30/minute"

    # Retrieval
    TOP_K_CHUNKS: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure directories exist
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path("faiss_store").mkdir(parents=True, exist_ok=True)
