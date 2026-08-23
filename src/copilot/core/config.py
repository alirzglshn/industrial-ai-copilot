from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "industrial-ai-copilot"
    environment: str = "development"

    database_url: str = "postgresql+psycopg2://copilot:copilot@localhost:5432/copilot"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_text: str = "manual_text_chunks"
    qdrant_collection_images: str = "manual_images"

    # Where uploaded PDFs and images extracted from them are written.
    upload_dir: str = "data/uploads"
    image_dir: str = "data/images"
    max_upload_mb: int = 50

    # Chunking. Chunks never span pages, so a citation always resolves to
    # exactly one page. See copilot.ingestion.chunker.
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Sized for CPU-only inference (no dedicated GPU) — see ARCHITECTURE.md.
    text_embedding_model: str = "BAAI/bge-small-en-v1.5"
    image_embedding_model: str = "openai/clip-vit-base-patch32"
    vlm_model: str = "vikhyatk/moondream2"


@lru_cache
def get_settings() -> Settings:
    return Settings()
