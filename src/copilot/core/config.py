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

    # uploaded pdfs and images extracted from them
    upload_dir: str = "data/uploads"
    image_dir: str = "data/images"
    max_upload_mb: int = 50
    # rendered source-page images, cached on first render
    preview_dir: str = "data/previews"

    # chunks never span pages, so a citation resolves to exactly one page
    chunk_size: int = 800
    chunk_overlap: int = 150

    # indexing on upload, skipped rather than failing when the model or qdrant is down
    auto_index_on_upload: bool = True
    search_top_k: int = 5
    embed_batch_size: int = 64

    # text and image scores are fused by rank, not compared directly
    include_images_in_search: bool = True
    image_search_top_k: int = 5
    page_context_images: bool = True
    rrf_k: int = 60

    # off by default, needs a vision model and seconds per image on cpu
    enable_image_captioning: bool = False
    caption_model: str = "Salesforce/blip-image-captioning-base"
    caption_max_new_tokens: int = 40
    # empty for a plain captioner, set for an instruction-following one
    caption_prompt: str = ""

    # a vlm is too slow for interactive use on cpu, so text is the default
    answer_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_vlm_for_answers: bool = False
    answer_max_new_tokens: int = 300
    answer_max_images: int = 3
    # small models lose the thread over long context
    answer_top_k: int = 5

    # single-shot planning, not iterative, given the model and hardware
    agent_max_steps: int = 4
    agent_planner_max_new_tokens: int = 220

    # sized for cpu-only inference, no dedicated gpu
    text_embedding_model: str = "BAAI/bge-small-en-v1.5"
    image_embedding_model: str = "openai/clip-vit-base-patch32"
    vlm_model: str = "vikhyatk/moondream2"


@lru_cache
def get_settings() -> Settings:
    return Settings()
