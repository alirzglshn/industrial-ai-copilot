"""assembling the retrieval stack, lazily and cached, so the api never builds one itself

text and image retrieval degrade independently, losing clip is smaller than losing search
"""

import logging
from dataclasses import dataclass
from functools import lru_cache

from copilot.core.config import Settings, get_settings
from copilot.db.session import SessionLocal
from copilot.retrieval.captioner import get_image_captioner
from copilot.retrieval.embedder import TextEmbedder, get_text_embedder
from copilot.retrieval.image_embedder import ImageEmbedder, get_image_embedder
from copilot.retrieval.image_indexer import ImageIndexer
from copilot.retrieval.image_retriever import DbPageImageSource, ImageRetriever
from copilot.retrieval.indexer import ChunkIndexer
from copilot.retrieval.multimodal import MultimodalRetriever
from copilot.retrieval.retriever import VectorRetriever
from copilot.retrieval.vector_store import (
    QdrantVectorStore,
    build_image_vector_store,
    build_text_vector_store,
)

logger = logging.getLogger(__name__)


class RetrievalUnavailable(RuntimeError):
    """raised when the text embedding model or the vector store cannot be reached"""


@dataclass
class RetrievalStack:
    embedder: TextEmbedder
    store: QdrantVectorStore
    indexer: ChunkIndexer
    retriever: VectorRetriever
    multimodal: MultimodalRetriever
    # none when clip could not be loaded, text search still works
    image_embedder: ImageEmbedder | None = None
    image_store: QdrantVectorStore | None = None
    image_indexer: ImageIndexer | None = None
    image_retriever: ImageRetriever | None = None

    @property
    def images_enabled(self) -> bool:
        return self.image_indexer is not None


def _build_image_side(settings: Settings):
    """the image half of the stack, or all-none if clip is unavailable"""
    try:
        image_embedder = get_image_embedder(settings.image_embedding_model)
        image_store = build_image_vector_store(image_embedder.dimension, settings)
    except Exception as error:
        logger.warning("Image retrieval unavailable, continuing with text only: %s", error)
        return None, None, None, None

    captioner = None
    if settings.enable_image_captioning:
        try:
            captioner = get_image_captioner(
                settings.caption_model,
                max_new_tokens=settings.caption_max_new_tokens,
                prompt=settings.caption_prompt or None,
            )
        except Exception as error:
            # captioning is an enhancement, without it images are still embedded and searchable
            logger.warning("Captioning enabled but unavailable: %s", error)

    return (
        image_embedder,
        image_store,
        ImageIndexer(image_embedder, image_store, captioner=captioner),
        ImageRetriever(image_embedder, image_store),
    )


@lru_cache(maxsize=1)
def get_retrieval_stack() -> RetrievalStack:
    settings = get_settings()
    try:
        embedder = get_text_embedder(settings.text_embedding_model)
        store = build_text_vector_store(embedder.dimension, settings)
    except Exception as error:  # missing torch, model download failure, no Qdrant
        logger.warning("Retrieval stack unavailable: %s", error)
        raise RetrievalUnavailable(str(error)) from error

    image_embedder, image_store, image_indexer, image_retriever = _build_image_side(settings)
    text_retriever = VectorRetriever(embedder, store)

    return RetrievalStack(
        embedder=embedder,
        store=store,
        indexer=ChunkIndexer(embedder, store, batch_size=settings.embed_batch_size),
        retriever=text_retriever,
        multimodal=MultimodalRetriever(
            text_retriever=text_retriever,
            image_retriever=image_retriever,
            page_images=DbPageImageSource(SessionLocal) if settings.page_context_images else None,
            rrf_k=settings.rrf_k,
            image_top_k=settings.image_search_top_k,
        ),
        image_embedder=image_embedder,
        image_store=image_store,
        image_indexer=image_indexer,
        image_retriever=image_retriever,
    )
