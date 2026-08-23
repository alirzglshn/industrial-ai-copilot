"""Assembly of the retrieval stack, kept in one place so the API layer never
constructs an embedder or a Qdrant client itself.

Everything is built lazily and cached: loading the embedding model is the
expensive step, and the API must still start when torch is not installed or
Qdrant is not reachable.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache

from copilot.core.config import get_settings
from copilot.retrieval.embedder import TextEmbedder, get_text_embedder
from copilot.retrieval.indexer import ChunkIndexer
from copilot.retrieval.retriever import VectorRetriever
from copilot.retrieval.vector_store import QdrantVectorStore, build_text_vector_store

logger = logging.getLogger(__name__)


class RetrievalUnavailable(RuntimeError):
    """Raised when the embedding model or the vector store cannot be reached."""


@dataclass
class RetrievalStack:
    embedder: TextEmbedder
    store: QdrantVectorStore
    indexer: ChunkIndexer
    retriever: VectorRetriever


@lru_cache(maxsize=1)
def get_retrieval_stack() -> RetrievalStack:
    settings = get_settings()
    try:
        embedder = get_text_embedder(settings)
        store = build_text_vector_store(embedder.dimension, settings)
    except Exception as error:  # missing torch, model download failure, no Qdrant
        logger.warning("Retrieval stack unavailable: %s", error)
        raise RetrievalUnavailable(str(error)) from error

    return RetrievalStack(
        embedder=embedder,
        store=store,
        indexer=ChunkIndexer(embedder, store),
        retriever=VectorRetriever(embedder, store),
    )
