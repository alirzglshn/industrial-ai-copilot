"""Phase 3: turning text into vectors.

The interface is separate from the sentence-transformers implementation for
two reasons: the rest of the retrieval stack can be tested without loading a
model (or installing torch), and swapping the embedding model stays a config
change rather than a rewrite.
"""

from abc import ABC, abstractmethod
from functools import lru_cache

from copilot.core.config import Settings, get_settings

# BGE models are trained with an asymmetric objective: queries carry an
# instruction prefix, passages do not. Omitting it measurably degrades
# retrieval, which is why embed_query and embed_documents are distinct
# operations rather than one embed() call.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class TextEmbedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector width, needed to size the Qdrant collection."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""


class SentenceTransformerEmbedder(TextEmbedder):
    def __init__(
        self,
        model_name: str,
        query_prefix: str = BGE_QUERY_PREFIX,
        batch_size: int = 32,
    ) -> None:
        # Imported lazily so that importing this module (and therefore the API)
        # does not pull in torch. Ingestion must stay runnable without it.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    @property
    def dimension(self) -> int:
        # Renamed in sentence-transformers 6.0; both spellings are supported so
        # the pinned range in pyproject.toml stays wide.
        getter = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        return int(getter())

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # Normalized vectors turn Qdrant's cosine distance into a plain dot
        # product and keep scores comparable across queries.
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"{self.query_prefix}{text}"])[0]


@lru_cache(maxsize=1)
def get_text_embedder(settings: Settings | None = None) -> TextEmbedder:
    """Process-wide embedder. Cached because loading the model is the expensive part."""
    settings = settings or get_settings()
    return SentenceTransformerEmbedder(settings.text_embedding_model)
