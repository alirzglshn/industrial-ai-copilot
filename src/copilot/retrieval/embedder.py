"""turning text into vectors, interface kept separate from the model backing it"""

from abc import ABC, abstractmethod
from functools import lru_cache

# bge models expect an instruction prefix on queries but not passages
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class TextEmbedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """vector width, needed to size the qdrant collection"""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """embedding passages for indexing"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """embedding a search query"""


class SentenceTransformerEmbedder(TextEmbedder):
    def __init__(
        self,
        model_name: str,
        query_prefix: str = BGE_QUERY_PREFIX,
        batch_size: int = 32,
    ) -> None:
        # imported lazily so importing this module does not pull in torch
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    @property
    def dimension(self) -> int:
        # supporting both spellings since the method was renamed in sentence-transformers 6.0
        getter = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        return int(getter())

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # normalized vectors turn qdrant's cosine distance into a plain dot product
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
def get_text_embedder(model_name: str) -> TextEmbedder:
    """process-wide embedder, cached on the model name since loading it is the expensive part"""
    return SentenceTransformerEmbedder(model_name)
