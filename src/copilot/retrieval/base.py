"""embeddings and qdrant similarity search, over text and images"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class EvidenceKind(str, Enum):
    TEXT = "text"
    IMAGE = "image"


@dataclass
class Evidence:
    kind: EvidenceKind
    document_id: str
    page_number: int
    # similarity from the source model, not comparable across modalities
    score: float
    chunk_id: str | None = None
    text: str | None = None
    image_id: str | None = None
    image_path: str | None = None
    # set only when several rankings were fused, see retrieval.multimodal
    fused_score: float | None = None


class Retriever(ABC):
    """top-k evidence, text and or image, for a natural-language query"""

    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 5, document_id: str | None = None
    ) -> list[Evidence]: ...
