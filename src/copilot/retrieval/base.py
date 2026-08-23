"""Phase 3-4: embeddings + Qdrant similarity search, over text and images.

Implemented in Phase 3 (text) and extended in Phase 4 (images, combined
evidence). Defined now so copilot.agent and copilot.api.routes.query can be
written against a stable contract.
"""

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
    score: float
    chunk_id: str | None = None
    text: str | None = None
    image_id: str | None = None
    image_path: str | None = None


class Retriever(ABC):
    """Retrieves top-k evidence (text and/or image) for a natural-language query."""

    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 5, document_id: str | None = None
    ) -> list[Evidence]: ...
