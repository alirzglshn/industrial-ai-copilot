"""text to image search, and page-context image lookup, fused by the multimodal retriever"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from copilot.db.models import Image
from copilot.retrieval.base import Evidence, EvidenceKind
from copilot.retrieval.image_embedder import ImageEmbedder
from copilot.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class ImageRetriever:
    def __init__(self, embedder: ImageEmbedder, store: QdrantVectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(
        self, query: str, top_k: int = 5, document_id: str | None = None
    ) -> list[Evidence]:
        if not query.strip():
            return []

        vector = self.embedder.embed_query(query)
        points = self.store.search(vector, top_k=top_k, document_id=document_id)

        return [
            Evidence(
                kind=EvidenceKind.IMAGE,
                document_id=point.payload["document_id"],
                page_number=point.payload["page_number"],
                score=point.score,
                image_id=point.id,
                image_path=point.payload.get("storage_path"),
                text=point.payload.get("caption"),
            )
            for point in points
        ]


class PageImageSource(ABC):
    @abstractmethod
    def for_pages(self, pages: Iterable[tuple[str, int]]) -> list[Evidence]:
        """images on the given document id, page number pairs, in that order"""


class DbPageImageSource(PageImageSource):
    """reading images straight from postgres, so this works even when clip embedding was skipped"""

    def __init__(self, session_factory: Callable[[], Session] | sessionmaker) -> None:
        self.session_factory = session_factory

    def for_pages(self, pages: Iterable[tuple[str, int]]) -> list[Evidence]:
        ordered = list(dict.fromkeys(pages))
        if not ordered:
            return []

        session = self.session_factory()
        try:
            document_ids = {document_id for document_id, _ in ordered}
            page_numbers = {page_number for _, page_number in ordered}
            rows = list(
                session.scalars(
                    select(Image)
                    .where(Image.document_id.in_(document_ids))
                    .where(Image.page_number.in_(page_numbers))
                    .order_by(Image.page_number, Image.image_index)
                )
            )
        finally:
            session.close()

        by_page: dict[tuple[str, int], list[Image]] = {}
        for row in rows:
            by_page.setdefault((row.document_id, row.page_number), []).append(row)

        evidence: list[Evidence] = []
        for key in ordered:
            for row in by_page.get(key, []):
                evidence.append(
                    Evidence(
                        kind=EvidenceKind.IMAGE,
                        document_id=row.document_id,
                        page_number=row.page_number,
                        # page context carries no similarity of its own, ordering comes from the text hit
                        score=0.0,
                        image_id=row.id,
                        image_path=row.storage_path,
                        text=row.caption,
                    )
                )
        return evidence
