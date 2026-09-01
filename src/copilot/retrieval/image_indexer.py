"""embedding extracted images and loading them into the image collection

mirrors chunkindexer: postgres is the source of truth, the qdrant point id is
the image row id, and re-indexing clears the document's points first
"""

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from copilot.db.models import Image
from copilot.retrieval.captioner import ImageCaptioner
from copilot.retrieval.image_embedder import ImageEmbedder
from copilot.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def _mean_normalized(left: list[float], right: list[float]) -> list[float]:
    """midpoint of two unit vectors, renormalized, valid only within clip's shared space"""
    summed = [a + b for a, b in zip(left, right)]
    norm = sum(value * value for value in summed) ** 0.5
    if norm == 0:
        return left
    return [value / norm for value in summed]


class ImageIndexer:
    def __init__(
        self,
        embedder: ImageEmbedder,
        store: QdrantVectorStore,
        captioner: ImageCaptioner | None = None,
        batch_size: int = 16,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.captioner = captioner
        self.batch_size = batch_size

    def index_document(self, db: Session, document_id: str) -> int:
        images = list(
            db.scalars(
                select(Image)
                .where(Image.document_id == document_id)
                .order_by(Image.page_number, Image.image_index)
            )
        )
        if not images:
            return 0

        self.store.delete_document(document_id)

        indexed = 0
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            indexed += self._index_batch(batch)

        db.commit()
        logger.info("Indexed %s images for document %s", indexed, document_id)
        return indexed

    def _index_batch(self, batch: list[Image]) -> int:
        paths = [image.storage_path for image in batch]
        vectors = self.embedder.embed_images(paths)

        captions: list[str | None] = [None] * len(batch)
        if self.captioner is not None:
            captions = self.captioner.caption(paths)
            caption_texts = [caption for caption in captions if caption]
            caption_vectors = (
                self.embedder.embed_texts(caption_texts) if caption_texts else []
            )
            caption_lookup = dict(zip(caption_texts, caption_vectors))
            vectors = [
                _mean_normalized(vector, caption_lookup[caption])
                if vector is not None and caption and caption in caption_lookup
                else vector
                for vector, caption in zip(vectors, captions)
            ]

        ids: list[str] = []
        usable: list[list[float]] = []
        payloads: list[dict] = []
        for image, vector, caption in zip(batch, vectors, captions):
            if vector is None:
                # leaving embedding_id null so the gap stays visible, not silently indexed
                logger.warning("Skipped unreadable image %s (%s)", image.id, image.storage_path)
                continue
            if caption:
                image.caption = caption
            ids.append(image.id)
            usable.append(vector)
            payloads.append(
                {
                    "document_id": image.document_id,
                    "page_number": image.page_number,
                    "image_index": image.image_index,
                    "storage_path": image.storage_path,
                    "caption": image.caption,
                }
            )
            image.embedding_id = image.id

        self.store.upsert(ids=ids, vectors=usable, payloads=payloads)
        return len(ids)
