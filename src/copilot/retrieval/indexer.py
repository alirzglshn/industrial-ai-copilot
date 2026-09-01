"""embedding persisted chunks and loading them into qdrant

postgres is the source of truth for chunk text and provenance, qdrant holds
only vectors and the payload needed to resolve a hit back to its page
"""

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from copilot.db.models import Chunk, Document
from copilot.retrieval.embedder import TextEmbedder
from copilot.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

STATUS_INDEXED = "indexed"


class ChunkIndexer:
    def __init__(self, embedder: TextEmbedder, store: QdrantVectorStore, batch_size: int = 64):
        self.embedder = embedder
        self.store = store
        self.batch_size = batch_size

    def index_document(self, db: Session, document_id: str) -> int:
        chunks = list(
            db.scalars(
                select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
            )
        )
        if not chunks:
            return 0

        # clearing old vectors first, or a shrinking document would keep answering from stale text
        self.store.delete_document(document_id)

        indexed = 0
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            vectors = self.embedder.embed_documents([chunk.text for chunk in batch])
            self.store.upsert(
                ids=[chunk.id for chunk in batch],
                vectors=vectors,
                payloads=[
                    {
                        "document_id": chunk.document_id,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    }
                    for chunk in batch
                ],
            )
            for chunk in batch:
                # the point id is the chunk id, recording that the chunk is live in the store
                chunk.embedding_id = chunk.id
            indexed += len(batch)

        document = db.get(Document, document_id)
        if document is not None:
            document.status = STATUS_INDEXED
        db.commit()

        logger.info("Indexed %s chunks for document %s", indexed, document_id)
        return indexed
