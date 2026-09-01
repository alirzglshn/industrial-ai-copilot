"""qdrant collection management, upsert, and filtered similarity search"""

import logging
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from copilot.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class ScoredPoint:
    id: str
    score: float
    payload: dict[str, Any]


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection_name: str, dimension: int) -> None:
        self.client = client
        self.collection_name = collection_name
        self.dimension = dimension

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            # cosine, paired with normalized embeddings from the embedder
            vectors_config=models.VectorParams(
                size=self.dimension, distance=models.Distance.COSINE
            ),
        )
        # document_id is the only field ever filtered on, unindexed forces qdrant to scan
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created Qdrant collection %s (dim=%s)", self.collection_name, self.dimension)

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        if not ids:
            return
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(id=point_id, vector=vector, payload=payload)
                for point_id, vector, payload in zip(ids, vectors, payloads)
            ],
        )

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[ScoredPoint]:
        query_filter = None
        if document_id is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            ScoredPoint(id=str(point.id), score=float(point.score), payload=point.payload or {})
            for point in response.points
        ]

    def delete_document(self, document_id: str) -> None:
        """removing a document's points so re-indexing cannot leave stale vectors behind"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
        )

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection_name, exact=True).count)


def build_text_vector_store(dimension: int, settings: Settings | None = None) -> QdrantVectorStore:
    settings = settings or get_settings()
    store = QdrantVectorStore(
        client=QdrantClient(url=settings.qdrant_url),
        collection_name=settings.qdrant_collection_text,
        dimension=dimension,
    )
    store.ensure_collection()
    return store


def build_image_vector_store(
    dimension: int, settings: Settings | None = None
) -> QdrantVectorStore:
    """own collection for images, since mixing clip and text vector spaces breaks similarity"""
    settings = settings or get_settings()
    store = QdrantVectorStore(
        client=QdrantClient(url=settings.qdrant_url),
        collection_name=settings.qdrant_collection_images,
        dimension=dimension,
    )
    store.ensure_collection()
    return store
