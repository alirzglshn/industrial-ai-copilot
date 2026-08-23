"""Phase 3: the text half of the Retriever interface declared in Phase 1.

Phase 4 adds image retrieval and merges both into one evidence list; the
interface is already shaped for that, which is why Evidence carries a kind.
"""

from copilot.retrieval.base import Evidence, EvidenceKind, Retriever
from copilot.retrieval.embedder import TextEmbedder
from copilot.retrieval.vector_store import QdrantVectorStore


class VectorRetriever(Retriever):
    def __init__(
        self,
        embedder: TextEmbedder,
        store: QdrantVectorStore,
        score_threshold: float = 0.0,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.score_threshold = score_threshold

    def retrieve(
        self, query: str, top_k: int = 5, document_id: str | None = None
    ) -> list[Evidence]:
        if not query.strip():
            return []

        vector = self.embedder.embed_query(query)
        points = self.store.search(vector, top_k=top_k, document_id=document_id)

        evidence: list[Evidence] = []
        for point in points:
            if point.score < self.score_threshold:
                continue
            payload = point.payload
            evidence.append(
                Evidence(
                    kind=EvidenceKind.TEXT,
                    document_id=payload["document_id"],
                    page_number=payload["page_number"],
                    score=point.score,
                    chunk_id=point.id,
                    text=payload.get("text"),
                )
            )
        return evidence
