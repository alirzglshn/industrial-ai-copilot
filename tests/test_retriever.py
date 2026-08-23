import uuid

from copilot.retrieval.base import EvidenceKind
from copilot.retrieval.retriever import VectorRetriever
from copilot.retrieval.vector_store import QdrantVectorStore
from tests.fakes import HashingEmbedder


def _seed(store: QdrantVectorStore, embedder: HashingEmbedder, rows: list[tuple]) -> None:
    """rows: (document_id, page_number, text)"""
    ids = [str(uuid.uuid4()) for _ in rows]
    store.upsert(
        ids=ids,
        vectors=embedder.embed_documents([row[2] for row in rows]),
        payloads=[
            {"document_id": row[0], "page_number": row[1], "chunk_index": i, "text": row[2]}
            for i, row in enumerate(rows)
        ],
    )


def test_returns_evidence_carrying_its_source_page(
    retrieval_stack, embedder: HashingEmbedder
) -> None:
    _seed(
        retrieval_stack.store,
        embedder,
        [
            ("doc-a", 37, "overheating is caused by insufficient cooling airflow"),
            ("doc-a", 4, "the warranty covers manufacturing defects"),
        ],
    )

    results = retrieval_stack.retriever.retrieve("cooling airflow overheating", top_k=2)

    assert results[0].kind is EvidenceKind.TEXT
    assert results[0].page_number == 37
    assert results[0].document_id == "doc-a"
    assert results[0].chunk_id is not None
    assert "cooling airflow" in results[0].text


def test_results_are_ordered_by_score(retrieval_stack, embedder: HashingEmbedder) -> None:
    _seed(
        retrieval_stack.store,
        embedder,
        [
            ("doc-a", 1, "cooling airflow overheating motor fins"),
            ("doc-a", 2, "cooling airflow"),
            ("doc-a", 3, "unrelated warranty text"),
        ],
    )

    results = retrieval_stack.retriever.retrieve("cooling airflow overheating", top_k=3)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_blank_query_returns_nothing_without_hitting_the_store(retrieval_stack) -> None:
    assert retrieval_stack.retriever.retrieve("   ") == []


def test_document_filter_restricts_to_one_manual(
    retrieval_stack, embedder: HashingEmbedder
) -> None:
    _seed(
        retrieval_stack.store,
        embedder,
        [("doc-a", 1, "cooling airflow"), ("doc-b", 1, "cooling airflow")],
    )

    results = retrieval_stack.retriever.retrieve("cooling airflow", top_k=5, document_id="doc-a")

    assert {r.document_id for r in results} == {"doc-a"}


def test_score_threshold_drops_weak_matches(
    vector_store: QdrantVectorStore, embedder: HashingEmbedder
) -> None:
    _seed(
        vector_store,
        embedder,
        [("doc-a", 1, "cooling airflow"), ("doc-a", 2, "entirely different subject matter")],
    )
    strict = VectorRetriever(embedder, vector_store, score_threshold=0.9)

    results = strict.retrieve("cooling airflow", top_k=5)

    assert len(results) == 1
    assert all(r.score >= 0.9 for r in results)


def test_query_text_reaches_the_embedder(retrieval_stack, embedder: HashingEmbedder) -> None:
    retrieval_stack.retriever.retrieve("what causes overheating")

    assert embedder.embedded_queries == ["what causes overheating"]
