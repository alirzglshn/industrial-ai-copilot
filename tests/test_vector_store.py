import uuid

from copilot.retrieval.vector_store import QdrantVectorStore
from tests.fakes import HashingEmbedder


def _id(label: str) -> str:
    """Qdrant point ids must be UUIDs or unsigned ints.

    Production satisfies this because chunk ids are UUIDs; tests derive a
    stable UUID from a readable label so assertions stay legible.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, label))


def _index(store: QdrantVectorStore, embedder: HashingEmbedder, rows: list[tuple]) -> None:
    """rows: (chunk_id, document_id, page_number, text)"""
    store.upsert(
        ids=[row[0] for row in rows],
        vectors=embedder.embed_documents([row[3] for row in rows]),
        payloads=[
            {"document_id": row[1], "page_number": row[2], "chunk_index": i, "text": row[3]}
            for i, row in enumerate(rows)
        ],
    )


def test_ensure_collection_is_idempotent(vector_store: QdrantVectorStore) -> None:
    vector_store.ensure_collection()
    vector_store.ensure_collection()
    assert vector_store.client.collection_exists(vector_store.collection_name)


def test_upsert_then_search_returns_the_closest_chunk(
    vector_store: QdrantVectorStore, embedder: HashingEmbedder
) -> None:
    _index(
        vector_store,
        embedder,
        [
            (_id("c1"), "doc-a", 37, "overheating is caused by insufficient cooling airflow"),
            (_id("c2"), "doc-a", 12, "the warranty covers manufacturing defects only"),
        ],
    )

    results = vector_store.search(embedder.embed_query("cooling airflow overheating"), top_k=2)

    assert [r.id for r in results][0] == _id("c1")
    assert results[0].score > results[1].score
    assert results[0].payload["page_number"] == 37


def test_search_respects_top_k(
    vector_store: QdrantVectorStore, embedder: HashingEmbedder
) -> None:
    _index(
        vector_store,
        embedder,
        [(_id(f"c{i}"), "doc-a", i, f"cooling airflow text {i}") for i in range(10)],
    )

    assert len(vector_store.search(embedder.embed_query("cooling"), top_k=3)) == 3


def test_document_filter_isolates_manuals(
    vector_store: QdrantVectorStore, embedder: HashingEmbedder
) -> None:
    _index(
        vector_store,
        embedder,
        [
            (_id("c1"), "doc-a", 1, "cooling airflow specification"),
            (_id("c2"), "doc-b", 1, "cooling airflow specification"),
        ],
    )

    results = vector_store.search(
        embedder.embed_query("cooling airflow"), top_k=5, document_id="doc-b"
    )

    assert [r.payload["document_id"] for r in results] == ["doc-b"]


def test_delete_document_removes_only_that_document(
    vector_store: QdrantVectorStore, embedder: HashingEmbedder
) -> None:
    _index(
        vector_store,
        embedder,
        [
            (_id("c1"), "doc-a", 1, "cooling airflow"),
            (_id("c2"), "doc-b", 1, "cooling airflow"),
        ],
    )

    vector_store.delete_document("doc-a")

    assert vector_store.count() == 1
    remaining = vector_store.search(embedder.embed_query("cooling"), top_k=5)
    assert [r.payload["document_id"] for r in remaining] == ["doc-b"]


def test_upsert_of_nothing_is_a_no_op(vector_store: QdrantVectorStore) -> None:
    vector_store.upsert(ids=[], vectors=[], payloads=[])
    assert vector_store.count() == 0
