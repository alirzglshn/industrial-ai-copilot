from sqlalchemy.orm import Session

from copilot.db.models import Chunk, Document
from copilot.retrieval.indexer import ChunkIndexer
from copilot.retrieval.vector_store import QdrantVectorStore
from tests.fakes import HashingEmbedder


def _document(db: Session, document_id: str = "doc-a", texts: list[str] | None = None) -> Document:
    document = Document(id=document_id, filename="manual.pdf", status="parsed", page_count=1)
    db.add(document)
    for index, text in enumerate(texts or []):
        db.add(
            Chunk(
                document_id=document_id,
                page_number=index + 1,
                chunk_index=index,
                text=text,
            )
        )
    db.commit()
    return document


def test_indexes_every_chunk_and_marks_the_document(
    db_session: Session, retrieval_stack
) -> None:
    _document(db_session, texts=["cooling airflow", "bearing lubrication"])

    indexed = retrieval_stack.indexer.index_document(db_session, "doc-a")

    assert indexed == 2
    assert retrieval_stack.store.count() == 2
    assert db_session.get(Document, "doc-a").status == "indexed"


def test_records_embedding_id_on_each_chunk(db_session: Session, retrieval_stack) -> None:
    _document(db_session, texts=["cooling airflow"])

    retrieval_stack.indexer.index_document(db_session, "doc-a")

    chunk = db_session.query(Chunk).one()
    # The point id is the chunk id, so this marks the chunk as live in Qdrant.
    assert chunk.embedding_id == chunk.id


def test_document_without_chunks_indexes_nothing(db_session: Session, retrieval_stack) -> None:
    _document(db_session, texts=[])

    assert retrieval_stack.indexer.index_document(db_session, "doc-a") == 0
    assert retrieval_stack.store.count() == 0


def test_reindexing_replaces_rather_than_accumulates(
    db_session: Session, retrieval_stack
) -> None:
    _document(db_session, texts=["cooling airflow", "bearing lubrication"])
    retrieval_stack.indexer.index_document(db_session, "doc-a")

    retrieval_stack.indexer.index_document(db_session, "doc-a")

    assert retrieval_stack.store.count() == 2


def test_reindexing_after_a_reparse_leaves_no_stale_vectors(
    db_session: Session, retrieval_stack, embedder: HashingEmbedder
) -> None:
    """A shrinking document must not keep answering from text it no longer has."""
    _document(db_session, texts=["cooling airflow", "obsolete paragraph about bearings"])
    retrieval_stack.indexer.index_document(db_session, "doc-a")

    db_session.query(Chunk).filter(Chunk.text.like("obsolete%")).delete(synchronize_session=False)
    db_session.commit()
    retrieval_stack.indexer.index_document(db_session, "doc-a")

    assert retrieval_stack.store.count() == 1
    hits = retrieval_stack.store.search(embedder.embed_query("obsolete bearings"), top_k=5)
    assert all("obsolete" not in (hit.payload.get("text") or "") for hit in hits)


def test_indexes_only_the_requested_document(db_session: Session, retrieval_stack) -> None:
    _document(db_session, "doc-a", ["cooling airflow"])
    _document(db_session, "doc-b", ["bearing lubrication"])

    retrieval_stack.indexer.index_document(db_session, "doc-a")

    assert retrieval_stack.store.count() == 1
    assert db_session.get(Document, "doc-b").status == "parsed"


def test_batches_larger_documents(
    db_session: Session, embedder: HashingEmbedder, vector_store: QdrantVectorStore
) -> None:
    _document(db_session, texts=[f"maintenance step {i}" for i in range(10)])
    indexer = ChunkIndexer(embedder, vector_store, batch_size=3)

    assert indexer.index_document(db_session, "doc-a") == 10
    assert vector_store.count() == 10
