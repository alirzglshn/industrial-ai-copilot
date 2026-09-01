"""end to end retrieval with the real embedding model, proving retrieval is semantic

deselected by default, downloads model weights on first run:
    pytest -m integration
"""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from copilot.ingestion.chunker import TextChunker
from copilot.ingestion.parser import PdfDocumentParser, PdfParserConfig
from copilot.ingestion.service import IngestionService
from copilot.retrieval.embedder import SentenceTransformerEmbedder
from copilot.retrieval.indexer import ChunkIndexer
from copilot.retrieval.retriever import VectorRetriever
from copilot.retrieval.vector_store import QdrantVectorStore

pytestmark = pytest.mark.integration

MODEL = "BAAI/bge-small-en-v1.5"


@pytest.fixture(scope="module")
def real_embedder() -> SentenceTransformerEmbedder:
    # module-scoped, loading the model is by far the slowest part of these tests
    return SentenceTransformerEmbedder(MODEL)


@pytest.fixture
def indexed_manual(
    real_embedder: SentenceTransformerEmbedder,
    db_session: Session,
    manual_pdf_bytes: bytes,
    tmp_path: Path,
):
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client, "integration", real_embedder.dimension)
    store.ensure_collection()

    service = IngestionService(
        parser=PdfDocumentParser(PdfParserConfig(image_dir=tmp_path / "img")),
        chunker=TextChunker(800, 150),
        upload_dir=tmp_path / "up",
    )
    document = service.ingest(db_session, "manual.pdf", manual_pdf_bytes)
    ChunkIndexer(real_embedder, store).index_document(db_session, document.id)

    try:
        yield VectorRetriever(real_embedder, store), document.id
    finally:
        client.close()


def test_model_produces_the_expected_vector_width(
    real_embedder: SentenceTransformerEmbedder,
) -> None:
    assert real_embedder.dimension == 384


def test_paraphrased_question_finds_the_right_page(indexed_manual) -> None:
    """the manual says "insufficient cooling airflow", the question does not"""
    retriever, _ = indexed_manual

    results = retriever.retrieve("Why is the motor running too hot?", top_k=3)

    assert results
    assert results[0].page_number == 1
    assert "airflow" in results[0].text.lower()


def test_question_about_specifications_finds_the_table_page(indexed_manual) -> None:
    retriever, _ = indexed_manual

    results = retriever.retrieve("maximum operating temperature for each variant", top_k=3)

    assert 2 in [r.page_number for r in results]


def test_retrieval_is_semantic_not_lexical(indexed_manual) -> None:
    """no content word here appears in the manual text"""
    retriever, _ = indexed_manual

    results = retriever.retrieve("clogged air intake reducing ventilation", top_k=3)

    assert results
    assert results[0].page_number == 1


def test_unrelated_question_scores_below_manual_content(indexed_manual) -> None:
    """groundwork for insufficient-evidence behaviour: a relative gap, not an absolute floor"""
    retriever, _ = indexed_manual

    on_topic = retriever.retrieve("what causes overheating", top_k=1)
    off_topic = retriever.retrieve("mortgage refinancing interest rates", top_k=1)

    assert on_topic and off_topic
    assert on_topic[0].score > off_topic[0].score + 0.1


def test_document_filter_works_with_real_vectors(indexed_manual) -> None:
    retriever, document_id = indexed_manual

    assert retriever.retrieve("cooling", top_k=5, document_id=document_id)
    assert retriever.retrieve("cooling", top_k=5, document_id="no-such-document") == []
