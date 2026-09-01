from pathlib import Path

import pytest
from PIL import Image as PILImage
from sqlalchemy.orm import Session, sessionmaker

from copilot.db.models import Document, Image
from copilot.retrieval.base import EvidenceKind
from copilot.retrieval.image_indexer import ImageIndexer
from copilot.retrieval.image_retriever import DbPageImageSource, ImageRetriever
from copilot.retrieval.vector_store import QdrantVectorStore
from tests.fakes import HashingImageEmbedder


@pytest.fixture
def png(tmp_path: Path):
    def make(name: str) -> Path:
        path = tmp_path / f"{name}.png"
        PILImage.new("RGB", (80, 80), color=(10, 120, 200)).save(path)
        return path

    return make


def _add_image(
    db: Session,
    path: Path,
    document_id: str = "doc-a",
    page_number: int = 1,
    image_index: int = 0,
    caption: str | None = None,
) -> Image:
    if db.get(Document, document_id) is None:
        db.add(Document(id=document_id, filename="m.pdf", status="parsed", page_count=1))
    row = Image(
        document_id=document_id,
        page_number=page_number,
        image_index=image_index,
        storage_path=str(path),
        caption=caption,
    )
    db.add(row)
    db.commit()
    return row


# --- CLIP-space search ----------------------------------------------------


def test_finds_the_image_matching_the_query(
    db_session: Session,
    image_embedder: HashingImageEmbedder,
    image_store: QdrantVectorStore,
    png,
) -> None:
    impeller, wiring = png("impeller"), png("wiring")
    image_embedder.describe(str(impeller), "exploded view of the impeller assembly")
    image_embedder.describe(str(wiring), "terminal box wiring schematic")
    _add_image(db_session, impeller, page_number=12)
    _add_image(db_session, wiring, page_number=30, image_index=1)
    ImageIndexer(image_embedder, image_store).index_document(db_session, "doc-a")

    results = ImageRetriever(image_embedder, image_store).retrieve(
        "exploded view of the impeller", top_k=2
    )

    assert results[0].kind is EvidenceKind.IMAGE
    assert results[0].page_number == 12
    assert results[0].image_path == str(impeller)
    assert results[0].score > results[1].score


def test_returns_the_caption_as_evidence_text(
    db_session: Session,
    image_embedder: HashingImageEmbedder,
    image_store: QdrantVectorStore,
    png,
) -> None:
    path = png("diagram")
    _add_image(db_session, path, caption="impeller clearance diagram")
    ImageIndexer(image_embedder, image_store).index_document(db_session, "doc-a")

    result = ImageRetriever(image_embedder, image_store).retrieve("diagram", top_k=1)[0]

    assert result.text == "impeller clearance diagram"


def test_blank_query_returns_nothing(
    image_embedder: HashingImageEmbedder, image_store: QdrantVectorStore
) -> None:
    assert ImageRetriever(image_embedder, image_store).retrieve("  ") == []


def test_document_filter_restricts_image_search(
    db_session: Session,
    image_embedder: HashingImageEmbedder,
    image_store: QdrantVectorStore,
    png,
) -> None:
    _add_image(db_session, png("a"), document_id="doc-a")
    _add_image(db_session, png("b"), document_id="doc-b")
    indexer = ImageIndexer(image_embedder, image_store)
    indexer.index_document(db_session, "doc-a")
    indexer.index_document(db_session, "doc-b")

    results = ImageRetriever(image_embedder, image_store).retrieve(
        "diagram", top_k=5, document_id="doc-b"
    )

    assert {r.document_id for r in results} == {"doc-b"}


# --- page context ---------------------------------------------------------


def test_returns_images_on_the_requested_pages(
    db_session: Session, session_factory: sessionmaker, png
) -> None:
    _add_image(db_session, png("p12"), page_number=12)
    _add_image(db_session, png("p30"), page_number=30, image_index=1)

    results = DbPageImageSource(session_factory).for_pages([("doc-a", 12)])

    assert [r.page_number for r in results] == [12]
    assert results[0].kind is EvidenceKind.IMAGE
    assert results[0].image_path.endswith("p12.png")


def test_preserves_the_order_pages_were_asked_for(
    db_session: Session, session_factory: sessionmaker, png
) -> None:
    """Ordering carries the rank of the text hit that surfaced each page."""
    _add_image(db_session, png("p5"), page_number=5)
    _add_image(db_session, png("p9"), page_number=9, image_index=1)

    results = DbPageImageSource(session_factory).for_pages([("doc-a", 9), ("doc-a", 5)])

    assert [r.page_number for r in results] == [9, 5]


def test_works_without_any_embeddings(
    db_session: Session, session_factory: sessionmaker, png
) -> None:
    """The point of reading Postgres: diagrams surface even with no CLIP index."""
    row = _add_image(db_session, png("p1"))
    assert row.embedding_id is None

    results = DbPageImageSource(session_factory).for_pages([("doc-a", 1)])

    assert len(results) == 1
    assert results[0].score == 0.0


def test_does_not_leak_images_from_another_document(
    db_session: Session, session_factory: sessionmaker, png
) -> None:
    """A page number alone is ambiguous; both manuals have a page 1."""
    _add_image(db_session, png("a1"), document_id="doc-a", page_number=1)
    _add_image(db_session, png("b1"), document_id="doc-b", page_number=1)

    results = DbPageImageSource(session_factory).for_pages([("doc-a", 1)])

    assert [r.document_id for r in results] == ["doc-a"]


def test_duplicate_page_requests_are_collapsed(
    db_session: Session, session_factory: sessionmaker, png
) -> None:
    _add_image(db_session, png("p1"), page_number=1)

    results = DbPageImageSource(session_factory).for_pages(
        [("doc-a", 1), ("doc-a", 1), ("doc-a", 1)]
    )

    assert len(results) == 1


def test_no_pages_requested_touches_nothing(session_factory: sessionmaker) -> None:
    assert DbPageImageSource(session_factory).for_pages([]) == []


def test_page_without_images_yields_nothing(
    db_session: Session, session_factory: sessionmaker, png
) -> None:
    _add_image(db_session, png("p1"), page_number=1)

    assert DbPageImageSource(session_factory).for_pages([("doc-a", 44)]) == []
