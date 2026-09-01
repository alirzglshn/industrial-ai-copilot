from pathlib import Path

import pytest
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from copilot.db.models import Document, Image
from copilot.retrieval.image_indexer import ImageIndexer, _mean_normalized
from copilot.retrieval.vector_store import QdrantVectorStore
from tests.fakes import HashingImageEmbedder, StubCaptioner


@pytest.fixture
def image_files(tmp_path: Path):
    """Writes real PNGs so the embedder reads actual files, as it does live."""

    def make(count: int) -> list[Path]:
        paths = []
        for index in range(count):
            path = tmp_path / f"diagram{index}.png"
            PILImage.new("RGB", (80, 80), color=(index * 20 % 255, 100, 150)).save(path)
            paths.append(path)
        return paths

    return make


def _document(
    db: Session, paths: list[Path], document_id: str = "doc-a", start_page: int = 1
) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        document = Document(id=document_id, filename="manual.pdf", status="parsed", page_count=1)
        db.add(document)
    for index, path in enumerate(paths):
        db.add(
            Image(
                document_id=document_id,
                page_number=start_page + index,
                image_index=0,
                storage_path=str(path),
            )
        )
    db.commit()
    return document


def test_indexes_every_readable_image(
    db_session: Session,
    image_embedder: HashingImageEmbedder,
    image_store: QdrantVectorStore,
    image_files,
) -> None:
    _document(db_session, image_files(3))
    indexer = ImageIndexer(image_embedder, image_store)

    assert indexer.index_document(db_session, "doc-a") == 3
    assert image_store.count() == 3


def test_records_embedding_id_and_payload(
    db_session: Session,
    image_embedder: HashingImageEmbedder,
    image_store: QdrantVectorStore,
    image_files,
) -> None:
    paths = image_files(1)
    _document(db_session, paths, start_page=7)
    ImageIndexer(image_embedder, image_store).index_document(db_session, "doc-a")

    row = db_session.query(Image).one()
    assert row.embedding_id == row.id

    hit = image_store.search(image_embedder.embed_query("anything"), top_k=1)[0]
    assert hit.id == row.id
    assert hit.payload["page_number"] == 7
    assert hit.payload["storage_path"] == str(paths[0])


def test_missing_file_is_skipped_without_failing_the_document(
    db_session: Session,
    image_embedder: HashingImageEmbedder,
    image_store: QdrantVectorStore,
    image_files,
    tmp_path: Path,
) -> None:
    paths = image_files(2)
    _document(db_session, [*paths, tmp_path / "does_not_exist.png"])

    indexed = ImageIndexer(image_embedder, image_store).index_document(db_session, "doc-a")

    assert indexed == 2
    assert image_store.count() == 2
    # The gap stays visible rather than being recorded as indexed.
    missing = db_session.query(Image).filter(Image.storage_path.like("%does_not_exist%")).one()
    assert missing.embedding_id is None


def test_document_without_images_indexes_nothing(
    db_session: Session, image_embedder: HashingImageEmbedder, image_store: QdrantVectorStore
) -> None:
    _document(db_session, [])

    assert ImageIndexer(image_embedder, image_store).index_document(db_session, "doc-a") == 0


def test_reindexing_replaces_rather_than_accumulates(
    db_session: Session,
    image_embedder: HashingImageEmbedder,
    image_store: QdrantVectorStore,
    image_files,
) -> None:
    _document(db_session, image_files(2))
    indexer = ImageIndexer(image_embedder, image_store)
    indexer.index_document(db_session, "doc-a")

    indexer.index_document(db_session, "doc-a")

    assert image_store.count() == 2


def test_indexes_only_the_requested_document(
    db_session: Session,
    image_embedder: HashingImageEmbedder,
    image_store: QdrantVectorStore,
    image_files,
) -> None:
    files = image_files(2)
    _document(db_session, [files[0]], document_id="doc-a")
    _document(db_session, [files[1]], document_id="doc-b")

    ImageIndexer(image_embedder, image_store).index_document(db_session, "doc-a")

    assert image_store.count() == 1


def test_batches_larger_documents(
    db_session: Session,
    image_embedder: HashingImageEmbedder,
    image_store: QdrantVectorStore,
    image_files,
) -> None:
    _document(db_session, image_files(7))

    indexer = ImageIndexer(image_embedder, image_store, batch_size=2)

    assert indexer.index_document(db_session, "doc-a") == 7
    assert image_store.count() == 7


# --- captioning -----------------------------------------------------------


def test_captions_are_stored_and_indexed_when_a_captioner_is_supplied(
    db_session: Session,
    image_embedder: HashingImageEmbedder,
    image_store: QdrantVectorStore,
    image_files,
) -> None:
    paths = image_files(1)
    _document(db_session, paths)
    captioner = StubCaptioner({str(paths[0]): "exploded view of the impeller"})

    ImageIndexer(image_embedder, image_store, captioner=captioner).index_document(
        db_session, "doc-a"
    )

    assert db_session.query(Image).one().caption == "exploded view of the impeller"
    hit = image_store.search(image_embedder.embed_query("impeller"), top_k=1)[0]
    assert hit.payload["caption"] == "exploded view of the impeller"


def test_no_captioner_leaves_captions_null(
    db_session: Session,
    image_embedder: HashingImageEmbedder,
    image_store: QdrantVectorStore,
    image_files,
) -> None:
    _document(db_session, image_files(1))

    ImageIndexer(image_embedder, image_store).index_document(db_session, "doc-a")

    assert db_session.query(Image).one().caption is None


def test_caption_pulls_the_image_vector_toward_its_words(
    db_session: Session,
    image_embedder: HashingImageEmbedder,
    image_store: QdrantVectorStore,
    image_files,
) -> None:
    """The point of captioning: a query phrased like the caption matches better."""
    paths = image_files(1)
    image_embedder.describe(str(paths[0]), "unrelated visual content")
    _document(db_session, paths)
    captioner = StubCaptioner({str(paths[0]): "impeller clearance diagram"})

    ImageIndexer(image_embedder, image_store, captioner=captioner).index_document(
        db_session, "doc-a"
    )

    captioned_score = image_store.search(
        image_embedder.embed_query("impeller clearance diagram"), top_k=1
    )[0].score
    assert captioned_score > 0.3


def test_failed_caption_still_indexes_the_image(
    db_session: Session,
    image_embedder: HashingImageEmbedder,
    image_store: QdrantVectorStore,
    image_files,
) -> None:
    paths = image_files(1)
    _document(db_session, paths)
    captioner = StubCaptioner({str(paths[0]): None})

    indexed = ImageIndexer(image_embedder, image_store, captioner=captioner).index_document(
        db_session, "doc-a"
    )

    assert indexed == 1
    assert db_session.query(Image).one().caption is None


def test_mean_normalized_returns_a_unit_vector() -> None:
    result = _mean_normalized([1.0, 0.0], [0.0, 1.0])

    assert pytest.approx(sum(v * v for v in result), rel=1e-6) == 1.0
    assert result[0] == pytest.approx(result[1])


def test_mean_normalized_handles_opposing_vectors() -> None:
    """Cancelling to zero must not produce NaNs; fall back to the image vector."""
    assert _mean_normalized([1.0, 0.0], [-1.0, 0.0]) == [1.0, 0.0]
