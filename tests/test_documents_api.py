from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from copilot.db.models import Chunk, Document, Image
from copilot.ingestion.chunker import TABLE_PREFIX


def _upload(client: TestClient, content: bytes, filename: str = "manual.pdf"):
    return client.post(
        "/documents/upload",
        files={"file": (filename, content, "application/pdf")},
    )


def test_upload_ingests_the_document(client: TestClient, manual_pdf_bytes: bytes) -> None:
    response = _upload(client, manual_pdf_bytes)

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "manual.pdf"
    # Upload also indexes, so a successful ingest ends at "indexed".
    assert body["status"] == "indexed"
    assert body["page_count"] == 2
    assert body["chunk_count"] > 0
    assert body["image_count"] == 1


def test_upload_persists_chunks_tied_to_their_page(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    document_id = _upload(client, manual_pdf_bytes).json()["id"]

    chunks = client.get(f"/documents/{document_id}/chunks").json()
    assert chunks
    assert all(c["document_id"] == document_id for c in chunks)
    assert {c["page_number"] for c in chunks} == {1, 2}
    assert [c["chunk_index"] for c in chunks] == sorted(c["chunk_index"] for c in chunks)
    assert any("insufficient cooling airflow" in c["text"] for c in chunks)


def test_table_content_is_retrievable_as_a_chunk(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    document_id = _upload(client, manual_pdf_bytes).json()["id"]

    chunks = client.get(f"/documents/{document_id}/chunks").json()
    table_chunks = [c for c in chunks if c["text"].startswith(TABLE_PREFIX)]

    assert len(table_chunks) == 1
    assert "Pump B | 95 C | 18 m3/h" in table_chunks[0]["text"]
    assert table_chunks[0]["page_number"] == 2


def test_chunks_can_be_filtered_by_page(client: TestClient, manual_pdf_bytes: bytes) -> None:
    document_id = _upload(client, manual_pdf_bytes).json()["id"]

    page_two = client.get(f"/documents/{document_id}/chunks", params={"page_number": 2}).json()

    assert page_two
    assert all(c["page_number"] == 2 for c in page_two)


def test_images_are_listed_with_a_readable_file(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    document_id = _upload(client, manual_pdf_bytes).json()["id"]

    images = client.get(f"/documents/{document_id}/images").json()

    assert len(images) == 1
    assert images[0]["page_number"] == 1
    assert Path(images[0]["storage_path"]).exists()


def test_uploaded_document_appears_in_listing(client: TestClient, manual_pdf_bytes: bytes) -> None:
    document_id = _upload(client, manual_pdf_bytes).json()["id"]

    listed = client.get("/documents").json()
    assert [d["id"] for d in listed] == [document_id]

    fetched = client.get(f"/documents/{document_id}").json()
    assert fetched["page_count"] == 2
    assert fetched["status"] == "indexed"


def test_non_pdf_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_empty_upload_is_rejected(client: TestClient) -> None:
    assert _upload(client, b"").status_code == 400


def test_oversized_upload_is_rejected(client: TestClient, settings) -> None:
    settings.max_upload_mb = 0
    assert _upload(client, b"%PDF-1.4 padding").status_code == 413


def test_unparseable_pdf_is_reported_and_marked_failed(
    client: TestClient, db_session: Session
) -> None:
    response = _upload(client, b"%PDF-1.4 this is not really a pdf")

    assert response.status_code == 422
    # The document row survives with a failed status so the failure is visible
    # rather than silently disappearing.
    document = db_session.scalars(select(Document)).one()
    assert document.status == "failed"
    assert db_session.scalars(select(Chunk)).all() == []
    assert db_session.scalars(select(Image)).all() == []


def test_unknown_document_returns_404(client: TestClient) -> None:
    assert client.get("/documents/does-not-exist").status_code == 404
    assert client.get("/documents/does-not-exist/chunks").status_code == 404
    assert client.get("/documents/does-not-exist/images").status_code == 404
