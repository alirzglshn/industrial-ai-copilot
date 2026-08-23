from fastapi.testclient import TestClient

from copilot.api.deps import get_optional_retrieval_stack, require_retrieval_stack
from copilot.main import app


def _upload(client: TestClient, content: bytes, filename: str = "manual.pdf"):
    return client.post(
        "/documents/upload", files={"file": (filename, content, "application/pdf")}
    )


def test_upload_indexes_the_document(client: TestClient, manual_pdf_bytes: bytes) -> None:
    body = _upload(client, manual_pdf_bytes).json()

    assert body["indexed_chunks"] == body["chunk_count"]
    assert body["indexed_chunks"] > 0
    assert client.get(f"/documents/{body['id']}").json()["status"] == "indexed"


def test_search_finds_the_page_that_answers_the_question(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    document_id = _upload(client, manual_pdf_bytes).json()["id"]

    response = client.post("/search", json={"query": "cooling airflow overheating", "top_k": 3})

    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    top = results[0]
    assert top["document_id"] == document_id
    assert top["page_number"] == 1
    assert top["kind"] == "text"
    assert "cooling airflow" in top["text"]
    assert top["chunk_id"]


def test_search_respects_top_k(client: TestClient, manual_pdf_bytes: bytes) -> None:
    _upload(client, manual_pdf_bytes)

    results = client.post("/search", json={"query": "pump", "top_k": 2}).json()["results"]

    assert len(results) <= 2


def test_search_can_be_restricted_to_one_manual(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    first = _upload(client, manual_pdf_bytes, "first.pdf").json()["id"]
    _upload(client, manual_pdf_bytes, "second.pdf")

    results = client.post(
        "/search", json={"query": "cooling airflow", "top_k": 10, "document_id": first}
    ).json()["results"]

    assert results
    assert {r["document_id"] for r in results} == {first}


def test_search_scores_are_descending(client: TestClient, manual_pdf_bytes: bytes) -> None:
    _upload(client, manual_pdf_bytes)

    results = client.post("/search", json={"query": "cooling airflow", "top_k": 5}).json()[
        "results"
    ]

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_is_rejected(client: TestClient) -> None:
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_reindexing_is_available_and_idempotent(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    document_id = _upload(client, manual_pdf_bytes).json()["id"]
    chunk_count = client.get(f"/documents/{document_id}/chunks").json()

    first = client.post(f"/documents/{document_id}/index").json()
    second = client.post(f"/documents/{document_id}/index").json()

    assert first["indexed_chunks"] == second["indexed_chunks"] == len(chunk_count)
    assert second["status"] == "indexed"


def test_reindexing_an_unknown_document_is_404(client: TestClient) -> None:
    assert client.post("/documents/nope/index").status_code == 404


def test_search_reports_503_when_retrieval_is_unavailable(client: TestClient) -> None:
    """A missing model or unreachable Qdrant must not read as a generic 500."""

    def unavailable():
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Retrieval unavailable: no model")

    app.dependency_overrides[require_retrieval_stack] = unavailable
    try:
        assert client.post("/search", json={"query": "cooling"}).status_code == 503
    finally:
        app.dependency_overrides.pop(require_retrieval_stack, None)


def test_upload_still_succeeds_when_indexing_is_unavailable(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    """Ingestion must not depend on the embedding model being present."""
    app.dependency_overrides[get_optional_retrieval_stack] = lambda: None
    try:
        response = _upload(client, manual_pdf_bytes)
    finally:
        app.dependency_overrides.pop(get_optional_retrieval_stack, None)

    assert response.status_code == 201
    body = response.json()
    assert body["chunk_count"] > 0
    assert body["indexed_chunks"] == 0
    # Still parsed, so it can be indexed later once retrieval comes back.
    assert body["status"] == "parsed"
