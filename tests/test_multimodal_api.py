"""end to end: upload a manual, then find its diagram"""

from pathlib import Path

from fastapi.testclient import TestClient

from copilot.api.deps import get_optional_retrieval_stack
from copilot.main import app
from copilot.retrieval.deps import RetrievalStack


def _upload(client: TestClient, content: bytes, filename: str = "manual.pdf"):
    return client.post(
        "/documents/upload", files={"file": (filename, content, "application/pdf")}
    )


def test_upload_indexes_images_as_well_as_chunks(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    body = _upload(client, manual_pdf_bytes).json()

    assert body["indexed_chunks"] > 0
    # the fixture manual carries one 200x200 diagram on page 1
    assert body["indexed_images"] == 1
    assert body["image_count"] == 1


def test_search_returns_the_diagram_alongside_the_text(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    document_id = _upload(client, manual_pdf_bytes).json()["id"]

    results = client.post(
        "/search", json={"query": "cooling airflow overheating", "top_k": 10}
    ).json()["results"]

    kinds = {r["kind"] for r in results}
    assert kinds == {"text", "image"}
    image = next(r for r in results if r["kind"] == "image")
    assert image["document_id"] == document_id
    assert image["page_number"] == 1
    assert image["image_id"]
    assert Path(image["image_path"]).exists()


def test_the_diagram_comes_from_the_page_the_text_matched(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    """the overheating discussion is on page 1, and so is the diagram"""
    _upload(client, manual_pdf_bytes)

    results = client.post(
        "/search", json={"query": "insufficient cooling airflow", "top_k": 10}
    ).json()["results"]

    text_pages = {r["page_number"] for r in results if r["kind"] == "text"}
    image_pages = {r["page_number"] for r in results if r["kind"] == "image"}
    assert 1 in text_pages
    assert image_pages <= text_pages


def test_images_can_be_excluded(client: TestClient, manual_pdf_bytes: bytes) -> None:
    _upload(client, manual_pdf_bytes)

    results = client.post(
        "/search",
        json={"query": "cooling airflow", "top_k": 10, "include_images": False},
    ).json()["results"]

    assert results
    assert {r["kind"] for r in results} == {"text"}


def test_image_results_are_scoped_to_the_requested_document(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    first = _upload(client, manual_pdf_bytes, "first.pdf").json()["id"]
    _upload(client, manual_pdf_bytes, "second.pdf")

    results = client.post(
        "/search", json={"query": "cooling airflow", "top_k": 10, "document_id": first}
    ).json()["results"]

    assert results
    assert {r["document_id"] for r in results} == {first}


def test_reindex_reports_both_counts(client: TestClient, manual_pdf_bytes: bytes) -> None:
    document_id = _upload(client, manual_pdf_bytes).json()["id"]

    body = client.post(f"/documents/{document_id}/index").json()

    assert body["indexed_chunks"] > 0
    assert body["indexed_images"] == 1
    assert body["status"] == "indexed"


def test_search_still_works_when_the_image_side_is_absent(
    client: TestClient, manual_pdf_bytes: bytes, retrieval_stack: RetrievalStack
) -> None:
    """losing clip must cost diagrams, not search"""
    _upload(client, manual_pdf_bytes)

    text_only = RetrievalStack(
        embedder=retrieval_stack.embedder,
        store=retrieval_stack.store,
        indexer=retrieval_stack.indexer,
        retriever=retrieval_stack.retriever,
        multimodal=retrieval_stack.multimodal,
    )
    text_only.multimodal.image_retriever = None
    app.dependency_overrides[get_optional_retrieval_stack] = lambda: text_only
    try:
        results = client.post("/search", json={"query": "cooling airflow", "top_k": 5}).json()[
            "results"
        ]
    finally:
        app.dependency_overrides.pop(get_optional_retrieval_stack, None)
        text_only.multimodal.image_retriever = retrieval_stack.image_retriever

    assert results
    assert any(r["kind"] == "text" for r in results)


def test_upload_succeeds_when_only_the_image_model_is_missing(
    client: TestClient, manual_pdf_bytes: bytes, retrieval_stack: RetrievalStack
) -> None:
    """text indexing has already succeeded, a missing clip must not undo it"""
    no_images = RetrievalStack(
        embedder=retrieval_stack.embedder,
        store=retrieval_stack.store,
        indexer=retrieval_stack.indexer,
        retriever=retrieval_stack.retriever,
        multimodal=retrieval_stack.multimodal,
        image_indexer=None,
    )
    app.dependency_overrides[get_optional_retrieval_stack] = lambda: no_images
    try:
        body = _upload(client, manual_pdf_bytes).json()
    finally:
        app.dependency_overrides.pop(get_optional_retrieval_stack, None)

    assert body["indexed_chunks"] > 0
    assert body["indexed_images"] == 0
    assert body["status"] == "indexed"
