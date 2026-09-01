"""end to end: ask a question about an ingested manual"""

from fastapi.testclient import TestClient

from copilot.api.deps import require_answer_generator, require_retrieval_stack
from copilot.generation.prompt import INSUFFICIENT_MARKER
from copilot.main import app
from tests.fakes import ScriptedAnswerGenerator


def _upload(client: TestClient, content: bytes, filename: str = "manual.pdf"):
    return client.post(
        "/documents/upload", files={"file": (filename, content, "application/pdf")}
    )


def _ask(client: TestClient, question: str, **extra):
    return client.post("/query", json={"question": question, **extra})


def test_answers_with_citations(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    document_id = _upload(client, manual_pdf_bytes).json()["id"]
    answer_generator.output = "Caused by insufficient cooling airflow [page 1]."

    body = _ask(client, "why does the pump overheat?").json()

    assert body["answer"] == "Caused by insufficient cooling airflow [page 1]."
    assert body["insufficient_evidence"] is False
    assert body["citations"]
    assert all(c["document_id"] == document_id for c in body["citations"])
    assert all(c["page_number"] == 1 for c in body["citations"])
    assert body["unsupported_pages"] == []


def test_the_question_and_retrieved_evidence_reach_the_generator(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    _upload(client, manual_pdf_bytes)

    _ask(client, "why does the pump overheat?")

    question, evidence = answer_generator.calls[-1]
    assert question == "why does the pump overheat?"
    assert evidence, "the generator must be given the retrieved evidence"


def test_refusal_is_reported_as_insufficient_evidence(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    _upload(client, manual_pdf_bytes)
    answer_generator.output = INSUFFICIENT_MARKER

    body = _ask(client, "what is the tax rate on dividends?").json()

    assert body["insufficient_evidence"] is True
    assert body["citations"] == []


def test_an_uncited_answer_is_flagged_as_ungrounded(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    """a claim backed by nothing must not be returned looking clean"""
    _upload(client, manual_pdf_bytes)
    answer_generator.output = "France has no capital city."

    body = _ask(client, "what is the capital of France?").json()

    assert body["grounded"] is False
    assert body["citations"] == []
    # not a refusal either, the model did make a claim
    assert body["insufficient_evidence"] is False


def test_a_cited_answer_is_reported_as_grounded(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    _upload(client, manual_pdf_bytes)
    answer_generator.output = "Caused by insufficient cooling airflow [page 1]."

    body = _ask(client, "why does it overheat?").json()

    assert body["grounded"] is True
    assert body["faithfulness"] > 0.5


def test_a_real_page_that_does_not_say_what_the_answer_claims_is_ungrounded(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    """the failure this exists to catch: a valid citation wearing a pretraining claim"""
    _upload(client, manual_pdf_bytes)
    answer_generator.output = "The capital city of France is Paris [page 1]."

    body = _ask(client, "what is the capital of France?").json()

    assert body["grounded"] is False
    assert body["faithfulness"] < 0.5
    # resolution still succeeds, page 1 is real, which is exactly why it alone cannot catch this
    assert body["citations"]


def test_a_refusal_is_reported_as_grounded(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    _upload(client, manual_pdf_bytes)
    answer_generator.output = INSUFFICIENT_MARKER

    body = _ask(client, "what is the tax rate?").json()

    assert body["insufficient_evidence"] is True
    assert body["grounded"] is True


def test_an_invented_citation_is_surfaced_not_hidden(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    """the caller must be able to see that a source was fabricated"""
    _upload(client, manual_pdf_bytes)
    answer_generator.output = "The limit is 90 C [page 99]."

    body = _ask(client, "what is the temperature limit?").json()

    assert body["unsupported_pages"] == [99]
    assert body["citations"] == []


def test_question_about_an_unindexed_corpus_refuses(
    client: TestClient, answer_generator: ScriptedAnswerGenerator
) -> None:
    """nothing uploaded means nothing retrieved, so the model is never asked"""
    body = _ask(client, "why does the pump overheat?").json()

    assert body["insufficient_evidence"] is True
    assert body["citations"] == []
    assert answer_generator.calls == [] or answer_generator.calls[-1][1] == []


def test_citations_identify_their_kind(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    """page 1 of the fixture carries both a passage and a diagram"""
    _upload(client, manual_pdf_bytes)
    answer_generator.output = "See the diagram [page 1]."

    citations = _ask(client, "show me the diagram").json()["citations"]

    kinds = {c["kind"] for c in citations}
    assert "text" in kinds or "image" in kinds
    for citation in citations:
        if citation["kind"] == "image":
            assert citation["image_id"]
            assert citation["image_path"]


def test_search_can_be_scoped_to_one_manual(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    first = _upload(client, manual_pdf_bytes, "first.pdf").json()["id"]
    _upload(client, manual_pdf_bytes, "second.pdf")

    _ask(client, "why does it overheat?", document_id=first)

    _, evidence = answer_generator.calls[-1]
    assert {e.document_id for e in evidence} == {first}


def test_images_can_be_excluded_from_evidence(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    _upload(client, manual_pdf_bytes)

    _ask(client, "why does it overheat?", include_images=False)

    _, evidence = answer_generator.calls[-1]
    assert all(e.kind.value == "text" for e in evidence)


def test_empty_question_is_rejected(client: TestClient) -> None:
    assert _ask(client, "").status_code == 422


def test_missing_answer_model_is_a_503(client: TestClient, manual_pdf_bytes: bytes) -> None:
    _upload(client, manual_pdf_bytes)

    def unavailable():
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Answer generation unavailable")

    app.dependency_overrides[require_answer_generator] = unavailable
    try:
        assert _ask(client, "why does it overheat?").status_code == 503
    finally:
        app.dependency_overrides.pop(require_answer_generator, None)


def test_search_still_works_when_answering_is_down(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    """retrieval is useful on its own, a missing answer model must not take it down"""
    _upload(client, manual_pdf_bytes)

    def unavailable():
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Answer generation unavailable")

    app.dependency_overrides[require_answer_generator] = unavailable
    try:
        response = client.post("/search", json={"query": "cooling airflow", "top_k": 3})
    finally:
        app.dependency_overrides.pop(require_answer_generator, None)

    assert response.status_code == 200
    assert response.json()["results"]


def test_retrieval_being_down_is_a_503(client: TestClient) -> None:
    def unavailable():
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Retrieval unavailable")

    app.dependency_overrides[require_retrieval_stack] = unavailable
    try:
        assert _ask(client, "why does it overheat?").status_code == 503
    finally:
        app.dependency_overrides.pop(require_retrieval_stack, None)
