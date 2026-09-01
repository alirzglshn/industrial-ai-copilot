"""q&a history: /query and /agent/query persist it, /conversations browses it"""

from fastapi.testclient import TestClient

from tests.fakes import ScriptedAnswerGenerator


def _upload(client: TestClient, content: bytes, filename: str = "manual.pdf"):
    return client.post("/documents/upload", files={"file": (filename, content, "application/pdf")})


def test_query_creates_a_conversation_and_returns_its_id(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    _upload(client, manual_pdf_bytes)

    body = client.post("/query", json={"question": "why does it overheat?"}).json()

    assert body["conversation_id"]


def test_the_conversation_is_browsable_with_both_turns(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    _upload(client, manual_pdf_bytes)
    answer_generator.output = "Caused by low airflow [page 1]."

    conversation_id = client.post("/query", json={"question": "why does it overheat?"}).json()[
        "conversation_id"
    ]

    detail = client.get(f"/conversations/{conversation_id}").json()
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][0]["text"] == "why does it overheat?"
    assert detail["messages"][1]["role"] == "assistant"
    assert detail["messages"][1]["text"] == "Caused by low airflow [page 1]."
    assert detail["messages"][1]["pipeline"] == "fixed"
    assert detail["messages"][1]["grounded"] is True


def test_a_second_question_with_the_same_conversation_id_appends_to_it(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    _upload(client, manual_pdf_bytes)
    first = client.post("/query", json={"question": "why does it overheat?"})
    conversation_id = first.json()["conversation_id"]

    second = client.post(
        "/query", json={"question": "how do I vent it?", "conversation_id": conversation_id}
    )

    assert second.json()["conversation_id"] == conversation_id
    detail = client.get(f"/conversations/{conversation_id}").json()
    assert len(detail["messages"]) == 4


def test_omitting_conversation_id_starts_a_new_one(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    _upload(client, manual_pdf_bytes)

    first = client.post("/query", json={"question": "why does it overheat?"}).json()["conversation_id"]
    second = client.post("/query", json={"question": "why does it overheat?"}).json()["conversation_id"]

    assert first != second


def test_conversation_title_is_the_first_question(client: TestClient, manual_pdf_bytes: bytes) -> None:
    _upload(client, manual_pdf_bytes)

    conversation_id = client.post("/query", json={"question": "why does it overheat?"}).json()[
        "conversation_id"
    ]

    summary = next(c for c in client.get("/conversations").json() if c["id"] == conversation_id)
    assert summary["title"] == "why does it overheat?"


def test_list_conversations_reports_message_counts(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    _upload(client, manual_pdf_bytes)
    conversation_id = client.post("/query", json={"question": "why does it overheat?"}).json()[
        "conversation_id"
    ]

    summary = next(c for c in client.get("/conversations").json() if c["id"] == conversation_id)
    assert summary["message_count"] == 2


def test_agent_query_also_records_history_with_its_own_pipeline_label(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    _upload(client, manual_pdf_bytes)

    conversation_id = client.post(
        "/agent/query", json={"question": "why does it overheat?"}
    ).json()["conversation_id"]

    detail = client.get(f"/conversations/{conversation_id}").json()
    assert detail["messages"][1]["pipeline"] == "agent"


def test_delete_conversation_removes_it(client: TestClient, manual_pdf_bytes: bytes) -> None:
    _upload(client, manual_pdf_bytes)
    conversation_id = client.post("/query", json={"question": "why does it overheat?"}).json()[
        "conversation_id"
    ]

    assert client.delete(f"/conversations/{conversation_id}").status_code == 204
    assert client.get(f"/conversations/{conversation_id}").status_code == 404


def test_unknown_conversation_id_falls_back_to_starting_a_new_one(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    """a stale id from cleared client-side state should not 500 or 404 the question"""
    _upload(client, manual_pdf_bytes)

    response = client.post(
        "/query", json={"question": "why does it overheat?", "conversation_id": "does-not-exist"}
    )

    assert response.status_code == 200
    assert response.json()["conversation_id"] != "does-not-exist"


def test_get_unknown_conversation_is_404(client: TestClient) -> None:
    assert client.get("/conversations/does-not-exist").status_code == 404


def test_delete_unknown_conversation_is_404(client: TestClient) -> None:
    assert client.delete("/conversations/does-not-exist").status_code == 404


def test_citations_are_persisted_in_history(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    _upload(client, manual_pdf_bytes)
    answer_generator.output = "Caused by low airflow [page 1]."

    conversation_id = client.post("/query", json={"question": "why does it overheat?"}).json()[
        "conversation_id"
    ]

    citations = client.get(f"/conversations/{conversation_id}").json()["messages"][1]["citations"]
    assert citations
    assert citations[0]["page_number"] == 1
