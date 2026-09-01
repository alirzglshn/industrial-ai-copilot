"""/query/stream and /agent/query/stream: the sse wire format and event order"""

import json

from fastapi.testclient import TestClient

from copilot.api.deps import require_answer_generator
from copilot.generation.base import AnswerGenerator
from copilot.main import app
from tests.fakes import ScriptedAnswerGenerator


def _upload(client: TestClient, content: bytes, filename: str = "manual.pdf"):
    return client.post("/documents/upload", files={"file": (filename, content, "application/pdf")})


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        event = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        events.append((event, json.loads(data)))
    return events


class UnstreamableAnswerGenerator(AnswerGenerator):
    """a generator that only implements generate(), like vlmanswergenerator"""

    def generate(self, question, evidence):
        raise NotImplementedError


def test_query_stream_emits_tokens_then_a_result(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    _upload(client, manual_pdf_bytes)
    answer_generator.output = "Caused by low airflow [page 1]."

    response = client.post("/query/stream", json={"question": "why does it overheat?"})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    kinds = [event for event, _ in events]

    assert kinds[-1] == "result"
    assert kinds.count("token") >= 1
    assert kinds.index("token") < kinds.index("result")

    result = events[-1][1]
    assert result["answer"] == "Caused by low airflow [page 1]."
    assert result["grounded"] is True
    assert result["citations"]


def test_query_stream_tokens_concatenate_to_the_final_answer(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    _upload(client, manual_pdf_bytes)
    answer_generator.output = "Caused by low airflow [page 1]."

    events = _parse_sse(client.post("/query/stream", json={"question": "why?"}).text)

    streamed_text = "".join(data["text"] for kind, data in events if kind == "token")
    assert streamed_text == "Caused by low airflow [page 1]."


def test_query_stream_records_history(
    client: TestClient, manual_pdf_bytes: bytes, answer_generator: ScriptedAnswerGenerator
) -> None:
    _upload(client, manual_pdf_bytes)
    answer_generator.output = "Caused by low airflow [page 1]."

    events = _parse_sse(client.post("/query/stream", json={"question": "why?"}).text)
    conversation_id = events[-1][1]["conversation_id"]

    detail = client.get(f"/conversations/{conversation_id}").json()
    assert len(detail["messages"]) == 2
    assert detail["messages"][1]["text"] == "Caused by low airflow [page 1]."


def test_query_stream_with_no_evidence_still_emits_a_result(client: TestClient) -> None:
    events = _parse_sse(client.post("/query/stream", json={"question": "why does it overheat?"}).text)

    assert events[-1][0] == "result"
    assert events[-1][1]["insufficient_evidence"] is True


def test_query_stream_501s_for_a_generator_without_streaming_support(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    _upload(client, manual_pdf_bytes)
    app.dependency_overrides[require_answer_generator] = lambda: UnstreamableAnswerGenerator()
    try:
        response = client.post("/query/stream", json={"question": "why?"})
    finally:
        app.dependency_overrides.pop(require_answer_generator, None)

    assert response.status_code == 501


def test_agent_stream_emits_tool_calls_then_tokens_then_result(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    _upload(client, manual_pdf_bytes)

    events = _parse_sse(client.post("/agent/query/stream", json={"question": "why?"}).text)
    kinds = [event for event, _ in events]

    assert kinds[0] == "tool_calls"
    assert kinds[-1] == "result"
    assert kinds.index("tool_calls") < kinds.index("token") < kinds.index("result")
    assert events[0][1]["tool_calls"]


def test_agent_stream_result_reports_the_tool_trace(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    _upload(client, manual_pdf_bytes)

    events = _parse_sse(client.post("/agent/query/stream", json={"question": "why?"}).text)

    result = events[-1][1]
    assert result["tool_calls"]
    assert any("search_documents" in call for call in result["tool_calls"])


def test_agent_stream_records_history_with_agent_pipeline(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    _upload(client, manual_pdf_bytes)

    events = _parse_sse(client.post("/agent/query/stream", json={"question": "why?"}).text)
    conversation_id = events[-1][1]["conversation_id"]

    detail = client.get(f"/conversations/{conversation_id}").json()
    assert detail["messages"][1]["pipeline"] == "agent"
