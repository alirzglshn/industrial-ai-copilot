"""end to end: /agent/query over an ingested manual

the default agent fixture uses fallbackonlyplanner (see tests/fakes.py),
running the real tools against the real retrieval stack
"""

from fastapi.testclient import TestClient

from copilot.agent.orchestrator import ToolUsingAgent
from copilot.agent.planner import Plan, ToolCall
from copilot.agent.tools import CalculatorTool, GetDocumentMetadataTool, GetPageTool
from copilot.api.deps import require_agent
from copilot.generation.prompt import INSUFFICIENT_MARKER
from copilot.main import app
from tests.fakes import ScriptedLocalLM


def _upload(client: TestClient, content: bytes, filename: str = "manual.pdf"):
    return client.post("/documents/upload", files={"file": (filename, content, "application/pdf")})


def _ask_agent(client: TestClient, question: str, **extra):
    return client.post("/agent/query", json={"question": question, **extra})


class _FixedPlanner:
    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    def plan(self, question: str) -> Plan:
        return self._plan


def _override_agent(tools: dict, plan: Plan, lm_outputs: list[str]):
    lm = ScriptedLocalLM(lm_outputs)
    agent = ToolUsingAgent(planner=_FixedPlanner(plan), tools=tools, lm=lm)
    app.dependency_overrides[require_agent] = lambda: agent
    return agent


def test_answers_with_citations_via_the_default_search_plan(
    client: TestClient, manual_pdf_bytes: bytes, agent_lm: ScriptedLocalLM
) -> None:
    _upload(client, manual_pdf_bytes)
    agent_lm.outputs = ["Caused by insufficient cooling airflow [page 1]."]

    body = _ask_agent(client, "why does the pump overheat?").json()

    assert body["answer"] == "Caused by insufficient cooling airflow [page 1]."
    assert body["insufficient_evidence"] is False
    assert body["citations"]
    assert body["tool_calls"]
    assert any("search_documents" in call for call in body["tool_calls"])


def test_tool_calls_are_reported_in_the_response(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    _upload(client, manual_pdf_bytes)

    body = _ask_agent(client, "why does the pump overheat?").json()

    assert len(body["tool_calls"]) >= 1


def test_query_endpoint_reports_no_tool_calls_for_comparison(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    """/query is the fixed pipeline, its tool_calls should always be empty"""
    _upload(client, manual_pdf_bytes)

    body = client.post("/query", json={"question": "why does it overheat?"}).json()

    assert body["tool_calls"] == []


def test_a_pure_calculation_question_uses_only_the_calculator(client: TestClient) -> None:
    plan = Plan(calls=[ToolCall(tool="calculate", arguments={"expression": "(95-80)/80*100"})])
    _override_agent({"calculate": CalculatorTool()}, plan, ["The percentage difference is 18.75%."])

    try:
        body = _ask_agent(client, "what is the percentage difference between 95 and 80?").json()
    finally:
        app.dependency_overrides.pop(require_agent, None)

    assert body["insufficient_evidence"] is False
    assert "18.75" in body["answer"]
    assert body["tool_calls"] == ["calculate(expression='(95-80)/80*100') -> 18.75"]


def test_a_metadata_question_uses_get_document_metadata(
    client: TestClient, manual_pdf_bytes: bytes, session_factory
) -> None:
    document_id = _upload(client, manual_pdf_bytes).json()["id"]
    plan = Plan(calls=[ToolCall(tool="get_document_metadata", arguments={"document_id": document_id})])
    _override_agent(
        {"get_document_metadata": GetDocumentMetadataTool(session_factory)},
        plan,
        ["This manual has 2 pages."],
    )

    try:
        body = _ask_agent(client, "how many pages does this manual have?").json()
    finally:
        app.dependency_overrides.pop(require_agent, None)

    assert "2 pages" in body["answer"]


def test_off_topic_question_is_refused(
    client: TestClient, manual_pdf_bytes: bytes, agent_lm: ScriptedLocalLM
) -> None:
    _upload(client, manual_pdf_bytes)
    agent_lm.outputs = [INSUFFICIENT_MARKER]

    body = _ask_agent(client, "what is the capital of France?").json()

    assert body["insufficient_evidence"] is True


def test_no_documents_uploaded_refuses_without_calling_the_model(
    client: TestClient, agent_lm: ScriptedLocalLM
) -> None:
    body = _ask_agent(client, "why does the pump overheat?").json()

    assert body["insufficient_evidence"] is True
    assert body["citations"] == []


def test_document_id_scopes_the_agents_search(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    first = _upload(client, manual_pdf_bytes, "first.pdf").json()["id"]
    _upload(client, manual_pdf_bytes, "second.pdf")

    body = _ask_agent(client, "why does it overheat?", document_id=first).json()

    assert body["citations"]
    assert all(c["document_id"] == first for c in body["citations"])


def test_empty_question_is_rejected(client: TestClient) -> None:
    assert _ask_agent(client, "").status_code == 422


def test_agent_unavailable_is_a_503(client: TestClient) -> None:
    def unavailable():
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Agent unavailable")

    app.dependency_overrides[require_agent] = unavailable
    try:
        assert _ask_agent(client, "why does it overheat?").status_code == 503
    finally:
        app.dependency_overrides.pop(require_agent, None)


def test_query_endpoint_is_unaffected_by_agent_unavailability(
    client: TestClient, manual_pdf_bytes: bytes
) -> None:
    """/query and /agent/query fail independently, retrieval works either way"""
    _upload(client, manual_pdf_bytes)

    def unavailable():
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Agent unavailable")

    app.dependency_overrides[require_agent] = unavailable
    try:
        response = client.post("/query", json={"question": "why does it overheat?"})
    finally:
        app.dependency_overrides.pop(require_agent, None)

    assert response.status_code == 200


def test_a_failing_tool_does_not_500_the_whole_request(
    client: TestClient, manual_pdf_bytes: bytes, session_factory
) -> None:
    """a get_page call for a missing page raises inside the tool, the endpoint must still respond"""
    document_id = _upload(client, manual_pdf_bytes).json()["id"]
    plan = Plan(calls=[ToolCall(tool="get_page", arguments={"document_id": document_id, "page_number": 999})])
    _override_agent({"get_page": GetPageTool(session_factory)}, plan, ["should not be reached"])

    try:
        response = _ask_agent(client, "what does page 999 say?")
    finally:
        app.dependency_overrides.pop(require_agent, None)

    assert response.status_code == 200
    assert response.json()["insufficient_evidence"] is True
