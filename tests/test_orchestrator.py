"""toolusingagent.run(): plan, execute tools, ground the final answer

the planner is stubbed throughout, so this covers what the orchestrator does
with a plan's results, not planning itself, which is test_planner.py's job
"""

from sqlalchemy.orm import Session, sessionmaker

from copilot.agent.orchestrator import ToolUsingAgent
from copilot.agent.planner import Plan, ToolCall
from copilot.agent.tools import (
    CalculatorTool,
    GetDocumentMetadataTool,
    GetPageTool,
    SearchDocumentsTool,
    SearchImagesTool,
)
from copilot.db.models import Chunk, Document
from copilot.generation.prompt import INSUFFICIENT_MARKER
from copilot.retrieval.deps import RetrievalStack
from tests.fakes import HashingEmbedder, ScriptedLocalLM


class StubPlanner:
    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self.asked: list[str] = []

    def plan(self, question: str) -> Plan:
        self.asked.append(question)
        return self._plan


def _tools(retrieval_stack: RetrievalStack, session_factory: sessionmaker) -> dict:
    return {
        "search_documents": SearchDocumentsTool(retrieval_stack.retriever),
        "search_images": SearchImagesTool(retrieval_stack.image_retriever),
        "get_page": GetPageTool(session_factory),
        "calculate": CalculatorTool(),
        "get_document_metadata": GetDocumentMetadataTool(session_factory),
    }


def _seed_chunk(
    db: Session, embedder: HashingEmbedder, store, document_id="doc-a", page=1, text="cooling airflow overheating"
) -> None:
    if db.get(Document, document_id) is None:
        db.add(Document(id=document_id, filename="manual.pdf", status="indexed", page_count=10))
        db.commit()
    chunk = Chunk(document_id=document_id, page_number=page, chunk_index=0, text=text)
    db.add(chunk)
    db.commit()
    store.upsert(
        ids=[chunk.id],
        vectors=embedder.embed_documents([text]),
        payloads=[{"document_id": document_id, "page_number": page, "chunk_index": 0, "text": text}],
    )


def test_evidence_from_a_search_tool_reaches_the_final_answer(
    retrieval_stack: RetrievalStack, session_factory: sessionmaker, db_session: Session, embedder: HashingEmbedder
) -> None:
    _seed_chunk(db_session, embedder, retrieval_stack.store)
    planner = StubPlanner(Plan(calls=[ToolCall(tool="search_documents", arguments={"query": "cooling airflow"})]))
    lm = ScriptedLocalLM(["Caused by insufficient cooling airflow [page 1]."])
    agent = ToolUsingAgent(planner, _tools(retrieval_stack, session_factory), lm)

    answer = agent.run("why does it overheat?")

    assert not answer.insufficient_evidence
    assert answer.evidence_used
    assert answer.evidence_used[0].page_number == 1
    assert "search_documents" in answer.tool_calls[0]


def test_a_calculation_reaches_the_prompt_as_a_computed_fact(
    retrieval_stack: RetrievalStack, session_factory: sessionmaker
) -> None:
    planner = StubPlanner(Plan(calls=[ToolCall(tool="calculate", arguments={"expression": "(95-80)/80*100"})]))
    lm = ScriptedLocalLM(["The percentage difference is 18.75%."])
    agent = ToolUsingAgent(planner, _tools(retrieval_stack, session_factory), lm)

    agent.run("what is the percentage difference between 95 and 80?")

    user_prompt_sent = lm.calls[0][1]
    assert "COMPUTED FACTS" in user_prompt_sent
    assert "18.75" in user_prompt_sent


def test_document_metadata_reaches_the_prompt_as_a_computed_fact(
    retrieval_stack: RetrievalStack, session_factory: sessionmaker, db_session: Session
) -> None:
    db_session.add(Document(id="doc-a", filename="grundfos_ups3.pdf", status="indexed", page_count=22))
    db_session.commit()
    planner = StubPlanner(Plan(calls=[ToolCall(tool="get_document_metadata", arguments={})]))
    lm = ScriptedLocalLM(["It has 22 pages."])
    agent = ToolUsingAgent(planner, _tools(retrieval_stack, session_factory), lm)

    agent.run("how many pages does grundfos_ups3 have?")

    user_prompt_sent = lm.calls[0][1]
    assert "grundfos_ups3.pdf" in user_prompt_sent
    assert "22 pages" in user_prompt_sent


def test_no_evidence_and_no_facts_refuses_without_calling_the_model(
    retrieval_stack: RetrievalStack, session_factory: sessionmaker
) -> None:
    """an empty context must not invite a pretraining guess"""
    planner = StubPlanner(Plan(calls=[ToolCall(tool="search_documents", arguments={"query": "anything"})]))
    lm = ScriptedLocalLM(["should never be called"])
    agent = ToolUsingAgent(planner, _tools(retrieval_stack, session_factory), lm)

    answer = agent.run("what is the capital of France?")

    assert answer.insufficient_evidence
    assert INSUFFICIENT_MARKER in answer.text
    assert lm.calls == []
    assert "search_documents" in answer.tool_calls[0]


def test_a_failing_tool_call_is_caught_and_logged_not_raised(
    retrieval_stack: RetrievalStack, session_factory: sessionmaker
) -> None:
    planner = StubPlanner(
        Plan(
            calls=[
                ToolCall(tool="calculate", arguments={"expression": "1/0"}),
                ToolCall(tool="calculate", arguments={"expression": "2+2"}),
            ]
        )
    )
    lm = ScriptedLocalLM(["The result is 4."])
    agent = ToolUsingAgent(planner, _tools(retrieval_stack, session_factory), lm)

    answer = agent.run("what is 1/0 and 2+2?")

    assert "error" in answer.tool_calls[0]
    assert "4" in answer.tool_calls[1] or "2+2" in answer.tool_calls[1]
    # the failed call did not stop the second one from running
    assert lm.calls


def test_a_missing_page_is_caught_as_a_tool_failure(
    retrieval_stack: RetrievalStack, session_factory: sessionmaker, db_session: Session
) -> None:
    db_session.add(Document(id="doc-a", filename="manual.pdf", status="indexed", page_count=1))
    db_session.commit()
    planner = StubPlanner(Plan(calls=[ToolCall(tool="get_page", arguments={"document_id": "doc-a", "page_number": 99})]))
    agent = ToolUsingAgent(planner, _tools(retrieval_stack, session_factory), ScriptedLocalLM())

    answer = agent.run("what does page 99 say?")

    assert answer.insufficient_evidence
    assert "error" in answer.tool_calls[0]


def test_duplicate_evidence_from_two_tool_calls_is_not_double_counted(
    retrieval_stack: RetrievalStack, session_factory: sessionmaker, db_session: Session, embedder: HashingEmbedder
) -> None:
    _seed_chunk(db_session, embedder, retrieval_stack.store, page=1)
    # get_page(1) and search_documents both surface the same chunk on page 1
    planner = StubPlanner(
        Plan(
            calls=[
                ToolCall(tool="search_documents", arguments={"query": "cooling airflow"}),
                ToolCall(tool="get_page", arguments={"document_id": "doc-a", "page_number": 1}),
            ]
        )
    )
    lm = ScriptedLocalLM(["Caused by low airflow [page 1]."])
    agent = ToolUsingAgent(planner, _tools(retrieval_stack, session_factory), lm)

    agent.run("why does it overheat?")

    user_prompt_sent = lm.calls[0][1]
    # exactly one numbered evidence line, not two, for the same chunk
    assert user_prompt_sent.count("(page 1)") == 1


def test_agent_level_document_id_scopes_calls_that_did_not_specify_one(
    retrieval_stack: RetrievalStack, session_factory: sessionmaker, db_session: Session, embedder: HashingEmbedder
) -> None:
    _seed_chunk(db_session, embedder, retrieval_stack.store, document_id="doc-a", page=1)
    _seed_chunk(db_session, embedder, retrieval_stack.store, document_id="doc-b", page=1)
    planner = StubPlanner(Plan(calls=[ToolCall(tool="search_documents", arguments={"query": "cooling airflow"})]))
    lm = ScriptedLocalLM(["Caused by low airflow [page 1]."])
    agent = ToolUsingAgent(planner, _tools(retrieval_stack, session_factory), lm)

    answer = agent.run("why does it overheat?", document_id="doc-b")

    assert answer.evidence_used
    assert all(item.document_id == "doc-b" for item in answer.evidence_used)


def test_a_calls_own_document_id_is_not_overridden(
    retrieval_stack: RetrievalStack, session_factory: sessionmaker, db_session: Session, embedder: HashingEmbedder
) -> None:
    """comparing two named manuals needs per-call scoping the top-level filter must not clobber"""
    _seed_chunk(db_session, embedder, retrieval_stack.store, document_id="doc-a", page=1)
    _seed_chunk(db_session, embedder, retrieval_stack.store, document_id="doc-b", page=1)
    planner = StubPlanner(
        Plan(
            calls=[
                ToolCall(tool="search_documents", arguments={"query": "cooling airflow", "document_id": "doc-a"}),
            ]
        )
    )
    lm = ScriptedLocalLM(["Caused by low airflow [page 1]."])
    agent = ToolUsingAgent(planner, _tools(retrieval_stack, session_factory), lm)

    answer = agent.run("why does model A overheat?", document_id="doc-b")

    assert answer.evidence_used
    assert all(item.document_id == "doc-a" for item in answer.evidence_used)


def test_max_steps_truncates_the_plan_even_if_the_planner_did_not(
    retrieval_stack: RetrievalStack, session_factory: sessionmaker
) -> None:
    calls = [ToolCall(tool="calculate", arguments={"expression": str(i)}) for i in range(10)]
    planner = StubPlanner(Plan(calls=calls))
    lm = ScriptedLocalLM(["ok"])
    agent = ToolUsingAgent(planner, _tools(retrieval_stack, session_factory), lm, max_steps=3)

    answer = agent.run("do a lot of math")

    assert len(answer.tool_calls) == 3


def test_answer_from_tools_is_still_faithfulness_checked(
    retrieval_stack: RetrievalStack, session_factory: sessionmaker, db_session: Session, embedder: HashingEmbedder
) -> None:
    """a citation resolving to a real tool-fetched page still has to say what that page says"""
    _seed_chunk(db_session, embedder, retrieval_stack.store, text="insufficient cooling airflow across fins")
    planner = StubPlanner(Plan(calls=[ToolCall(tool="search_documents", arguments={"query": "cooling airflow"})]))
    lm = ScriptedLocalLM(["The capital of France is Paris [page 1]."])
    agent = ToolUsingAgent(planner, _tools(retrieval_stack, session_factory), lm)

    answer = agent.run("what is the capital of France?")

    assert not answer.grounded
    assert answer.faithfulness < 0.5
