from pathlib import Path
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from qdrant_client import QdrantClient

from copilot.agent.orchestrator import ToolUsingAgent
from copilot.agent.tools import (
    CalculatorTool,
    GetDocumentMetadataTool,
    GetPageTool,
    SearchDocumentsTool,
    SearchImagesTool,
)
from copilot.api.deps import (
    get_optional_retrieval_stack,
    require_agent,
    require_answer_generator,
    require_retrieval_stack,
)
from copilot.core.config import Settings, get_settings
from copilot.db.models import Base
from copilot.db.session import get_db, get_session_factory
from copilot.ingestion.service import build_ingestion_service, get_ingestion_service
from copilot.main import app
from copilot.retrieval.deps import RetrievalStack
from copilot.retrieval.image_indexer import ImageIndexer
from copilot.retrieval.image_retriever import DbPageImageSource, ImageRetriever
from copilot.retrieval.indexer import ChunkIndexer
from copilot.retrieval.multimodal import MultimodalRetriever
from copilot.retrieval.retriever import VectorRetriever
from copilot.retrieval.vector_store import QdrantVectorStore
from tests.fakes import (
    FallbackOnlyPlanner,
    HashingEmbedder,
    HashingImageEmbedder,
    ScriptedAnswerGenerator,
    ScriptedLocalLM,
)
from tests.pdf_fixtures import build_manual_pdf


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a throwaway directory so tests never touch ./data."""
    return Settings(
        upload_dir=str(tmp_path / "uploads"),
        image_dir=str(tmp_path / "images"),
        preview_dir=str(tmp_path / "previews"),
    )


@pytest.fixture
def session_factory(tmp_path: Path) -> Iterator[sessionmaker]:
    """A file-backed SQLite database.

    On disk rather than in memory because the page-context image lookup opens
    its own short-lived session, exactly as it does in production.
    """
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    try:
        yield sessionmaker(bind=engine, autoflush=False, autocommit=False)
    finally:
        engine.dispose()


@pytest.fixture
def db_session(session_factory: sessionmaker) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def vector_store(embedder: HashingEmbedder) -> Iterator[QdrantVectorStore]:
    """A real Qdrant running in-process.

    qdrant-client's local mode implements the same query semantics as the
    server, so filtering, scoring and top-k are exercised for real without
    needing a container in the test run.
    """
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(
        client=client, collection_name="test_chunks", dimension=embedder.dimension
    )
    store.ensure_collection()
    try:
        yield store
    finally:
        client.close()


@pytest.fixture
def image_embedder() -> HashingImageEmbedder:
    return HashingImageEmbedder()


@pytest.fixture
def image_store(image_embedder: HashingImageEmbedder) -> Iterator[QdrantVectorStore]:
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(
        client=client, collection_name="test_images", dimension=image_embedder.dimension
    )
    store.ensure_collection()
    try:
        yield store
    finally:
        client.close()


@pytest.fixture
def retrieval_stack(
    embedder: HashingEmbedder,
    vector_store: QdrantVectorStore,
    image_embedder: HashingImageEmbedder,
    image_store: QdrantVectorStore,
    session_factory: sessionmaker,
) -> RetrievalStack:
    text_retriever = VectorRetriever(embedder, vector_store)
    image_retriever = ImageRetriever(image_embedder, image_store)
    return RetrievalStack(
        embedder=embedder,
        store=vector_store,
        indexer=ChunkIndexer(embedder, vector_store),
        retriever=text_retriever,
        multimodal=MultimodalRetriever(
            text_retriever=text_retriever,
            image_retriever=image_retriever,
            page_images=DbPageImageSource(session_factory),
        ),
        image_embedder=image_embedder,
        image_store=image_store,
        image_indexer=ImageIndexer(image_embedder, image_store),
        image_retriever=image_retriever,
    )


@pytest.fixture
def answer_generator() -> ScriptedAnswerGenerator:
    return ScriptedAnswerGenerator()


@pytest.fixture
def agent_lm() -> ScriptedLocalLM:
    """Only backs the agent's final-answer step; planning uses FallbackOnlyPlanner.

    Mutate `agent_lm.outputs` per test the same way `answer_generator.output`
    is mutated, to script what the "model" answers once tools have run.
    """
    return ScriptedLocalLM(["Caused by insufficient cooling airflow [page 1]."])


@pytest.fixture
def agent(
    retrieval_stack: RetrievalStack, session_factory: sessionmaker, agent_lm: ScriptedLocalLM
) -> ToolUsingAgent:
    tools = {
        "search_documents": SearchDocumentsTool(retrieval_stack.retriever),
        "search_images": SearchImagesTool(retrieval_stack.image_retriever),
        "get_page": GetPageTool(session_factory),
        "calculate": CalculatorTool(),
        "get_document_metadata": GetDocumentMetadataTool(session_factory),
    }
    return ToolUsingAgent(planner=FallbackOnlyPlanner(), tools=tools, lm=agent_lm)


@pytest.fixture
def client(
    db_session: Session,
    session_factory: sessionmaker,
    settings: Settings,
    retrieval_stack: RetrievalStack,
    answer_generator: ScriptedAnswerGenerator,
    agent: ToolUsingAgent,
) -> Iterator[TestClient]:
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_ingestion_service] = lambda: build_ingestion_service(settings)
    app.dependency_overrides[get_optional_retrieval_stack] = lambda: retrieval_stack
    app.dependency_overrides[require_retrieval_stack] = lambda: retrieval_stack
    app.dependency_overrides[require_answer_generator] = lambda: answer_generator
    app.dependency_overrides[require_agent] = lambda: agent
    # Streaming routes open their own session after the request handler has
    # returned (see get_session_factory's docstring), so they need the same
    # test engine wired in separately from get_db.
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    # Deliberately not used as a context manager: that would run the app
    # lifespan, which creates tables against the real (Postgres) engine.
    # Tables here are created on the SQLite engine by the db_session fixture.
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def manual_pdf(tmp_path: Path) -> Path:
    return build_manual_pdf(tmp_path / "manual.pdf")


@pytest.fixture
def manual_pdf_bytes(manual_pdf: Path) -> bytes:
    return manual_pdf.read_bytes()
