from pathlib import Path
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from qdrant_client import QdrantClient

from copilot.api.deps import get_optional_retrieval_stack, require_retrieval_stack
from copilot.core.config import Settings, get_settings
from copilot.db.models import Base
from copilot.db.session import get_db
from copilot.ingestion.service import build_ingestion_service, get_ingestion_service
from copilot.main import app
from copilot.retrieval.deps import RetrievalStack
from copilot.retrieval.indexer import ChunkIndexer
from copilot.retrieval.retriever import VectorRetriever
from copilot.retrieval.vector_store import QdrantVectorStore
from tests.fakes import HashingEmbedder
from tests.pdf_fixtures import build_manual_pdf


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a throwaway directory so tests never touch ./data."""
    return Settings(
        upload_dir=str(tmp_path / "uploads"),
        image_dir=str(tmp_path / "images"),
    )


@pytest.fixture
def db_session(tmp_path: Path) -> Iterator[Session]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


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
def retrieval_stack(
    embedder: HashingEmbedder, vector_store: QdrantVectorStore
) -> RetrievalStack:
    return RetrievalStack(
        embedder=embedder,
        store=vector_store,
        indexer=ChunkIndexer(embedder, vector_store),
        retriever=VectorRetriever(embedder, vector_store),
    )


@pytest.fixture
def client(
    db_session: Session, settings: Settings, retrieval_stack: RetrievalStack
) -> Iterator[TestClient]:
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_ingestion_service] = lambda: build_ingestion_service(settings)
    app.dependency_overrides[get_optional_retrieval_stack] = lambda: retrieval_stack
    app.dependency_overrides[require_retrieval_stack] = lambda: retrieval_stack
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
