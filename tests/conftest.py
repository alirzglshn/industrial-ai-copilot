from pathlib import Path
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from copilot.core.config import Settings, get_settings
from copilot.db.models import Base
from copilot.db.session import get_db
from copilot.ingestion.service import build_ingestion_service, get_ingestion_service
from copilot.main import app
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
def client(db_session: Session, settings: Settings) -> Iterator[TestClient]:
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_ingestion_service] = lambda: build_ingestion_service(settings)
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
