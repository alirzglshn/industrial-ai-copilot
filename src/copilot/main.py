from contextlib import asynccontextmanager

from fastapi import FastAPI

from copilot.api.routes import documents, health, query, search
from copilot.core.config import get_settings
from copilot.core.logging import configure_logging
from copilot.db.models import Base
from copilot.db.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Phase 1: create tables directly from the models. Replace with Alembic
    # migrations once the schema needs to evolve without dropping data.
    Base.metadata.create_all(bind=engine)
    yield


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(query.router)

    return app


app = create_app()
