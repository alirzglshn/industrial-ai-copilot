import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from copilot.api.routes import agent, conversations, documents, health, query, search
from copilot.core.config import get_settings
from copilot.core.logging import configure_logging
from copilot.db.models import Base
from copilot.db.session import engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # tables created directly from the models, migrations come later
    Base.metadata.create_all(bind=engine)
    yield


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    # wide open, local only, lets the vite dev server call the api directly
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://localhost:\d+",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # traceback logged server-side, client gets a clean message
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(query.router)
    app.include_router(agent.router)
    app.include_router(conversations.router)

    return app


app = create_app()
