"""Exactly two business endpoints. External dependencies connect during startup."""

import logging
import secrets
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from qdrant_client import QdrantClient
from redis import Redis

from app.config import Settings
from app.database import make_database
from app.documents import DocumentError
from app.llm import OpenAILanguageModel
from app.memory import ChatMemory
from app.schemas import ChatRequest, ChatResponse, IngestResponse, Strategy
from app.services import ConflictError, NotFoundError, Services
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


@contextmanager
def build_services(settings: Settings) -> Iterator[Services]:
    engine, sessions = make_database(settings.database_url)
    redis: Redis = Redis.from_url(
        settings.redis_url, decode_responses=True, socket_timeout=5, socket_connect_timeout=5
    )
    qdrant = QdrantClient(
        url=settings.qdrant_url,
        timeout=10,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
    )
    llm = OpenAILanguageModel(settings)
    try:
        redis.ping()
        vectors = VectorStore(qdrant, settings)
        vectors.initialize()
        yield Services(settings, sessions, llm, vectors, ChatMemory(redis, settings))
    finally:
        llm.client.close()
        qdrant.close()
        redis.close()
        engine.dispose()


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            application.state.services = services
            yield
        else:
            with build_services(Settings()) as built:
                application.state.services = built
                yield

    application = FastAPI(
        title="PalmMind RAG Backend",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    def get_services(request: Request) -> Services:
        result: Services = request.app.state.services
        return result

    def authorize(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
        service: Annotated[Services, Depends(get_services)],
    ) -> None:
        expected = service.settings.api_key.get_secret_value()
        if credentials is None or not secrets.compare_digest(credentials.credentials, expected):
            raise HTTPException(
                401, "Invalid bearer API key", headers={"WWW-Authenticate": "Bearer"}
            )

    @application.exception_handler(DocumentError)
    async def document_error(request: Request, exc: DocumentError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @application.exception_handler(ConflictError)
    async def conflict_error(request: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @application.exception_handler(NotFoundError)
    async def missing_error(request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @application.exception_handler(Exception)
    async def dependency_error(request: Request, exc: Exception) -> JSONResponse:
        # Do not log request content, booking PII, secrets, or raw provider error bodies.
        logger.error("Request failed: %s", type(exc).__name__)
        return JSONResponse(
            status_code=503,
            content={"detail": "A dependency failed. Retry chat with the same request_id."},
        )

    @application.post(
        "/api/v1/documents",
        response_model=IngestResponse,
        status_code=201,
        dependencies=[Depends(authorize)],
    )
    def ingest_document(
        service: Annotated[Services, Depends(get_services)],
        file: Annotated[UploadFile, File()],
        strategy: Annotated[Strategy, Form()] = "fixed",
        chunk_size: Annotated[int, Form(ge=100, le=2000)] = 1000,
        overlap: Annotated[int | None, Form(ge=0)] = None,
    ) -> IngestResponse:
        # Read at most the accepted file size + 1; multipart parsing is handled by Starlette.
        try:
            data = file.file.read(service.settings.max_upload_bytes + 1)
        finally:
            file.file.close()
        if len(data) > service.settings.max_upload_bytes:
            raise HTTPException(413, "File exceeds the upload size limit")
        effective_overlap = overlap if overlap is not None else (100 if strategy == "fixed" else 0)
        return service.ingest(file.filename or "", data, strategy, chunk_size, effective_overlap)

    @application.post(
        "/api/v1/chat",
        response_model=ChatResponse,
        dependencies=[Depends(authorize)],
    )
    def converse(
        payload: ChatRequest,
        service: Annotated[Services, Depends(get_services)],
    ) -> ChatResponse:
        return service.chat(payload)

    return application


app = create_app()
