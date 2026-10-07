from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import fakeredis
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.config import Settings
from app.database import make_database
from app.main import create_app
from app.memory import ChatMemory
from app.schemas import GroundedAnswer, Plan, SessionState, Source
from app.services import Services
from app.vector_store import VectorStore


class FakeLLM:
    """Deterministic provider double; never used by the actual application."""

    def __init__(self) -> None:
        self.plans: list[Plan] = []
        self.seen_states: list[SessionState] = []
        self.queries: list[str] = []
        self.answer_result = GroundedAnswer(answer="Python is a useful skill.", source_indices=[0])
        self.fail_embedding = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail_embedding:
            raise RuntimeError("Embedding outage")
        return [[1.0, 0.1, 0.0] for _ in texts]

    def plan(self, message: str, state: SessionState, now: datetime) -> Plan:
        self.seen_states.append(state.model_copy(deep=True))
        if self.plans:
            return self.plans.pop(0)
        return plan(standalone_query=message)

    def answer(self, query: str, sources: list[Source]) -> GroundedAnswer:
        self.queries.append(query)
        return self.answer_result


def plan(**overrides: object) -> Plan:
    fields = dict(
        intent="question",
        standalone_query="skills",
        name=None,
        email=None,
        date=None,
        time=None,
        confirm=False,
    )
    fields.update(overrides)
    return Plan.model_validate(fields)


@pytest.fixture
def service(tmp_path: Path) -> Iterator[Services]:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-openai-key-123456789",
        api_key="test-api-key-123456789",
        database_url=f"sqlite:///{tmp_path}/test.db",
        embedding_dimensions=3,
    )
    engine, sessions = make_database(settings.database_url)
    client = QdrantClient(":memory:")
    vectors = VectorStore(client, settings)
    vectors.initialize()
    redis = fakeredis.FakeRedis(decode_responses=True)
    yield Services(settings, sessions, FakeLLM(), vectors, ChatMemory(redis, settings))
    redis.close()
    client.close()
    engine.dispose()


@pytest.fixture
def client(service: Services) -> Iterator[TestClient]:
    with TestClient(create_app(service), raise_server_exceptions=False) as test_client:
        test_client.headers["Authorization"] = "Bearer test-api-key-123456789"
        yield test_client
