"""Redis session state with expiration, bounded history and per-session locking."""

from typing import cast

from redis import Redis
from redis.lock import Lock

from app.config import Settings
from app.schemas import SessionState


class ChatMemory:
    def __init__(self, client: Redis, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def get(self, session_id: str) -> SessionState:
        data = cast(str | None, self.client.get(f"chat:{session_id}"))
        return SessionState.model_validate_json(data) if data else SessionState()

    def save(self, session_id: str, state: SessionState) -> None:
        state.messages = state.messages[-self.settings.history_messages :]
        self.client.set(
            f"chat:{session_id}", state.model_dump_json(), ex=self.settings.memory_ttl_seconds
        )

    def lock(self, session_id: str) -> Lock:
        # API calls have 30s timeouts and no retries; a chat uses at most three calls.
        return cast(Lock, self.client.lock(f"lock:{session_id}", timeout=300, blocking_timeout=2))
