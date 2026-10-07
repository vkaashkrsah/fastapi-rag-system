"""Validated environment configuration; no secrets in source control."""

from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr
    api_key: SecretStr
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=1)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "palmmind_documents_v1"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "sqlite:///./data/app.db"
    booking_timezone: str = "Asia/Kolkata"
    memory_ttl_seconds: int = Field(default=86400, ge=60)
    history_messages: int = Field(default=20, ge=2, le=40)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_text_chars: int = Field(default=500_000, ge=1)
    max_chunks: int = Field(default=1000, ge=1)
    top_k: int = Field(default=5, ge=1, le=10)
    score_threshold: float = Field(default=0.25, ge=-1, le=1)

    @field_validator("booking_timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value

    @field_validator("api_key", "openai_api_key")
    @classmethod
    def nonempty_secret(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value().strip()) < 16:
            raise ValueError("Provide a real secret of at least 16 characters")
        return value
