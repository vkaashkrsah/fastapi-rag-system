"""Public contracts and validated structured LLM results."""

from datetime import date as Date
from datetime import time as Time
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

Strategy = Literal["fixed", "paragraph"]


class IngestResponse(BaseModel):
    document_id: UUID
    filename: str
    strategy: Strategy
    chunk_count: int
    status: Literal["ready"] = "ready"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: UUID
    request_id: UUID
    message: str = Field(min_length=1, max_length=4000)
    document_ids: list[UUID] = Field(default_factory=list, max_length=20)

    @field_validator("message")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message must not be blank")
        return value.strip()


class Source(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


class BookingDraft(BaseModel):
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None


class BookingInput(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    date: Date
    time: Time

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Name must have at least two non-space characters")
        return value

    @field_validator("time")
    @classmethod
    def local_minutes(cls, value: Time) -> Time:
        if value.tzinfo is not None or value.second or value.microsecond:
            raise ValueError("Use local time HH:MM without seconds or an offset")
        return value


class BookingResult(BookingInput):
    booking_id: str
    timezone: str
    status: Literal["booked"] = "booked"


class ChatResponse(BaseModel):
    session_id: UUID
    request_id: UUID
    answer: str
    sources: list[Source] = Field(default_factory=list)
    booking: BookingResult | None = None
    booking_draft: BookingDraft | None = None


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionState(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    booking_draft: BookingDraft | None = None
    awaiting_confirmation: bool = False


class Plan(BaseModel):
    """Required nullable fields are compatible with strict structured outputs."""

    intent: Literal["question", "booking", "cancel_booking"]
    standalone_query: str
    name: str | None
    email: str | None
    date: str | None
    time: str | None
    confirm: bool


class GroundedAnswer(BaseModel):
    answer: str
    source_indices: list[int]
