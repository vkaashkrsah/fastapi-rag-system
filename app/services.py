"""Application orchestration with explicit ingestion, RAG, and booking flows."""

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.database import Booking, ChatTurn, Chunk, Document
from app.documents import DocumentError, chunk_text, extract_text
from app.llm import LanguageModel
from app.memory import ChatMemory
from app.schemas import (
    BookingDraft,
    BookingInput,
    BookingResult,
    ChatRequest,
    ChatResponse,
    IngestResponse,
    Message,
    Plan,
    SessionState,
    Strategy,
)
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ConflictError(ValueError):
    pass


class NotFoundError(ValueError):
    pass


class Services:
    def __init__(
        self,
        settings: Settings,
        sessions: sessionmaker[Session],
        llm: LanguageModel,
        vectors: VectorStore,
        memory: ChatMemory,
    ) -> None:
        self.settings = settings
        self.sessions = sessions
        self.llm = llm
        self.vectors = vectors
        self.memory = memory

    def ingest(
        self,
        filename: str,
        data: bytes,
        strategy: Strategy,
        size: int,
        overlap: int,
    ) -> IngestResponse:
        filename = Path(filename.replace("\\", "/")).name[:255]
        text = extract_text(filename, data, self.settings.max_text_chars)
        chunks = chunk_text(text, strategy, size, overlap)
        if len(chunks) > self.settings.max_chunks:
            raise DocumentError("Too many chunks; increase chunk_size or reduce the file")
        document_id = str(uuid4())
        chunk_ids = [str(uuid4()) for _ in chunks]
        with self.sessions.begin() as db:
            db.add(
                Document(
                    id=document_id,
                    filename=filename,
                    sha256=hashlib.sha256(data).hexdigest(),
                    strategy=strategy,
                    chunk_size=size,
                    overlap=overlap,
                    chunk_count=len(chunks),
                    embedding_model=self.settings.embedding_model,
                    embedding_dimensions=self.settings.embedding_dimensions,
                    status="pending",
                )
            )
        try:
            embeddings = self.llm.embed(chunks)
            self.vectors.insert(document_id, filename, chunk_ids, chunks, embeddings)
            with self.sessions.begin() as db:
                document = db.get(Document, document_id)
                assert document is not None
                document.status = "ready"
                db.add_all(
                    [
                        Chunk(
                            id=chunk_ids[i],
                            document_id=document_id,
                            chunk_index=i,
                            char_count=len(chunk),
                        )
                        for i, chunk in enumerate(chunks)
                    ]
                )
        except Exception:
            # SQL readiness gates retrieval, even if vector cleanup itself is unavailable.
            with self.sessions.begin() as db:
                document = db.get(Document, document_id)
                assert document is not None
                document.status = "failed"
            try:
                self.vectors.delete_document(document_id)
            except Exception:
                logger.warning("Vector cleanup needed for document %s", document_id)
            raise
        return IngestResponse(
            document_id=UUID(document_id),
            filename=filename,
            strategy=strategy,
            chunk_count=len(chunks),
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        session_id = str(request.session_id)
        request_hash = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        lock = self.memory.lock(session_id)
        if not lock.acquire(blocking=True):
            raise ConflictError("Another message is processing for this session; retry shortly")
        try:
            with self.sessions() as db:
                cached = db.scalar(
                    select(ChatTurn).where(
                        ChatTurn.session_id == session_id,
                        ChatTurn.request_id == str(request.request_id),
                    )
                )
                if cached:
                    if cached.request_hash != request_hash:
                        raise ConflictError("request_id was already used with a different payload")
                    # Restore the newest snapshot, never rewind memory on an old retry.
                    latest = db.scalar(
                        select(ChatTurn)
                        .where(ChatTurn.session_id == session_id)
                        .order_by(ChatTurn.created_at.desc())
                        .limit(1)
                    )
                    assert latest is not None
                    self.memory.save(
                        session_id, SessionState.model_validate_json(latest.state_json)
                    )
                    return ChatResponse.model_validate_json(cached.response_json)
            state = self.memory.get(session_id)
            now = datetime.now(ZoneInfo(self.settings.booking_timezone))
            plan = self.llm.plan(request.message, state, now)
            response = ChatResponse(
                session_id=request.session_id, request_id=request.request_id, answer=""
            )
            with self.sessions.begin() as db:
                if plan.intent == "cancel_booking":
                    state.booking_draft = None
                    state.awaiting_confirmation = False
                    response.answer = "The unfinished booking draft has been cleared."
                elif plan.intent == "booking":
                    self._booking(db, request, plan, state, response, now)
                else:
                    self._rag(db, request, plan, response)
                state.messages.extend(
                    [
                        Message(role="user", content=request.message),
                        Message(role="assistant", content=response.answer),
                    ]
                )
                state.messages = state.messages[-self.settings.history_messages :]
                db.add(
                    ChatTurn(
                        id=str(uuid4()),
                        session_id=session_id,
                        request_id=str(request.request_id),
                        request_hash=request_hash,
                        response_json=response.model_dump_json(),
                        state_json=state.model_dump_json(),
                    )
                )
            # SQL commit first: if Redis fails, the same request_id safely replays the result.
            self.memory.save(session_id, state)
            return response
        finally:
            if lock.owned():
                lock.release()

    def _rag(
        self,
        db: Session,
        request: ChatRequest,
        plan: Plan,
        response: ChatResponse,
    ) -> None:
        ids = list(dict.fromkeys(str(x) for x in request.document_ids))
        if not ids:
            response.answer = "Upload a document, then include its document_id in document_ids."
            return
        documents = db.scalars(
            select(Document).where(Document.id.in_(ids), Document.status == "ready")
        ).all()
        if len(documents) != len(ids):
            raise NotFoundError("One or more documents do not exist or are not ready")
        if any(
            d.embedding_model != self.settings.embedding_model
            or d.embedding_dimensions != self.settings.embedding_dimensions
            for d in documents
        ):
            raise ConflictError("Embedding configuration changed; re-ingest into a new collection")
        query = plan.standalone_query.strip() or request.message
        sources = self.vectors.search(self.llm.embed([query])[0], ids)
        if not sources:
            response.answer = "I could not find relevant information in the selected documents."
            return
        answer = self.llm.answer(query, sources)
        indices = list(dict.fromkeys(answer.source_indices))
        if not indices or any(i < 0 or i >= len(sources) for i in indices):
            response.answer = "I could not find a supported answer in the selected documents."
            return
        response.answer = answer.answer
        response.sources = [sources[i] for i in indices]

    def _booking(
        self,
        db: Session,
        request: ChatRequest,
        plan: Plan,
        state: SessionState,
        response: ChatResponse,
        now: datetime,
    ) -> None:
        draft = state.booking_draft or BookingDraft()
        updates = {
            key: getattr(plan, key)
            for key in ("name", "email", "date", "time")
            if getattr(plan, key) is not None
        }
        changed = any(getattr(draft, key) != value for key, value in updates.items())
        draft = draft.model_copy(update=updates)
        state.booking_draft = draft
        response.booking_draft = draft
        was_awaiting = state.awaiting_confirmation
        state.awaiting_confirmation = False
        missing = [key for key, value in draft.model_dump().items() if not value]
        if missing:
            response.answer = "To book your interview, please provide: " + ", ".join(missing) + "."
            return
        try:
            booking = BookingInput.model_validate(draft.model_dump())
        except ValidationError as exc:
            fields = sorted({str(error["loc"][0]) for error in exc.errors()})
            response.answer = (
                "Please correct: "
                + ", ".join(fields)
                + ". Use a valid email, YYYY-MM-DD date, and 24-hour HH:MM time."
            )
            return
        scheduled = datetime.combine(booking.date, booking.time, tzinfo=now.tzinfo)
        if scheduled <= now:
            response.answer = "Please choose a future interview date and time."
            return
        # Reject ambiguous/nonexistent local times at daylight-saving transitions.
        zone = ZoneInfo(self.settings.booking_timezone)
        if (
            scheduled.astimezone(UTC).astimezone(zone).replace(tzinfo=None)
            != scheduled.replace(tzinfo=None)
            or scheduled.replace(fold=0).utcoffset() != scheduled.replace(fold=1).utcoffset()
        ):
            response.answer = "That local time is ambiguous or unavailable; choose another time."
            return
        if not (plan.confirm and was_awaiting and not changed):
            state.awaiting_confirmation = True
            response.answer = (
                f"Please confirm: {booking.name}, {booking.email}, {booking.date.isoformat()} "
                f"at {booking.time.strftime('%H:%M')} ({self.settings.booking_timezone}). "
                "Reply 'confirm' to save the booking, or provide corrections."
            )
            return
        booking_id = str(uuid4())
        try:
            with db.begin_nested():
                db.add(
                    Booking(
                        id=booking_id,
                        session_id=str(request.session_id),
                        name=booking.name,
                        email=str(booking.email),
                        scheduled_at=scheduled.astimezone(UTC).isoformat(),
                        timezone=self.settings.booking_timezone,
                    )
                )
                db.flush()
        except IntegrityError:
            response.answer = "That interview start time is already booked; choose another time."
            return
        response.booking = BookingResult(
            **booking.model_dump(), booking_id=booking_id, timezone=self.settings.booking_timezone
        )
        response.booking_draft = None
        state.booking_draft = None
        response.answer = (
            f"Interview booking saved for {booking.name} on {booking.date.isoformat()} "
            f"at {booking.time.strftime('%H:%M')} ({self.settings.booking_timezone}). "
            f"Booking ID: {booking_id}."
        )
