from uuid import uuid4

from sqlalchemy import func, select

from app.database import Booking, ChatTurn
from app.schemas import GroundedAnswer
from tests.conftest import plan


def chat(client, session=None, message="What skills?", documents=None, request_id=None):
    return client.post(
        "/api/v1/chat",
        json={
            "session_id": session or str(uuid4()),
            "request_id": request_id or str(uuid4()),
            "message": message,
            "document_ids": documents or [],
        },
    )


def upload(client, text=b"Python skills"):
    return client.post("/api/v1/documents", files={"file": ("sample.txt", text)}).json()[
        "document_id"
    ]


def test_auth_and_exactly_two_business_endpoints(client):
    client.headers.pop("Authorization")
    assert chat(client).status_code == 401
    assert client.post("/api/v1/documents", files={"file": ("a.txt", b"hi")}).status_code == 401
    assert set(client.get("/openapi.json").json()["paths"]) == {"/api/v1/documents", "/api/v1/chat"}
    assert client.get("/docs").status_code == 404


def test_validation(client):
    assert chat(client, message="   ").status_code == 422
    assert chat(client, session="invalid-uuid").status_code == 422
    assert chat(client, message="x" * 4001).status_code == 422


def test_rag_multi_turn_and_document_filter(client, service):
    document = upload(client)
    other = upload(client, b"Different document")
    session = str(uuid4())
    first = chat(client, session, documents=[document])
    assert first.status_code == 200
    assert {s["document_id"] for s in first.json()["sources"]} == {document}
    assert all(s["document_id"] != other for s in first.json()["sources"])
    service.llm.plans.append(plan(standalone_query="What Python skills are useful?"))
    second = chat(client, session, "Tell me more about those", [document])
    assert second.status_code == 200
    assert len(service.llm.seen_states[-1].messages) == 2
    assert service.llm.queries[-1] == "What Python skills are useful?"
    assert service.memory.client.ttl(f"chat:{session}") > 0
    chat(client, str(uuid4()), documents=[document])
    assert service.llm.seen_states[-1].messages == []


def test_missing_documents_and_invalid_citations(client, service):
    assert "Upload" in chat(client).json()["answer"]
    assert chat(client, documents=[str(uuid4())]).status_code == 404
    document = upload(client)
    service.llm.answer_result = GroundedAnswer(answer="Invented claim", source_indices=[99])
    result = chat(client, documents=[document]).json()
    assert "supported answer" in result["answer"]
    assert result["sources"] == []


def test_empty_retrieval(client, service, monkeypatch):
    document = upload(client)
    monkeypatch.setattr(service.vectors, "search", lambda *args: [])
    result = chat(client, documents=[document]).json()
    assert "could not find" in result["answer"]
    assert service.llm.queries == []


def fill_booking(client, service, session, email="vikash@example.com"):
    service.llm.plans.append(plan(intent="booking", name="Vikash", email=email))
    first = chat(client, session, "Book an interview; I am Vikash, " + email)
    assert "date, time" in first.json()["answer"]
    service.llm.plans.append(plan(intent="booking", date="2099-10-10", time="14:30"))
    second = chat(client, session, "2099-10-10 at 14:30")
    assert "Please confirm" in second.json()["answer"]
    return second


def test_booking_multi_turn_confirm_replay_conflict(client, service):
    session = str(uuid4())
    fill_booking(client, service, session)
    with service.sessions() as db:
        assert db.scalar(select(func.count()).select_from(Booking)) == 0
    service.llm.plans.append(plan(intent="booking", confirm=True))
    request = str(uuid4())
    result = chat(client, session, "confirm", request_id=request)
    assert result.status_code == 200
    assert result.json()["booking"]["email"] == "vikash@example.com"
    assert result.json()["booking"]["timezone"] == "Asia/Kolkata"
    assert chat(client, session, "confirm", request_id=request).json() == result.json()
    assert chat(client, session, "changed", request_id=request).status_code == 409
    with service.sessions() as db:
        assert db.scalar(select(func.count()).select_from(Booking)) == 1
        assert db.scalar(select(func.count()).select_from(ChatTurn)) == 3


def test_booking_validation_and_past_date(client, service):
    session = str(uuid4())
    service.llm.plans.append(
        plan(intent="booking", name="Vikash", email="bad-email", date="2099-10-10", time="14:30")
    )
    assert "correct: email" in chat(client, session).json()["answer"]
    service.llm.plans.append(plan(intent="booking", email="vikash@example.com", date="2000-01-01"))
    assert "future" in chat(client, session).json()["answer"]
    service.llm.plans.append(plan(intent="booking", date="2099-02-30"))
    assert "correct: date" in chat(client, session).json()["answer"]
    with service.sessions() as db:
        assert db.scalar(select(func.count()).select_from(Booking)) == 0


def test_corrections_require_fresh_confirmation(client, service):
    session = str(uuid4())
    fill_booking(client, service, session)
    service.llm.plans.append(plan(intent="booking", time="15:00", confirm=True))
    result = chat(client, session).json()
    assert "Please confirm" in result["answer"]
    assert result["booking"] is None


def test_slot_collision(client, service):
    for i in range(2):
        session = str(uuid4())
        fill_booking(client, service, session)
        service.llm.plans.append(plan(intent="booking", confirm=True))
        result = chat(client, session, "confirm").json()
        if i == 0:
            assert result["booking"] is not None
        else:
            assert "already booked" in result["answer"]
            assert result["booking"] is None


def test_clear_draft(client, service):
    session = str(uuid4())
    fill_booking(client, service, session)
    service.llm.plans.append(plan(intent="cancel_booking"))
    chat(client, session, "Cancel that draft")
    assert service.memory.get(session).booking_draft is None


def test_lock_rejects_concurrent_session(client, service):
    session = str(uuid4())
    lock = service.memory.lock(session)
    assert lock.acquire()
    try:
        assert chat(client, session).status_code == 409
    finally:
        lock.release()


def test_redis_failure_after_commit_replays_without_duplicate(client, service, monkeypatch):
    session = str(uuid4())
    fill_booking(client, service, session)
    service.llm.plans.append(plan(intent="booking", confirm=True))
    original = service.memory.save

    def unavailable(*args):
        raise RuntimeError("Redis unavailable")

    monkeypatch.setattr(service.memory, "save", unavailable)
    request = str(uuid4())
    assert chat(client, session, "confirm", request_id=request).status_code == 503
    monkeypatch.setattr(service.memory, "save", original)
    result = chat(client, session, "confirm", request_id=request)
    assert result.status_code == 200
    assert result.json()["booking"] is not None
    with service.sessions() as db:
        assert db.scalar(select(func.count()).select_from(Booking)) == 1


def test_old_retry_does_not_rewind_memory_and_history_is_bounded(client, service):
    session = str(uuid4())
    request = str(uuid4())
    service.settings.history_messages = 4
    chat(client, session, "first", request_id=request)
    chat(client, session, "second")
    chat(client, session, "third")
    chat(client, session, "first", request_id=request)
    state = service.memory.get(session)
    assert len(state.messages) == 4
    assert state.messages[-2].content == "third"


def test_booking_rolls_back_when_turn_persistence_fails(client, service):
    from sqlalchemy import event

    session = str(uuid4())
    fill_booking(client, service, session)
    service.llm.plans.append(plan(intent="booking", confirm=True))
    engine = service.sessions.kw["bind"]

    def fail_turn(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO chat_turns"):
            raise RuntimeError("Simulated SQL failure")

    event.listen(engine, "before_cursor_execute", fail_turn)
    try:
        assert chat(client, session, "confirm").status_code == 503
    finally:
        event.remove(engine, "before_cursor_execute", fail_turn)
    with service.sessions() as db:
        assert db.scalar(select(func.count()).select_from(Booking)) == 0
