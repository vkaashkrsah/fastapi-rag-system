"""SQL persistence. Booking and chat replay records commit in one transaction."""

from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Connection as SQLiteConnection

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, create_engine, event
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import ConnectionPoolEntry


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64))
    strategy: Mapped[str] = mapped_column(String(20))
    chunk_size: Mapped[int]
    overlap: Mapped[int]
    chunk_count: Mapped[int]
    embedding_model: Mapped[str] = mapped_column(String(100))
    embedding_dimensions: Mapped[int]
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int]
    char_count: Mapped[int]


class Booking(Base):
    __tablename__ = "bookings"
    __table_args__ = (UniqueConstraint("scheduled_at", name="unique_interview_slot"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(254))
    scheduled_at: Mapped[str] = mapped_column(String(40))
    timezone: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class ChatTurn(Base):
    __tablename__ = "chat_turns"
    __table_args__ = (UniqueConstraint("session_id", "request_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    request_id: Mapped[str] = mapped_column(String(36))
    request_hash: Mapped[str] = mapped_column(String(64))
    response_json: Mapped[str] = mapped_column(Text)
    state_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


def make_database(url: str) -> tuple[Engine, sessionmaker[Session]]:
    parsed = make_url(url)
    if parsed.drivername.startswith("sqlite") and parsed.database not in (None, ":memory:"):
        Path(str(parsed.database)).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False, "timeout": 30}
        if parsed.drivername.startswith("sqlite")
        else {},
    )
    if parsed.drivername.startswith("sqlite"):
        # Explicit BEGIN makes savepoints participate in the outer transaction on Python 3.11.
        @event.listens_for(engine, "connect")
        def configure_sqlite(connection: SQLiteConnection, record: ConnectionPoolEntry) -> None:
            connection.isolation_level = None
            connection.execute("PRAGMA foreign_keys=ON")

        @event.listens_for(engine, "begin")
        def begin_sqlite(connection: Connection) -> None:
            connection.exec_driver_sql("BEGIN")

    Base.metadata.create_all(engine)
    return engine, sessionmaker(engine, expire_on_commit=False)
