from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import select

from app.database import Chunk, Document
from app.documents import DocumentError, chunk_text, extract_text


def text_pdf() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 50 700 Td (Python skills for the internship.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_fixed_overlap_and_tail():
    text = "a" * 100 + "b" * 80 + "c" * 20
    chunks = chunk_text(text, "fixed", 100, 20)
    assert [len(c) for c in chunks] == [100, 100, 40]
    assert chunks[0][-20:] == chunks[1][:20]
    assert chunks[0] + "".join(c[20:] for c in chunks[1:]) == text


def test_paragraph_boundaries_and_long_fallback():
    text = "a" * 40 + "\n\n" + "b" * 40 + "\n\n" + "c" * 130
    assert chunk_text(text, "paragraph", 100, 0) == [
        "a" * 40 + "\n\n" + "b" * 40,
        "c" * 100,
        "c" * 30,
    ]


@pytest.mark.parametrize(
    "strategy,size,overlap", [("fixed", 100, 100), ("fixed", 99, 0), ("paragraph", 100, 10)]
)
def test_invalid_chunk_options(strategy, size, overlap):
    with pytest.raises(DocumentError):
        chunk_text("hello", strategy, size, overlap)


@pytest.mark.parametrize(
    "filename,data",
    [
        ("a.exe", b"hello"),
        ("a.txt", b"\xff"),
        ("a.txt", b" \n"),
        ("a.txt", b"a\x00b"),
        ("a.pdf", b"not pdf"),
        ("a.pdf", b"%PDF-broken"),
    ],
)
def test_invalid_documents(filename, data):
    with pytest.raises(DocumentError):
        extract_text(filename, data, 1000)


def test_pdf_extraction_and_ingestion(client, service):
    pdf = text_pdf()
    assert "Python skills" in extract_text("sample.pdf", pdf, 1000)
    result = client.post("/api/v1/documents", files={"file": ("sample.pdf", pdf)})
    assert result.status_code == 201
    with service.sessions() as db:
        assert db.scalar(select(Document)).status == "ready"
        assert db.scalar(select(Chunk)).char_count > 0


def test_empty_and_encrypted_pdf():
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    empty = BytesIO()
    writer.write(empty)
    with pytest.raises(DocumentError, match="No text"):
        extract_text("a.pdf", empty.getvalue(), 1000)
    writer.encrypt("password")
    encrypted = BytesIO()
    writer.write(encrypted)
    with pytest.raises(DocumentError, match="Encrypted"):
        extract_text("a.pdf", encrypted.getvalue(), 1000)


def test_ingest_strategies_and_metadata(client, service):
    for strategy in ("fixed", "paragraph"):
        result = client.post(
            "/api/v1/documents",
            files={"file": ("../sample.txt", b"Python skills")},
            data={"strategy": strategy},
        )
        assert result.status_code == 201
        assert result.json()["filename"] == "sample.txt"
        assert result.json()["strategy"] == strategy
    with service.sessions() as db:
        assert len(db.scalars(select(Document)).all()) == 2
    assert service.vectors.client.count(service.vectors.collection).count == 2


def test_upload_limit_and_bad_strategy(client, service):
    service.settings.max_upload_bytes = 4
    assert client.post("/api/v1/documents", files={"file": ("a.txt", b"12345")}).status_code == 413
    assert (
        client.post(
            "/api/v1/documents", files={"file": ("a.txt", b"123")}, data={"strategy": "unknown"}
        ).status_code
        == 422
    )


def test_embedding_failure_marks_document_failed(client, service):
    service.llm.fail_embedding = True
    result = client.post("/api/v1/documents", files={"file": ("a.txt", b"Python")})
    assert result.status_code == 503
    with service.sessions() as db:
        assert db.scalar(select(Document)).status == "failed"
    assert service.vectors.client.count(service.vectors.collection).count == 0


def test_partial_vector_failure_cleans_up(client, service, monkeypatch):
    original = service.vectors.insert

    def fail_after_insert(*args):
        original(*args)
        raise RuntimeError("Simulated partial write")

    monkeypatch.setattr(service.vectors, "insert", fail_after_insert)
    assert client.post("/api/v1/documents", files={"file": ("a.txt", b"Python")}).status_code == 503
    assert service.vectors.client.count(service.vectors.collection).count == 0
