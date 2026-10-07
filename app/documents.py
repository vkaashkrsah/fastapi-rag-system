"""Text extraction and two explicit, deterministic chunking strategies."""

import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from app.schemas import Strategy


class DocumentError(ValueError):
    pass


def extract_text(filename: str, data: bytes, max_chars: int) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentError("TXT files must be UTF-8 encoded") from exc
        if "\x00" in text:
            raise DocumentError("TXT file contains binary data")
    elif suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise DocumentError("Invalid PDF header")
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise DocumentError("Encrypted PDFs are not supported")
            if len(reader.pages) > 200:
                raise DocumentError("PDF exceeds the 200-page limit")
            pages: list[str] = []
            total = 0
            for page in reader.pages:
                content = page.extract_text() or ""
                total += len(content)
                if total > max_chars:
                    raise DocumentError("Extracted text exceeds the configured limit")
                pages.append(content)
            text = "\n\n".join(pages)
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError("PDF could not be parsed") from exc
    else:
        raise DocumentError("Only .pdf and .txt files are supported")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise DocumentError("No text found; scanned PDFs need OCR before upload")
    if len(text) > max_chars:
        raise DocumentError("Extracted text exceeds the configured limit")
    return text


def chunk_text(text: str, strategy: Strategy, size: int, overlap: int) -> list[str]:
    if size < 100 or size > 2000:
        raise DocumentError("chunk_size must be between 100 and 2000 characters")
    if overlap < 0 or overlap >= size:
        raise DocumentError("overlap must satisfy 0 <= overlap < chunk_size")
    if strategy == "fixed":
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + size])
            if start + size >= len(text):
                break
            start += size - overlap
        return [part for part in chunks if part.strip()]
    if overlap:
        raise DocumentError("paragraph strategy requires overlap=0")
    chunks = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            # Oversized paragraphs use bounded windows without overlap.
            chunks.extend(paragraph[i : i + size] for i in range(0, len(paragraph), size))
        elif current and len(current) + 2 + len(paragraph) > size:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks
