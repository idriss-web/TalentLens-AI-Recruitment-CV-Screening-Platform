from __future__ import annotations

import io
import re
from pathlib import Path


class CVParseError(Exception):
    pass


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def _read_docx(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")
    try:
        if ext == "pdf":
            text = _read_pdf(data)
        elif ext == "docx":
            text = _read_docx(data)
        elif ext in {"txt", "md"}:
            text = data.decode("utf-8", errors="ignore")
        else:
            raise CVParseError(f"Unsupported file type: .{ext}")
    except CVParseError:
        raise
    except Exception as exc:
        raise CVParseError(f"Could not read {filename}: {exc}") from exc

    text = _clean(text)
    if len(text) < 30:
        raise CVParseError(
            f"{filename}: almost no text found (scanned image PDF? OCR is not included)."
        )
    return text
