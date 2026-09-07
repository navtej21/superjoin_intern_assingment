from __future__ import annotations
from dataclasses import dataclass
import hashlib
import sqlite3
from pathlib import Path


"""PDF ingestion: hashing, storage, and page-level text extraction.

Deliberately generic — nothing here knows about Delhivery, the IMF, or any
particular document. It only knows "PDF in, pages with text out," which is
what lets the same code run unmodified against a reviewer's own PDFs.
"""

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from pymupdf import *

from config import LOW_TEXT_CHAR_THRESHOLD, UPLOAD_DIR


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str
    char_count: int
    low_text: bool


@dataclass
class ExtractedDocument:
    doc_id: str
    content_hash: str
    file_path: Path
    num_pages: int
    pages: list[PageText]


def compute_doc_id(content: bytes) -> tuple[str, str]:
    """Return (doc_id, full_content_hash). doc_id is a short, stable,
    human-inspectable prefix; full_content_hash is what dedup is keyed on."""
    full_hash = hashlib.sha256(content).hexdigest()
    return full_hash[:16], full_hash


def extract_pages(pdf_path: Path) -> list[PageText]:
    """Extract text per page. A page's own text is what evidence quotes are
    verified against later (Phase B's grounding gate), so we keep this as
    close to the PDF's actual reading-order text as PyMuPDF gives us for
    free, and flag — never silently drop — pages that yield almost nothing.
    """
    pages: list[PageText] = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            raw = page.get_text("text") or ""
            stripped = raw.strip()
            pages.append(
                PageText(
                    page_number=i + 1,
                    text=raw,
                    char_count=len(stripped),
                    low_text=len(stripped) < LOW_TEXT_CHAR_THRESHOLD,
                )
            )
    return pages


def save_upload(filename: str, content: bytes, doc_id: str) -> Path:
    suffix = Path(filename).suffix or ".pdf"
    dest = UPLOAD_DIR / f"{doc_id}{suffix}"
    dest.write_bytes(content)
    return dest


def ingest_document(filename: str, content: bytes) -> ExtractedDocument:
    """Full Phase-A ingest of one PDF's bytes: hash, store, extract pages.
    Does not touch the database — callers decide how to persist/dedup."""
    doc_id, content_hash = compute_doc_id(content)
    file_path = save_upload(filename, content, doc_id)
    pages = extract_pages(file_path)
    return ExtractedDocument(
        doc_id=doc_id,
        content_hash=content_hash,
        file_path=file_path,
        num_pages=len(pages),
        pages=pages,
    )


def persist_document(conn: sqlite3.Connection, filename: str, extracted: ExtractedDocument) -> None:
    """Write a freshly extracted document + its pages into the DB.
    Caller is responsible for having already checked content_hash is new."""
    conn.execute(
        """INSERT INTO documents (id, filename, content_hash, file_path, num_pages, status)
           VALUES (?, ?, ?, ?, ?, 'extracted')""",
        (extracted.doc_id, filename, extracted.content_hash, str(extracted.file_path), extracted.num_pages),
    )
    conn.executemany(
        """INSERT INTO pages (doc_id, page_number, text, char_count, low_text)
           VALUES (?, ?, ?, ?, ?)""",
        [
            (extracted.doc_id, p.page_number, p.text, p.char_count, int(p.low_text))
            for p in extracted.pages
        ],
    )