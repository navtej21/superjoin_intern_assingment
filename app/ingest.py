"""PDF ingestion: hashing, storage, page text, and printed-page recovery.

Deliberately generic — nothing here knows about Delhivery, the IMF, or any
particular document. It knows "PDF in, pages with text out," which is what lets
the same code run unmodified against a reviewer's own PDFs.

Two properties matter downstream and are established here. First, a document's
identity is the hash of its bytes, so re-ingesting the same file is a no-op and
the pipeline is safe to re-run. Second, a page keeps its text exactly as the
extractor produced it — grounding compares against *this* string, so any tidying
done at storage time would quietly widen what counts as verified.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.chunker import Chunk
from app.config import LOW_TEXT_CHAR_THRESHOLD, UPLOAD_DIR
from app.pagenum import infer_printed_pages


@dataclass
class PageText:
    page_number: int          # 1-indexed, matches PDF page order
    text: str
    char_count: int
    low_text: bool
    printed_page: str | None  # as printed in the document, e.g. "42-43"


@dataclass
class ExtractedDocument:
    doc_id: str
    content_hash: str
    file_path: Path
    num_pages: int
    pages: list[PageText]

    @property
    def low_text_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.low_text]


def compute_doc_id(content: bytes) -> tuple[str, str]:
    """Return (doc_id, full_content_hash). doc_id is a short, stable,
    human-inspectable prefix; dedup is keyed on the full hash."""
    full_hash = hashlib.sha256(content).hexdigest()
    return full_hash[:16], full_hash


def extract_pages(pdf_path: Path) -> list[PageText]:
    """Extract text per page and recover each page's printed number.

    A page's own text is what evidence quotes are verified against, so we keep
    it as close to the extractor's reading-order output as we can, and flag —
    never silently drop — pages that yield almost nothing. Those flags are the
    honest record of where a text-only pipeline cannot see, which is exactly
    what the assignment's fourth case asks us to report.
    """
    with pymupdf.open(pdf_path) as doc:
        raw_texts = [(page.get_text("text") or "") for page in doc]

    labels = infer_printed_pages(raw_texts)

    pages: list[PageText] = []
    for index, raw in enumerate(raw_texts):
        stripped = raw.strip()
        label = labels[index]
        pages.append(
            PageText(
                page_number=index + 1,
                text=raw,
                char_count=len(stripped),
                low_text=len(stripped) < LOW_TEXT_CHAR_THRESHOLD,
                printed_page=label.label if label else None,
            )
        )
    return pages


def save_upload(filename: str, content: bytes, doc_id: str) -> Path:
    suffix = Path(filename).suffix or ".pdf"
    dest = UPLOAD_DIR / f"{doc_id}{suffix}"
    dest.write_bytes(content)
    return dest


def ingest_document(filename: str, content: bytes) -> ExtractedDocument:
    """Hash, store, and extract one PDF's bytes. Does not touch the database —
    callers decide how to persist and dedup."""
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


def persist_document(
    conn: sqlite3.Connection, filename: str, extracted: ExtractedDocument
) -> None:
    """Write a freshly extracted document and its pages. The caller is
    responsible for having checked that content_hash is new."""
    conn.execute(
        """INSERT INTO documents (id, filename, content_hash, file_path, num_pages, status)
           VALUES (?, ?, ?, ?, ?, 'extracted')""",
        (
            extracted.doc_id,
            filename,
            extracted.content_hash,
            str(extracted.file_path),
            extracted.num_pages,
        ),
    )
    conn.executemany(
        """INSERT INTO pages (doc_id, page_number, printed_page, text, char_count, low_text)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (
                extracted.doc_id,
                p.page_number,
                p.printed_page,
                p.text,
                p.char_count,
                int(p.low_text),
            )
            for p in extracted.pages
        ],
    )


def persist_chunks(
    conn: sqlite3.Connection, doc_id: str, chunks: list[Chunk]
) -> None:
    """Write a document's chunks and mark it chunked.

    The per-chunk content hash is the cache key for extraction: re-running the
    pipeline, or ingesting a document that shares boilerplate with one already
    seen, must not re-pay for the same model call.
    """
    conn.executemany(
        """INSERT OR REPLACE INTO chunks
             (id, doc_id, chunk_index, page_start, page_end, char_offset,
              text, token_count, content_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                f"{doc_id}:{c.chunk_index}",
                doc_id,
                c.chunk_index,
                c.page_start,
                c.page_end,
                c.char_offset,
                c.text,
                c.token_count,
                hashlib.sha256(c.text.encode("utf-8")).hexdigest(),
            )
            for c in chunks
        ],
    )
    conn.execute("UPDATE documents SET status = 'chunked' WHERE id = ?", (doc_id,))