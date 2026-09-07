from __future__ import annotations

from app.ingestion.ingest import ingest_document


content = open(
    r"C:\Users\Navtej\Desktop\starter-datasets\india-macroeconomy\03-imf-india-2025-article-iv-excerpt.pdf",
    "rb"
).read()


ext = ingest_document(
    "03-imf-india-2025-article-iv-excerpt.pdf",
    content
)


print(
    ext.doc_id,
    ext.num_pages,
    [p.page_number for p in ext.pages if p.low_text]
)