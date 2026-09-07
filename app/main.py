from fastapi import FastAPI, UploadFile, File, HTTPException
from db import init_db, db_session
from ingestion.ingest import compute_doc_id, ingest_document, persist_document
from ingestion.chunker import chunk_document, persist_chunks

app = FastAPI(title="Fact Knowledge Layer")

@app.on_event("startup")
def startup():
    init_db()

@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    doc_id, content_hash = compute_doc_id(content)

    # dedup check FIRST, before doing any real work
    with db_session() as conn:
        existing = conn.execute(
            "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        
        
        
           
        if existing:
            return {"doc_id": existing["id"], "status": existing["status"], "note": "already ingested"}

    # new document: ingest, chunk, persist — all in one transaction
    extracted = ingest_document(file.filename, content)
    chunks = chunk_document(extracted.doc_id, extracted.pages)

    with db_session() as conn:
        persist_document(conn, file.filename, extracted)
        persist_chunks(conn, chunks)
        conn.execute("UPDATE documents SET status = 'chunked' WHERE id = ?", (extracted.doc_id,))

    return {
        "doc_id": extracted.doc_id,
        "num_pages": extracted.num_pages,
        "num_chunks": len(chunks),
        "low_text_pages": [p.page_number for p in extracted.pages if p.low_text],
        "status": "chunked",
    }


@app.get("/documents")
def list_documents():
    with db_session() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

@app.get("/documents/{doc_id}/chunks")
def get_chunks(doc_id: str):
    with db_session() as conn:
        rows = conn.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)).fetchall()
        return [dict(r) for r in rows]