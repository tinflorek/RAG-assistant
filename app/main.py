# app/main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
from pathlib import Path

from ingest import ingest, list_documents, delete_document, DocumentExistsError
from query import query, QueryResult

app = FastAPI(title="RAG Assistant")

DOCS_DIR = Path(os.getenv("DOCS_DIR", "/app/docs"))
DOCS_DIR.mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".md", ".txt")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    dest = DOCS_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        ingest(str(dest))
    except DocumentExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ingested", "file": file.filename}


@app.get("/documents")
def documents():
    return list_documents()


@app.delete("/documents/{filename}")
def delete_document_endpoint(filename: str):
    if not delete_document(filename):
        raise HTTPException(status_code=404, detail=f"'{filename}' is not indexed")
    (DOCS_DIR / Path(filename).name).unlink(missing_ok=True)
    return {"status": "deleted", "file": filename}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result: QueryResult = query(req.question)
    return QueryResponse(answer=result.answer, sources=result.sources)
