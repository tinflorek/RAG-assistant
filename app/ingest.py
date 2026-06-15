# app/ingest.py
import os
import logging
import chromadb
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pathlib import Path

# chromadb 0.6.3 wywołuje posthog.capture() z niezgodną sygnaturą i loguje błąd
# telemetryczny przy każdym starcie klienta — wyciszamy ten konkretny logger.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# Domyślnie localhost (testy z hosta); w compose nadpisywane na chromadb/ollama
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))

ollama_client = ollama.Client(host=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")


class DocumentExistsError(ValueError):
    """Raised when ingesting a filename that is already indexed."""

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def load_document(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in (".md", ".txt"):
        return load_text(path)
    raise ValueError(f"Unsupported file type: {suffix}")

def embed(texts: list[str]) -> list[list[float]]:
    return [
        ollama_client.embeddings(model="nomic-embed-text", prompt=t)["embedding"]
        for t in texts
    ]

def get_collection():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_or_create_collection(
        "documents", metadata={"hnsw:space": "cosine"}
    )

def source_exists(collection, source: str) -> bool:
    return len(collection.get(where={"source": source}, limit=1)["ids"]) > 0

def list_documents() -> list[dict]:
    collection = get_collection()
    result = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in result["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return [{"source": source, "chunks": n} for source, n in counts.items()]

def delete_document(source: str) -> bool:
    collection = get_collection()
    if not source_exists(collection, source):
        return False
    collection.delete(where={"source": source})
    return True

def ingest(file_path: str):
    source = Path(file_path).name
    collection = get_collection()
    if source_exists(collection, source):
        raise DocumentExistsError(
            f"'{source}' is already indexed; delete it first to re-ingest."
        )

    text = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(text)

    embeddings = embed(chunks)

    collection.add(
        ids=[f"{source}_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
    )
    print(f"Ingested {len(chunks)} chunks from {source}")

if __name__ == "__main__":
    ingest("../docs/test.pdf")   # wrzuć dowolny PDF do docs/