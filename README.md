# RAG Assistant

A small, fully local **Retrieval-Augmented Generation** service. Upload documents
(PDF / Markdown / text), and ask questions that are answered **grounded in those
documents**, with inline source citations. Everything runs on your machine — no
external APIs. Embeddings and the LLM are served by [Ollama](https://ollama.com/),
vectors are stored in [Chroma](https://www.trychroma.com/), and the HTTP layer is
[FastAPI](https://fastapi.tiangolo.com/).

## How it works

```
          ┌──────────── ingest ────────────┐         ┌─────────────── query (agent loop) ──────────────┐
 PDF/MD/TXT → chunk (512/50) → embed ──────►  Chroma  ◄── search_documents(query) ──┐                    │
                              (nomic-embed-text)  │                                  │  LLM (qwen2.5:3b)  │
                                                  └──── chunks ──────────────────────┘  decides & loops  │
                                                                                        → answer + cites │
                                                                       └─────────────────────────────────┘
```

1. **Ingest** — a document is loaded, split into ~512-character chunks (50-char
   overlap), embedded with `nomic-embed-text`, and stored in a Chroma collection
   named `documents` (cosine similarity).
2. **Query (agentic)** — instead of a single fixed retrieval, `qwen2.5:3b` is given
   two tools and drives retrieval itself: `search_documents(query)` and `list_documents()`
   (see what is indexed). The model searches as many times as it needs — refining or
   issuing separate queries for multi-part questions — then answers grounded only in
   what it retrieved, citing sources as `[filename, chunk N]`. The loop is capped at
   `MAX_STEPS` (5) tool-calling rounds, after which a final answer is forced. Sources
   from every search are aggregated and deduplicated across the run.
3. **Hybrid retrieval + reranking** — `search_documents` is backed by hybrid search:
   dense vector search **and** BM25 keyword search each return candidates, which are
   merged with Reciprocal Rank Fusion and then re-ranked by a cross-encoder
   (`fastembed`, ONNX) that scores each `(query, chunk)` pair. The top few survivors
   (above a rerank-score floor) become the context. This catches both semantic matches
   (vectors) and exact terms like model/error codes (BM25). Set `RETRIEVAL_MODE=vector`
   to fall back to the original dense-only path.
4. **Citation guard** — after the answer is produced, every `[filename, chunk N]`
   citation it wrote is checked against the chunks actually retrieved during the run.
   Any citation referencing a chunk that was never retrieved is returned in
   `unsupported_citations` (and flagged in the UI). The answer text is left untouched.

## Stack

| Component        | Choice                          |
| ---------------- | ------------------------------- |
| API              | FastAPI + Uvicorn               |
| Embeddings       | `nomic-embed-text` (via Ollama) |
| LLM              | `qwen2.5:3b` (via Ollama)       |
| Keyword search   | BM25 (`rank-bm25`)              |
| Reranker         | cross-encoder (`fastembed`, ONNX) |
| Vector store     | Chroma `0.6.3`                  |
| Package manager  | [uv](https://docs.astral.sh/uv/) |
| Python           | 3.12                            |

## Project layout

```
.
├── app/
│   ├── main.py        # FastAPI app: web UI + /health, /ingest, /query, /documents
│   ├── index.html     # self-contained web UI served at /
│   ├── ingest.py      # load → chunk → embed → store in Chroma
│   ├── retrieval.py   # hybrid search: vector + BM25 → RRF → cross-encoder rerank
│   └── query.py       # agent loop: LLM drives search_documents/list_documents tools → answer
├── eval/              # evaluation harness: fixture doc, golden Q/A, run_eval.py
├── docs/              # documents to ingest (mounted into the container)
├── Dockerfile         # uv-based image for the API
├── docker-compose.yml # ollama + chromadb + rag-api
├── pyproject.toml     # dependencies (managed by uv)
└── uv.lock            # pinned, reproducible lockfile
```

## API

| Method | Path      | Description                                                        |
| ------ | --------- | ----------------------------------------------------------------- |
| GET    | `/`                    | Web UI — upload documents, ask questions, manage indexed docs    |
| GET    | `/health`              | Liveness check → `{"status": "ok"}`                              |
| POST   | `/ingest`              | Upload a `.pdf` / `.md` / `.txt` file (multipart) and index it. Re-uploading an already-indexed filename returns **409** — delete it first |
| GET    | `/documents`           | List indexed documents → `[{"source": ..., "chunks": N}, ...]`   |
| DELETE | `/documents/{filename}`| Remove a document — deletes its chunks **and** the uploaded file from `docs/` (404 if not indexed) |
| POST   | `/query`               | Body `{"question": "..."}` → `{"answer": ..., "sources": [...], "unsupported_citations": [...]}`. The LLM agentically calls `search_documents`/`list_documents` (hybrid retrieval + rerank) to gather context; if nothing relevant is found, the answer says so and `sources` is empty. `unsupported_citations` lists any `[file, chunk N]` the model cited that wasn't actually retrieved |

A browser UI is served at `/` (the simplest way to use the app), and interactive
API docs are at `/docs` (Swagger) once the API is running.

## Running with Docker (recommended)

```bash
# Start the whole stack — that's it.
docker compose up -d --build
```

On first run the `ollama-init` service automatically pulls both models before the
API starts, so no manual `ollama pull` is needed. Then **open
<http://localhost:8080> in your browser** to upload documents and ask questions.

Prefer the command line? The same endpoints are on host port 8080:

```bash
curl -F "file=@docs/test.pdf" http://localhost:8080/ingest
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

> The first run downloads ~2 GB of models; they live in the `ollama_data` named
> volume, so later starts are fast and need no re-pull. Use `docker compose down`
> to stop; avoid `down -v`, which **wipes the volumes** (models and the vector
> store).

## Running locally (without Docker)

Requires [uv](https://docs.astral.sh/uv/) and a running Ollama with both models
pulled (`ollama pull nomic-embed-text && ollama pull qwen2.5:3b`).

```bash
# Install dependencies into a project venv
uv sync

# Start a local Chroma server (separate terminal)
uv run chroma run --host localhost --port 8000 --path ./chroma_data

# Ingest a document (scripts run from the app/ directory)
cd app && uv run python ingest.py            # ingests ../docs/test.pdf

# Ask a question from the CLI
uv run python query.py "What is RAG?"

# Or run the API
uv run uvicorn main:app --reload             # http://localhost:8000
```

> When running the API locally, set `DOCS_DIR` to a real path (it defaults to the
> container path `/app/docs`), e.g. `DOCS_DIR=../docs uv run uvicorn main:app`.

## Configuration

The scripts read these environment variables (with host-friendly defaults), so the
same code works both locally and inside the compose network:

| Variable      | Default     | Purpose                          |
| ------------- | ----------- | -------------------------------- |
| `CHROMA_HOST` | `localhost` | Chroma server host               |
| `CHROMA_PORT` | `8000`      | Chroma server port               |
| `OLLAMA_HOST` | `localhost` | Ollama host                      |
| `OLLAMA_PORT` | `11434`     | Ollama port                      |
| `DOCS_DIR`    | `/app/docs` | Where uploaded files are written |
| `SIMILARITY_THRESHOLD` | `0.4` | Minimum cosine similarity for a chunk in the `vector`-mode fallback |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` (vector + BM25 + rerank) or `vector` (dense-only fallback) |
| `VECTOR_CANDIDATES` | `20` | Dense candidates fetched before fusion |
| `BM25_CANDIDATES` | `20` | Keyword candidates fetched before fusion |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RERANK_TOP_K` | `3` | Chunks kept after reranking |
| `RERANK_THRESHOLD` | `0.0` | Minimum cross-encoder score for a chunk to be kept |
| `RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | fastembed cross-encoder model |
| `FASTEMBED_CACHE_PATH` | (temp dir) | Where the reranker model is cached (pre-baked into the Docker image) |

## Evaluation

A small harness in `eval/` measures retrieval and answer quality against a fixed,
committed ground-truth set (`eval/fixtures/handbook.md` + `eval/golden.jsonl`), so
changes to retrieval can be compared rather than guessed at. It reports **hit-rate@k**
and **MRR** for retrieval, plus an LLM-as-judge **faithfulness** and **relevance** score
(1–5; a 3B judge is a coarse signal). Requires Chroma + Ollama running.

```bash
# Hybrid retrieval (default)
uv run python eval/run_eval.py

# Compare against the dense-only baseline
RETRIEVAL_MODE=vector uv run python eval/run_eval.py
```

Each run prints a per-question table and a summary, and writes `eval/report.json`.

## Notes

- **Chroma is pinned to `0.6.3`** to match the Python client. A newer server image
  breaks the `0.6.3` client (`KeyError: '_type'`).
- The collection uses **cosine** similarity, so `query.py` reports a similarity
  score in `[0, 1]` (computed as `1 - cosine_distance`).
- Chroma is **not** exposed on the host — only `rag-api` reaches it, over the
  internal compose network as `chromadb:8000`. Keeping the vector store
  network-internal avoids unnecessary attack surface. (For one-off debugging you
  can publish it via a `docker-compose.override.yml` rather than the base file.)
