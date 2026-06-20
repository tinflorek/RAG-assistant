# Build context is the repo root (see docker-compose.yml), so paths below are
# relative to the project root, not the app/ directory.
FROM python:3.12-slim

# Bring in the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# uv settings: install into a project venv, copy (not symlink) for a clean image
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# Install dependencies first, in their own cached layer (no app code yet)
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy application code into the workdir root (flat layout: modules import each
# other as `from ingest import ...`, so they must sit directly on the path)
COPY app/ ./

# Put the venv on PATH so `uvicorn` resolves
ENV PATH="/app/.venv/bin:$PATH"

# Pre-bake the cross-encoder reranker into the image so the first /query isn't
# slow or dependent on network access at runtime. Cached at a fixed path that the
# app reads back via FASTEMBED_CACHE_PATH.
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
RUN python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
