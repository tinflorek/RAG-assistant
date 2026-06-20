# app/retrieval.py
"""Hybrid retrieval: dense (vector) + sparse (BM25) fused with Reciprocal Rank
Fusion, then re-ranked by a cross-encoder. Sits behind query.search()."""
import os
import re
import logging
import ollama
from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi

# Same posthog telemetry bug silenced in ingest.py/query.py.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
EMBED_MODEL = "nomic-embed-text"

# How many candidates each retriever contributes before fusion, the RRF constant,
# how many chunks survive the rerank, and the rerank-score floor. All env-tunable.
VECTOR_CANDIDATES = int(os.getenv("VECTOR_CANDIDATES", "20"))
BM25_CANDIDATES = int(os.getenv("BM25_CANDIDATES", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.0"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

ollama_client = ollama.Client(host=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")

_token_re = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _token_re.findall(text.lower())


def embed_query(text: str) -> list[float]:
    return ollama_client.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


_reranker = None


def _get_reranker() -> TextCrossEncoder:
    """Lazily load the cross-encoder (downloads the ONNX model on first use)."""
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker


# BM25 needs the whole corpus in memory; rebuild only when the chunk count changes
# (i.e. after an ingest or delete). The local corpus is small, so this is cheap.
_bm25_cache: dict = {}


def _get_bm25(collection) -> dict:
    count = collection.count()
    cached = _bm25_cache.get("v")
    if cached is None or cached["count"] != count:
        data = collection.get(include=["documents", "metadatas"])
        docs = data["documents"]
        _bm25_cache["v"] = {
            "count": count,
            "bm25": BM25Okapi([_tokenize(d) for d in docs]) if docs else None,
            "ids": data["ids"],
            "documents": docs,
            "metadatas": data["metadatas"],
        }
    return _bm25_cache["v"]


def vector_search(query: str, collection, k: int) -> list[dict]:
    res = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "id": f"{meta['source']}_{meta['chunk_index']}",
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "document": doc,
        }
        for doc, meta in zip(res["documents"][0], res["metadatas"][0])
    ]


def keyword_search(query: str, collection, k: int) -> list[dict]:
    idx = _get_bm25(collection)
    if idx["bm25"] is None:
        return []
    scores = idx["bm25"].get_scores(_tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    out = []
    for i in top:
        if scores[i] <= 0:
            continue  # no term overlap — not a real keyword match
        meta = idx["metadatas"][i]
        out.append(
            {
                "id": idx["ids"][i],
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "document": idx["documents"][i],
            }
        )
    return out


def reciprocal_rank_fusion(result_lists: list[list[dict]], rrf_k: int) -> list[dict]:
    fused: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            key = item["id"]
            if key not in fused:
                fused[key] = {"item": dict(item), "rrf": 0.0}
            fused[key]["rrf"] += 1.0 / (rrf_k + rank + 1)
    ranked = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
    return [e["item"] for e in ranked]


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []
    scores = _get_reranker().rerank(query, [c["document"] for c in candidates])
    for c, s in zip(candidates, scores):
        c["score"] = round(float(s), 4)
    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    return [c for c in ranked if c["score"] >= RERANK_THRESHOLD][:top_k]


def hybrid_search(query: str, collection, top_k: int = RERANK_TOP_K) -> list[dict]:
    """Dense + sparse → RRF → cross-encoder rerank. Returns the same shape as
    query.search(): dicts with source, chunk_index, score, document."""
    vec = vector_search(query, collection, VECTOR_CANDIDATES)
    kw = keyword_search(query, collection, BM25_CANDIDATES)
    fused = reciprocal_rank_fusion([vec, kw], RRF_K)
    return [
        {
            "source": m["source"],
            "chunk_index": m["chunk_index"],
            "score": m["score"],
            "document": m["document"],
        }
        for m in rerank(query, fused, top_k)
    ]
