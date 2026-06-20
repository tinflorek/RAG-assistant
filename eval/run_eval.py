"""RAG evaluation harness.

Runs a fixed golden dataset against the live retrieval + query pipeline and reports:
  - Retrieval: hit-rate@k and MRR (did the correct passage surface in the top-k?)
  - Answers:   LLM-as-judge faithfulness + relevance (1-5; a 3B judge is a COARSE signal)

Requires Chroma and Ollama running (same services the app uses). Run from anywhere:
    uv run python eval/run_eval.py
"""
import json
import re
import sys
import time
from pathlib import Path

# The app uses flat modules that must be importable as `import query`/`import ingest`.
EVAL_DIR = Path(__file__).resolve().parent
APP_DIR = EVAL_DIR.parent / "app"
sys.path.insert(0, str(APP_DIR))

import ingest  # noqa: E402
import query  # noqa: E402

FIXTURE = EVAL_DIR / "fixtures" / "handbook.md"
GOLDEN = EVAL_DIR / "golden.jsonl"
REPORT = EVAL_DIR / "report.json"
TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 3

JUDGE_PROMPT = """You are grading a question-answering system.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{reference}

RETRIEVED CONTEXT (what the system was allowed to use):
{context}

SYSTEM ANSWER:
{answer}

Score two things from 1 to 5:
- "faithfulness": is every claim in the SYSTEM ANSWER supported by the RETRIEVED CONTEXT?
  5 = fully grounded, 1 = mostly fabricated.
- "relevance": does the SYSTEM ANSWER actually answer the QUESTION and agree with the
  REFERENCE ANSWER? 5 = correct and on-point, 1 = wrong or off-topic.

Reply with ONLY a JSON object: {{"faithfulness": <int>, "relevance": <int>}}"""


def normalize(text: str) -> str:
    """Collapse whitespace so must_contain snippets survive chunk line-wrapping."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load_golden() -> list[dict]:
    with GOLDEN.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def ensure_fixture_ingested():
    try:
        ingest.ingest(str(FIXTURE))
    except ingest.DocumentExistsError:
        pass  # already indexed from a previous run — fine


def judge(question: str, reference: str, context: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference, context=context, answer=answer
    )
    resp = query.ollama_client.chat(
        model=query.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    raw = resp["message"]["content"]
    try:
        data = json.loads(raw)
        return {
            "faithfulness": int(data["faithfulness"]),
            "relevance": int(data["relevance"]),
        }
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return {"faithfulness": None, "relevance": None}


def evaluate() -> dict:
    ensure_fixture_ingested()
    collection = ingest.get_collection()
    golden = load_golden()

    rows = []
    for item in golden:
        snippet = normalize(item["must_contain"])
        matches = query.search(item["question"], collection, top_k=TOP_K)

        rank = next(
            (i + 1 for i, m in enumerate(matches) if snippet in normalize(m["document"])),
            None,
        )
        context = "\n\n".join(m["document"] for m in matches) or "(nothing retrieved)"

        result = query.query(item["question"])
        scores = judge(item["question"], item["reference_answer"], context, result.answer)

        rows.append(
            {
                "question": item["question"],
                "hit": rank is not None,
                "rank": rank,
                "rr": 1 / rank if rank else 0.0,
                "faithfulness": scores["faithfulness"],
                "relevance": scores["relevance"],
                "answer": result.answer,
            }
        )

    n = len(rows)
    faith = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    rel = [r["relevance"] for r in rows if r["relevance"] is not None]
    summary = {
        "n": n,
        "top_k": TOP_K,
        f"hit_rate@{TOP_K}": round(sum(r["hit"] for r in rows) / n, 4),
        "mrr": round(sum(r["rr"] for r in rows) / n, 4),
        "avg_faithfulness": round(sum(faith) / len(faith), 3) if faith else None,
        "avg_relevance": round(sum(rel) / len(rel), 3) if rel else None,
    }
    return {"summary": summary, "rows": rows}


def main():
    report = evaluate()
    s = report["summary"]

    print(f"\n=== RETRIEVAL & ANSWER EVAL (top_k={s['top_k']}, n={s['n']}) ===\n")
    print(f"{'hit':>4} {'rank':>4} {'faith':>6} {'rel':>4}  question")
    for r in report["rows"]:
        hit = "✓" if r["hit"] else "✗"
        rank = r["rank"] if r["rank"] else "-"
        faith = r["faithfulness"] if r["faithfulness"] is not None else "?"
        rel = r["relevance"] if r["relevance"] is not None else "?"
        print(f"{hit:>4} {rank:>4} {faith:>6} {rel:>4}  {r['question'][:60]}")

    hit_key = f"hit_rate@{s['top_k']}"
    print("\n=== SUMMARY ===")
    print(f"  {hit_key}   : {s[hit_key]}")
    print(f"  MRR            : {s['mrr']}")
    print(f"  faithfulness   : {s['avg_faithfulness']}  (1-5, coarse 3B-judge signal)")
    print(f"  relevance      : {s['avg_relevance']}  (1-5, coarse 3B-judge signal)")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **report}
    REPORT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {REPORT.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
