"""Evaluation harness.

Eval set format (one JSON object per line in eval/questions.jsonl):
  {
    "id": "q001",
    "question": "...",
    "ticker": "AAPL",            # used for metadata filtering
    "fiscal_year": 2025,
    "answer": "...",             # reference answer ("" if unanswerable)
    "evidence": ["exact sentence or table row from the filing", ...],
    "type": "fact | numeric | table | multi_section | unanswerable",
    "answerable": true
  }

Gold evidence is stored as TEXT, not chunk IDs, because chunk IDs change every
time you change the chunking. A retrieved chunk counts as a hit if it contains
the evidence text (after normalizing whitespace/punctuation), so the same eval
set works across every chunking experiment.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .chunk import normalize
from .generate import answer as generate_answer
from .llm import LLM

JUDGE_SYSTEM = """You grade answers from a question-answering system over SEC 10-K filings. Be strict and consistent. Reply with JSON only."""

JUDGE_PROMPT = """Question: {question}

Reference answer: {reference}

System answer: {candidate}

Sources the system was given:
{sources}

Grade two things:
1. correctness: 1 if the system answer matches the reference on every key fact and number (minor wording or rounding differences are fine), 0.5 if partially correct or missing part, 0 if wrong or if it refused.
2. faithfulness: 1 if every claim in the system answer is supported by the sources, 0 if any claim is unsupported or contradicted. If the system refused, give 1.

Reply as JSON: {{"correctness": <0|0.5|1>, "faithfulness": <0|1>, "reason": "<one sentence>"}}"""


def load_questions(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def evidence_hit(chunk_text: str, evidence: str, min_coverage: float = 0.8) -> bool:
    c, e = normalize(chunk_text), normalize(evidence)
    if not e:
        return False
    if e in c:
        return True
    # fall back to token coverage, which tolerates small parsing differences
    e_tokens = e.split()
    c_tokens = set(c.split())
    return sum(t in c_tokens for t in e_tokens) / len(e_tokens) >= min_coverage


def retrieval_metrics(hits: list[dict], evidence: list[str]) -> dict:
    """Recall@k = share of evidence items found in the top-k chunks.
    MRR = 1 / rank of the first chunk containing any evidence."""
    found = [any(evidence_hit(h["text"], ev) for h in hits) for ev in evidence]
    first = next((r for r, h in enumerate(hits, 1)
                  if any(evidence_hit(h["text"], ev) for ev in evidence)), None)
    return {"recall": sum(found) / len(found) if found else 0.0,
            "mrr": 1.0 / first if first else 0.0,
            "all_found": all(found) if found else False}


def run_eval(questions: list[dict], retriever, llm: LLM | None, out_path: Path,
             settings: dict, generate: bool = True) -> dict:
    from .generate import format_sources
    rows = []
    for q in questions:
        t0 = time.time()
        hits = retriever.search(q["question"], q.get("ticker"), q.get("fiscal_year"))
        row = {"id": q["id"], "type": q.get("type", ""), "answerable": q.get("answerable", True),
               "retrieved": [h["chunk_id"] for h in hits]}
        if row["answerable"] and q.get("evidence"):
            row.update(retrieval_metrics(hits, q["evidence"]))
        row["retrieval_seconds"] = round(time.time() - t0, 3)

        if generate and llm is not None:
            t1 = time.time()
            gen = generate_answer(llm, q["question"], hits)
            row.update(gen)
            row["generation_seconds"] = round(time.time() - t1, 2)
            if row["answerable"]:
                grade = llm.complete_json(JUDGE_SYSTEM, JUDGE_PROMPT.format(
                    question=q["question"], reference=q["answer"], candidate=gen["answer"],
                    sources=format_sources(hits)))
                row.update({"correctness": float(grade["correctness"]),
                            "faithfulness": float(grade["faithfulness"]),
                            "judge_reason": grade.get("reason", "")})
        rows.append(row)
        print(f"  {q['id']}: recall={row.get('recall', '-')} mrr={row.get('mrr', '-')}"
              f" correct={row.get('correctness', '-')} refused={row.get('refused', '-')}")

    summary = summarize(rows)
    summary["settings"] = settings
    if llm is not None:
        summary["llm_usage"] = llm.usage
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    return summary


def _mean(rows, key):
    vals = [r[key] for r in rows if key in r]
    return round(statistics.mean(vals), 3) if vals else None


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    s = {
        "n_questions": len(rows),
        "recall_at_k": _mean(answerable, "recall"),
        "mrr": _mean(answerable, "mrr"),
        "correctness": _mean(answerable, "correctness"),
        "faithfulness": _mean(answerable, "faithfulness"),
        "false_refusal_rate": round(sum(r.get("refused", False) for r in answerable) / len(answerable), 3)
        if answerable and "refused" in answerable[0] else None,
        "correct_refusal_rate": round(sum(r.get("refused", False) for r in unanswerable) / len(unanswerable), 3)
        if unanswerable and "refused" in unanswerable[0] else None,
        "avg_retrieval_seconds": _mean(rows, "retrieval_seconds"),
        "avg_generation_seconds": _mean(rows, "generation_seconds"),
    }
    by_type = {}
    for t in sorted({r["type"] for r in answerable if r["type"]}):
        sub = [r for r in answerable if r["type"] == t]
        by_type[t] = {"n": len(sub), "recall_at_k": _mean(sub, "recall"),
                      "correctness": _mean(sub, "correctness")}
    s["by_type"] = by_type
    return s


def compare(results_dir: str | Path) -> str:
    """Markdown table comparing every experiment run so far."""
    cols = ["recall_at_k", "mrr", "correctness", "faithfulness",
            "correct_refusal_rate", "avg_retrieval_seconds"]
    lines = ["| experiment | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for p in sorted(Path(results_dir).glob("*.json")):
        s = json.loads(p.read_text())["summary"]
        lines.append(f"| {p.stem} | " + " | ".join(str(s.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)
