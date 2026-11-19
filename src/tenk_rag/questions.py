"""Draft candidate eval questions with an LLM, for YOU to review.

LLM-written questions are a starting point, not ground truth. The script
checks that each evidence quote really appears in the source chunk and drops
the ones that don't. Review every kept question before moving it into
eval/questions.jsonl, and write the hard ones (multi-section, unanswerable,
cross-year) yourself.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .evaluate import evidence_hit
from .llm import LLM

SYSTEM = "You write evaluation questions for a question-answering system over SEC 10-K filings. Reply with JSON only."

PROMPT = """Below is a passage from {header}.

{text}

Write {n} questions an analyst might ask that this passage answers. Rules:
- Each question must name the company and fiscal year, as a real user would.
- Prefer specific facts and numbers over vague summaries.
- If the passage contains a table, make at least one question about a number in the table.
- "evidence" must be copied EXACTLY from the passage (one sentence or one table row).

Reply as a JSON list: [{{"question": "...", "answer": "...", "evidence": "...", "type": "fact|numeric|table"}}]"""

# Items most analysts ask about; skip boilerplate sections.
USEFUL_ITEMS = {"1", "1A", "1C", "2", "3", "5", "7", "7A", "8"}


def draft_questions(chunks: list[dict], llm: LLM, n_chunks: int, per_chunk: int,
                    out_path: Path, seed: int = 0) -> int:
    rng = random.Random(seed)
    pool = [c for c in chunks if c["item"] in USEFUL_ITEMS and len(c["text"].split()) > 80]
    # oversample table chunks: they are where RAG systems fail most
    tables = [c for c in pool if c["has_table"]]
    sample = rng.sample(tables, min(len(tables), n_chunks // 3))
    rest = [c for c in pool if c not in sample]
    sample += rng.sample(rest, min(len(rest), n_chunks - len(sample)))

    kept = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        for c in sample:
            try:
                items = llm.complete_json(SYSTEM, PROMPT.format(header=c["header"], text=c["text"], n=per_chunk))
            except Exception as e:
                print(f"[skip] {c['chunk_id']}: {e}")
                continue
            for it in items:
                if not evidence_hit(c["text"], it.get("evidence", ""), min_coverage=0.95):
                    continue  # model paraphrased instead of quoting: drop it
                kept += 1
                f.write(json.dumps({
                    "id": f"cand_{c['chunk_id']}_{kept}",
                    "question": it["question"], "ticker": c["ticker"],
                    "fiscal_year": c["fiscal_year"], "answer": it["answer"],
                    "evidence": [it["evidence"]], "type": it.get("type", "fact"),
                    "answerable": True, "source_chunk": c["chunk_id"], "reviewed": False,
                }) + "\n")
    return kept
