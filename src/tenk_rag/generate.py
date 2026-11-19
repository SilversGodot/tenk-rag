"""Answer a question from retrieved chunks, with citations and an explicit
"not in the documents" path."""
from __future__ import annotations

import re

from .llm import LLM

NOT_FOUND = "NOT_FOUND"

SYSTEM = f"""You answer questions about SEC 10-K filings using ONLY the numbered sources provided.

Rules:
- Every factual sentence must end with citations like [1] or [2][3] pointing to the sources that support it.
- Quote numbers exactly as they appear, including units (millions, billions, %), and say which fiscal year they refer to.
- If the sources do not contain the answer, reply with exactly: {NOT_FOUND}
- Do not use outside knowledge. Do not guess."""


def format_sources(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] ({c['header']})\n{c['text']}" for i, c in enumerate(chunks, 1))


def answer(llm: LLM, question: str, chunks: list[dict]) -> dict:
    prompt = f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"
    text = llm.complete(SYSTEM, prompt).strip()
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(chunks)})
    return {
        "answer": text,
        "refused": text.startswith(NOT_FOUND),
        "cited_chunk_ids": [chunks[n - 1]["chunk_id"] for n in cited],
    }
