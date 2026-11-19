"""Readable terminal output for retrieved chunks.

Uses plain ASCII separators so it displays correctly in any terminal and when
redirected to a file (python -m tenk_rag.cli ask ... --no-llm > results.txt).
"""
from __future__ import annotations

import textwrap

WIDTH = 100


def _found_by(h: dict) -> str:
    parts = []
    if h.get("vector_rank") is not None:
        parts.append(f"vector #{h['vector_rank']}")
    if h.get("bm25_rank") is not None:
        parts.append(f"keyword #{h['bm25_rank']}")
    if "rerank_score" in h:
        parts.append(f"rerank score {h['rerank_score']:.3f}")
    return ", ".join(parts) if parts else "-"


def _wrap_body(text: str) -> str:
    """Wrap prose to the screen width; leave table rows untouched."""
    out, in_table = [], False
    for line in text.split("\n"):
        if line == "[TABLE]":
            in_table = True
            out.append("  +-- table " + "-" * (WIDTH - 13))
            continue
        if line == "[/TABLE]":
            in_table = False
            out.append("  +" + "-" * (WIDTH - 3))
            continue
        if in_table:
            out.append("  | " + line)
        elif not line.strip():
            out.append("")
        else:
            out.extend(textwrap.wrap(line, WIDTH - 2, initial_indent="  ", subsequent_indent="  "))
    return "\n".join(out)


def format_chunk(rank: int, total: int, h: dict) -> str:
    title = f" CHUNK {rank} of {total} "
    top = "=" * 3 + title + "=" * (WIDTH - 3 - len(title))
    meta = [
        f"  Company   : {h['ticker']}   Fiscal year: {h['fiscal_year']}",
        f"  Section   : Item {h['item']} - {h['item_title']}",
        f"  Chunk ID  : {h['chunk_id']}",
        f"  Found by  : {_found_by(h)}",
        f"  Size      : {len(h['text'].split())} words{'   (contains a table)' if h.get('has_table') else ''}",
    ]
    if h.get("url"):
        meta.append(f"  Source    : {h['url']}")
    end = f" END OF CHUNK {rank} "
    return "\n".join([
        top,
        *meta,
        "-" * WIDTH,
        _wrap_body(h["text"]),
        "=" * 3 + end + "=" * (WIDTH - 3 - len(end)),
    ])


def format_results(question: str, hits: list[dict], settings: dict | None = None) -> str:
    lines = ["", "#" * WIDTH, f"  QUESTION : {question}"]
    if settings:
        lines.append(f"  SEARCH   : {settings.get('retrieval')}"
                     f"{' + rerank' if settings.get('rerank') else ''}"
                     f", chunks = {settings.get('chunk_strategy')} {settings.get('chunk_size')} words"
                     f", filters {'on' if settings.get('use_filters') else 'off'}")
    lines.append(f"  RESULTS  : {len(hits)} chunks")
    lines.append("#" * WIDTH)
    for rank, h in enumerate(hits, 1):
        lines += ["", format_chunk(rank, len(hits), h)]
    lines.append("")
    return "\n".join(lines)
