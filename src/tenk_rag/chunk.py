"""Chunking strategies.

- fixed:   sliding window of N words over each section, ignoring paragraph breaks.
- section: pack whole paragraphs into chunks of up to N words, never crossing
           an Item boundary, and keep each table in one piece where possible.

Every chunk carries metadata (ticker, year, Item) and a context header that is
prepended before embedding, e.g. "AAPL 10-K FY2025 | Item 7 MD&A".
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _words(text: str) -> list[str]:
    return text.split()


def _paragraphs(text: str) -> list[str]:
    # Keep [TABLE] ... [/TABLE] blocks together as one paragraph.
    parts, buf, in_table = [], [], False
    for line in text.split("\n"):
        if line == "[TABLE]":
            if buf:
                parts.append("\n".join(buf)); buf = []
            in_table = True
            buf.append(line)
            continue
        if line == "[/TABLE]":
            buf.append(line)
            parts.append("\n".join(buf)); buf = []
            in_table = False
            continue
        if not line.strip() and not in_table:
            if buf:
                parts.append("\n".join(buf)); buf = []
            continue
        buf.append(line)
    if buf:
        parts.append("\n".join(buf))
    return [p for p in parts if p.strip()]


def _fixed(text: str, size: int, overlap: int) -> list[str]:
    words = _words(text)
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)
            if words[i:i + size]]


def _section(text: str, size: int, overlap: int) -> list[str]:
    chunks, cur, cur_len = [], [], 0
    for para in _paragraphs(text):
        n = len(_words(para))
        if n > size:  # oversized paragraph or table: flush, then window it
            if cur:
                chunks.append("\n\n".join(cur)); cur, cur_len = [], 0
            chunks.extend(_fixed(para, size, overlap))
            continue
        if cur_len + n > size and cur:
            chunks.append("\n\n".join(cur))
            # carry the last paragraph forward as overlap if it is small enough
            tail = cur[-1] if overlap and len(_words(cur[-1])) <= overlap else None
            cur, cur_len = ([tail], len(_words(tail))) if tail else ([], 0)
        cur.append(para)
        cur_len += n
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_document(doc: dict, strategy: str, size: int, overlap: int) -> list[dict]:
    splitter = {"fixed": _fixed, "section": _section}[strategy]
    out = []
    for sec in doc["sections"]:
        for n, text in enumerate(splitter(sec["text"], size, overlap)):
            header = f"{doc['ticker']} 10-K FY{doc['fiscal_year']} | Item {sec['item']} {sec['title']}"
            out.append({
                "chunk_id": f"{doc['doc_id']}_item{sec['item']}_{n:03d}",
                "doc_id": doc["doc_id"],
                "ticker": doc["ticker"],
                "fiscal_year": doc["fiscal_year"],
                "item": sec["item"],
                "item_title": sec["title"],
                "header": header,
                "text": text,
                "has_table": "[TABLE]" in text,
                "url": doc.get("url", ""),
            })
    return out


def chunk_corpus(parsed_dir: str | Path, strategy: str, size: int, overlap: int) -> list[dict]:
    chunks = []
    for path in sorted(Path(parsed_dir).glob("*.json")):
        chunks.extend(chunk_document(json.loads(path.read_text()), strategy, size, overlap))
    return chunks


def normalize(text: str) -> str:
    text = text.lower().replace("[table]", " ").replace("[/table]", " ").replace("|", " ")
    return re.sub(r"[^a-z0-9.%$]+", " ", text).strip()
