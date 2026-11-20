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


DATA_CELL_RE = re.compile(r"\d,\d{3}|\d\.\d|^\(?\$?\d")
YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def _header_rows(rows: list[str]) -> int:
    """How many leading rows form the column header. Usually 1 ("Segment |
    2025 | 2024"), sometimes 2 (Apple: "Years ended" above a row of dates)."""
    for i, row in enumerate(rows[:3]):
        cells = row.split(" | ")
        if len(cells) >= 2:
            values = [c for c in cells[1:] if not YEAR_RE.match(c.strip())]
            is_data = any(DATA_CELL_RE.search(c.strip()) for c in values) and not any(
                re.search(r"[A-Za-z]{3,}", c) for c in values)
            return i if is_data else i + 1
    return 1


def _split_table(table: str, size: int) -> list[str]:
    """Split an oversized table between rows, never inside one. Each piece
    repeats the header row (usually the years) so its numbers keep their
    column meaning, and stays wrapped in [TABLE] ... [/TABLE]."""
    rows = [r for r in table.split("\n") if r not in ("[TABLE]", "[/TABLE]")]
    nh = _header_rows(rows)
    header, body = "\n".join(rows[:nh]), rows[nh:]
    budget = max(size - len(_words(header)), 1)
    pieces, cur, cur_len = [], [], 0
    for row in body:
        n = len(_words(row))
        if cur and cur_len + n > budget:
            pieces.append(cur); cur, cur_len = [], 0
        cur.append(row); cur_len += n
    if cur:
        pieces.append(cur)
    out = []
    for k, piece in enumerate(pieces):
        head = [header] if header else []
        if k > 0:
            head = ["(table continued)"] + head
        out.append("\n".join(["[TABLE]", *head, *piece, "[/TABLE]"]))
    return out


def _section(text: str, size: int, overlap: int) -> list[str]:
    chunks, cur, cur_len = [], [], 0
    for para in _paragraphs(text):
        n = len(_words(para))
        if n > size:  # oversized paragraph or table: flush, then split it
            if cur:
                chunks.append("\n\n".join(cur)); cur, cur_len = [], 0
            if para.startswith("[TABLE]"):
                chunks.extend(_split_table(para, size))
            else:
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
        chunks.extend(chunk_document(json.loads(path.read_text(encoding="utf-8")), strategy, size, overlap))
    return chunks


def normalize(text: str) -> str:
    text = text.lower().replace("[table]", " ").replace("[/table]", " ").replace("|", " ")
    return re.sub(r"[^a-z0-9.%$]+", " ", text).strip()
