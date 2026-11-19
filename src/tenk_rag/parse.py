"""Turn 10-K HTML into clean text split by Item (1, 1A, 7, 7A, 8, ...).

Tables are kept as pipe-delimited rows so numbers stay next to their labels,
which matters a lot for financial questions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

ITEM_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures", "5": "Market for Registrant's Common Equity",
    "6": "Reserved", "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation", "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions", "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules", "16": "Form 10-K Summary",
}

ITEM_RE = re.compile(r"^\s*item\s*(\d{1,2}[a-c]?)\s*[.:\-—–]?\s*(.*)$", re.IGNORECASE)
BLOCK_TAGS = ["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"]


def _table_to_text(table) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in {"$", "%", ")"}]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    # Inline XBRL puts hidden metadata in display:none blocks.
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:header$", re.I)):
        tag.decompose()
    for table in soup.find_all("table"):
        text = _table_to_text(table)
        table.replace_with(soup.new_string(f"\n[TABLE]\n{text}\n[/TABLE]\n" if text else "\n"))
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_after("\n")
    text = soup.get_text()
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    out, blank = [], 0
    for ln in lines:
        if not ln:
            blank += 1
            if blank == 1:
                out.append("")
            continue
        blank = 0
        out.append(ln)
    return "\n".join(out).strip()


def split_items(text: str) -> list[dict]:
    """Split text into Item sections.

    The table of contents also lists every Item, so each Item heading appears
    at least twice. For each Item we keep the occurrence whose section body is
    longest, which is the real section rather than the TOC line.
    """
    lines = text.split("\n")
    marks = []
    for i, ln in enumerate(lines):
        if len(ln) > 200:
            continue
        m = ITEM_RE.match(ln)
        if m and m.group(1).upper() in ITEM_TITLES:
            marks.append((i, m.group(1).upper()))
    if not marks:
        return [{"item": "ALL", "title": "Full document", "text": text}]

    candidates: dict[str, tuple[int, int, int]] = {}  # item -> (start, end, chars)
    for n, (start, item) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
        length = sum(len(l) for l in lines[start:end])
        if item not in candidates or length > candidates[item][2]:
            candidates[item] = (start, end, length)

    sections = []
    for item, (start, end, _) in sorted(candidates.items(), key=lambda kv: kv[1][0]):
        body = "\n".join(lines[start:end]).strip()
        if len(body) < 50:
            continue
        sections.append({"item": item, "title": ITEM_TITLES[item], "text": body})
    return sections


def parse_filing(html_path: Path, out_dir: Path) -> Path:
    meta_path = html_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if not meta:
        ticker, year = html_path.stem.split("_")
        meta = {"ticker": ticker, "fiscal_year": int(year)}
    text = html_to_text(html_path.read_text(errors="ignore"))
    doc = {
        "doc_id": html_path.stem,
        "ticker": meta["ticker"],
        "fiscal_year": meta["fiscal_year"],
        "company": meta.get("company", ""),
        "url": meta.get("url", ""),
        "sections": split_items(text),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{html_path.stem}.json"
    out.write_text(json.dumps(doc))
    return out


def parse_all(raw_dir: str | Path, out_dir: str | Path) -> list[Path]:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    outs = []
    for html in sorted(raw_dir.glob("*.html")):
        out = parse_filing(html, out_dir)
        n = len(json.loads(out.read_text())["sections"])
        print(f"[parsed] {html.stem}: {n} sections")
        outs.append(out)
    return outs
