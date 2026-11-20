"""Turn 10-K HTML into clean text split by Item (1, 1A, 7, 7A, 8, ...).

Tables are kept as pipe-delimited rows so numbers stay next to their labels,
which matters a lot for financial questions.
"""
from __future__ import annotations

import json
import textwrap
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


def html_to_text(html: str | bytes) -> str:
    # Passing raw bytes lets BeautifulSoup detect the encoding (UTF-8 or Windows-1252).
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


PART_RE = re.compile(r"^\s*part\s+(i{1,3}|iv)\s*\.?\s*$", re.IGNORECASE)


def split_items(text: str) -> list[dict]:
    """Split text into Item sections.

    Two complications:
    - The table of contents also lists every Item, so each heading appears at
      least twice. For each Item we keep the occurrence with the longest body,
      which is the real section rather than the TOC line.
    - Some filers (e.g. Microsoft) repeat a running header such as "PART II /
      Item 7" at the top of every page. Consecutive marks for the same Item are
      merged into one section, and the repeated header lines are removed.
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

    # Collapse runs of the same Item (running page headers) into one span.
    runs: list[tuple[str, list[int]]] = []
    for idx, item in marks:
        if runs and runs[-1][0] == item:
            runs[-1][1].append(idx)
        else:
            runs.append((item, [idx]))

    candidates: dict[str, tuple[int, int, set[int], int]] = {}  # item -> (start, end, header lines, chars)
    for n, (item, idxs) in enumerate(runs):
        start = idxs[0]
        end = runs[n + 1][1][0] if n + 1 < len(runs) else len(lines)
        length = sum(len(l) for l in lines[start:end])
        if item not in candidates or length > candidates[item][3]:
            candidates[item] = (start, end, set(idxs[1:]), length)

    sections = []
    for item, (start, end, headers, _) in sorted(candidates.items(), key=lambda kv: kv[1][0]):
        body_lines = []
        for i in range(start, end):
            if i in headers:
                # drop a running header and the "PART II" line just before it
                if body_lines and PART_RE.match(body_lines[-1]):
                    body_lines.pop()
                continue
            body_lines.append(lines[i])
        body = "\n".join(body_lines).strip()
        if len(body) < 50:
            continue
        sections.append({"item": item, "title": ITEM_TITLES[item], "text": body})
    return relocate_financial_sections(sections)


# ---- Filers that put MD&A / financial statements outside Items 7 and 8 -------
# Chevron places them after Part III (they land in Item 14), JPMorgan attaches
# the full annual report after Item 15, and NVIDIA puts its statements in
# Item 15. In those filings Items 7/8 are a one-line cross-reference. We find
# the real content inside the oversized section and move it back.

TOC_LINE_RE = re.compile(r"\|\s*\d{1,3}(\s*[–\-]\s*\d{1,3})?\s*$")
MDNA_HEAD_RE = re.compile(r"^\s*management.s discussion and analysis", re.IGNORECASE)
AUDIT_HEAD_RE = re.compile(r"^\s*report of independent registered public accounting firm\s*$", re.IGNORECASE)
EXHIBIT_HEAD_RE = re.compile(r"^\s*exhibit index\s*$", re.IGNORECASE)
STUB_WORDS = 400      # Item 7/8 shorter than this is treated as a cross-reference
HOST_WORDS = 5000     # a non-core Item longer than this is probably hosting them


def _words(text: str) -> int:
    return len(text.split())


def relocate_financial_sections(sections: list[dict]) -> list[dict]:
    by_item = {s["item"]: s for s in sections}
    need7 = _words(by_item.get("7", {}).get("text", "")) < STUB_WORDS
    need8 = _words(by_item.get("8", {}).get("text", "")) < STUB_WORDS
    if not (need7 or need8):
        return sections
    hosts = [s for s in sections if s["item"] not in {"1", "1A", "7", "8"} and _words(s["text"]) > HOST_WORDS]
    if not hosts:
        return sections
    host = max(hosts, key=lambda s: _words(s["text"]))
    lines = host["text"].split("\n")

    # Skip the host's own table of contents (lines ending in "| page number").
    # Only the first cluster counts: financial table rows can also end in a
    # small number, and they must not be mistaken for TOC entries.
    toc = []
    for i, l in enumerate(lines[:400]):
        if TOC_LINE_RE.search(l):
            if toc and i - toc[-1] > 10:
                break
            toc.append(i)
    body_start = toc[-1] + 1 if len(toc) >= 3 else 0

    audit = next((i for i in range(body_start, len(lines)) if AUDIT_HEAD_RE.match(lines[i])), None)
    if audit is None:
        return sections
    exhibits = next((i for i in range(audit, len(lines)) if EXHIBIT_HEAD_RE.match(lines[i])), len(lines))
    mdna = next((i for i in range(body_start, audit) if MDNA_HEAD_RE.match(lines[i])), None)

    moved = {}
    keep_until = audit
    if need7 and mdna is not None and _words("\n".join(lines[mdna:audit])) > STUB_WORDS:
        moved["7"] = "\n".join(lines[mdna:audit]).strip()
        keep_until = mdna
    if need8 and _words("\n".join(lines[audit:exhibits])) > STUB_WORDS:
        moved["8"] = "\n".join(lines[audit:exhibits]).strip()
    elif "7" not in moved:
        return sections
    else:
        exhibits = audit  # Item 8 not moved: leave statements in the host
    host["text"] = "\n".join(lines[:keep_until] + lines[exhibits:]).strip()

    for item, text in moved.items():
        note = f"[Relocated from Item {host['item']}: this filer places this content outside Item {item}]\n"
        if item in by_item:
            by_item[item]["text"] = by_item[item]["text"] + "\n\n" + note + text
        else:
            new = {"item": item, "title": ITEM_TITLES[item], "text": note + text}
            sections.append(new)
            by_item[item] = new
    order = list(ITEM_TITLES)
    return sorted(sections, key=lambda s: order.index(s["item"]) if s["item"] in order else -1)


def parse_filing(html_path: Path, out_dir: Path) -> Path:
    meta_path = html_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if not meta:
        ticker, year = html_path.stem.split("_")
        meta = {"ticker": ticker, "fiscal_year": int(year)}
    text = html_to_text(html_path.read_bytes())
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
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    write_preview(doc, out_dir.parent / "preview" / f"{html_path.stem}.txt")
    return out


def write_preview(doc: dict, path: Path) -> None:
    """Human-readable copy of a parsed filing: a summary table of sections,
    then each section's text with real line breaks. For checking parse quality;
    the pipeline itself reads only the .json files."""
    rule = "=" * 100
    lines = [
        rule,
        f"  {doc['ticker']}  10-K  fiscal year {doc['fiscal_year']}   {doc.get('company', '')}",
        f"  {doc.get('url', '')}",
        rule,
        "",
        f"  {'Item':<6}{'Title':<60}{'Words':>9}{'Tables':>9}",
        "  " + "-" * 84,
    ]
    for s in doc["sections"]:
        lines.append(f"  {s['item']:<6}{s['title'][:58]:<60}{len(s['text'].split()):>9,}"
                     f"{s['text'].count('[TABLE]'):>9}")
    total = sum(len(s["text"].split()) for s in doc["sections"])
    lines += [f"  {'':<6}{'TOTAL':<60}{total:>9,}", ""]
    for s in doc["sections"]:
        lines += ["", rule, f"  ITEM {s['item']}: {s['title'].upper()}", rule, "",
                  pretty_text(s["text"]), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


NUMERIC_RE = re.compile(r"^[($\-—–]*[\d,.]+\)?\s?%?$|^[—–\-]$")
MAX_LABEL = 60   # wrap long row labels so tables stay a readable width


def format_table(rows: list[str]) -> list[str]:
    """Align pipe-delimited rows into columns: labels left, numbers right.
    Used only for preview files; the pipeline keeps the compact pipe format."""
    table = [[c.strip() for c in r.split(" | ")] for r in rows]
    ncols = max(len(r) for r in table)
    # Numeric rows usually have fewer cells than ncols when header cells span
    # columns; right-align short rows so their numbers line up under the years.
    padded = [[r[0]] + [""] * (ncols - len(r)) + r[1:] if len(r) > 1 else r + [""] * (ncols - 1)
              for r in table]
    widths = [min(MAX_LABEL, max(len(r[0]) for r in padded))]
    widths += [max(len(r[i]) for r in padded) for i in range(1, ncols)]

    out = ["  +" + "-" * (sum(widths) + 3 * (ncols - 1) + 2) + "+"]
    for r in padded:
        label_lines = textwrap.wrap(r[0], widths[0]) or [""]
        for n, label in enumerate(label_lines):
            cells = [label.ljust(widths[0])]
            for i in range(1, ncols):
                val = r[i] if n == len(label_lines) - 1 else ""  # numbers on the label's last line
                cells.append(val.rjust(widths[i]) if NUMERIC_RE.match(val or "0") else val.ljust(widths[i]))
            out.append("  | " + " | ".join(cells) + " |")
    out.append(out[0])
    return out


def pretty_text(text: str) -> str:
    """Section text for humans: aligned tables, wrapped paragraphs."""
    out, table = [], None
    for line in text.split("\n"):
        if line == "[TABLE]":
            table = []
        elif line == "[/TABLE]":
            if table:
                out += [""] + format_table(table) + [""]
            table = None
        elif table is not None:
            table.append(line)
        elif not line.strip():
            out.append("")
        else:
            out += textwrap.wrap(line, 100) or [""]
    return "\n".join(out)


def write_all_previews(parsed_dir: str | Path) -> int:
    parsed_dir = Path(parsed_dir)
    n = 0
    for p in sorted(parsed_dir.glob("*.json")):
        write_preview(json.loads(p.read_text(encoding="utf-8")),
                      parsed_dir.parent / "preview" / f"{p.stem}.txt")
        n += 1
    return n


def parse_all(raw_dir: str | Path, out_dir: str | Path) -> list[Path]:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    outs = []
    for html in sorted(raw_dir.glob("*.html")):
        out = parse_filing(html, out_dir)
        n = len(json.loads(out.read_text(encoding="utf-8"))["sections"])
        print(f"[parsed] {html.stem}: {n} sections")
        outs.append(out)
    return outs
