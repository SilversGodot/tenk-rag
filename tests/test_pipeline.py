"""Offline tests on a small synthetic 10-K (no network, no API key, no model download)."""
import json
from pathlib import Path

from tenk_rag.chunk import chunk_document
from tenk_rag.config import PipelineSettings
from tenk_rag.evaluate import evidence_hit, retrieval_metrics, summarize
from tenk_rag.index import Index
from tenk_rag.parse import html_to_text, parse_filing, split_items
from tenk_rag.retrieve import Retriever, rrf

FILLER = " ".join(["The company continues to monitor general business conditions."] * 30)

SAMPLE_HTML = f"""<html><body>
<div style="display:none"><ix:header>hidden xbrl junk</ix:header></div>
<p>TABLE OF CONTENTS</p>
<p>Item 1. Business</p><p>Item 1A. Risk Factors</p><p>Item 7. Management's Discussion and Analysis</p>
<p>PART I</p>
<p>Item 1. Business</p>
<p>Acme Widgets designs and sells industrial widgets in 40 countries. {FILLER}</p>
<p>Item 1A. Risk Factors</p>
<p>Our supply chain depends on a single supplier of rare earth magnets located in one region. {FILLER}</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Total net sales increased 12% compared with the prior year, driven by higher widget volumes.</p>
<table>
<tr><td>Segment</td><td>2025</td><td>2024</td></tr>
<tr><td>Industrial widgets</td><td>$</td><td>4,210</td><td>$</td><td>3,750</td></tr>
<tr><td>Consumer widgets</td><td>$</td><td>1,105</td><td>$</td><td>995</td></tr>
</table>
<p>{FILLER}</p>
</body></html>"""


def _write_sample(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "ACME_2025.html").write_text(SAMPLE_HTML)
    (raw / "ACME_2025.json").write_text(json.dumps({"ticker": "ACME", "fiscal_year": 2025}))
    return parse_filing(raw / "ACME_2025.html", tmp_path / "parsed")


def test_html_to_text_drops_hidden_and_keeps_tables():
    text = html_to_text(SAMPLE_HTML)
    assert "hidden xbrl junk" not in text
    assert "Industrial widgets | 2025" not in text  # header row is separate
    assert "Industrial widgets | 4,210 | 3,750" in text


def test_split_items_skips_table_of_contents():
    sections = split_items(html_to_text(SAMPLE_HTML))
    items = [s["item"] for s in sections]
    assert items == ["1", "1A", "7"]
    assert "rare earth magnets" in sections[1]["text"]
    assert "4,210" in sections[2]["text"]


def test_chunking_strategies(tmp_path):
    doc = json.loads(_write_sample(tmp_path).read_text())
    for strategy in ("fixed", "section"):
        chunks = chunk_document(doc, strategy, size=120, overlap=20)
        assert chunks and all(len(c["text"].split()) <= 120 for c in chunks)
        assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    section_chunks = chunk_document(doc, "section", size=400, overlap=40)
    table_chunks = [c for c in section_chunks if c["has_table"]]
    assert table_chunks and "[/TABLE]" in table_chunks[0]["text"]  # table kept whole


def test_bm25_retrieval_and_metrics(tmp_path):
    _write_sample(tmp_path)
    s = PipelineSettings(chunk_strategy="section", chunk_size=150, chunk_overlap=20,
                         retrieval="bm25", top_k=3, candidates=10)
    idx = Index.build(tmp_path / "parsed", tmp_path / "index", s.chunk_strategy,
                      s.chunk_size, s.chunk_overlap, embedder=None)
    r = Retriever(idx, s)
    hits = r.search("industrial widgets net sales 2025 segment", ticker="ACME", year=2025)
    m = retrieval_metrics(hits, ["Industrial widgets | 4,210 | 3,750"])
    assert m["recall"] == 1.0 and m["mrr"] > 0
    hits = r.search("single supplier rare earth magnets", ticker="ACME", year=2025)
    assert hits[0]["item"] == "1A"
    # filter for a company that does not exist falls back to everything, not nothing
    assert r.search("magnets", ticker="ZZZZ")


def test_evidence_matching_tolerates_formatting():
    chunk = "Total net sales increased 12%  compared with the\nprior year, driven by higher widget volumes."
    assert evidence_hit(chunk, "Total net sales increased 12% compared with the prior year")
    assert not evidence_hit(chunk, "Net income decreased sharply due to impairments")


def test_rrf_and_summary():
    assert rrf([[1, 2, 3], [3, 1, 2]])[0] == 1
    rows = [{"answerable": True, "type": "table", "recall": 1.0, "mrr": 0.5},
            {"answerable": True, "type": "fact", "recall": 0.0, "mrr": 0.0},
            {"answerable": False, "type": "unanswerable"}]
    s = summarize(rows)
    assert s["recall_at_k"] == 0.5 and s["by_type"]["table"]["recall_at_k"] == 1.0
