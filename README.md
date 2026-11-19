# 10-K RAG: question answering over SEC filings, measured

A retrieval-augmented generation (RAG) system that answers questions about
companies' annual reports (SEC Form 10-K) with citations, plus an evaluation
harness that measures which design choices actually improve it.

**The point of this project is the evaluation, not the demo.** Every design
choice (chunking, search method, reranking, filters) is an experiment with
numbers attached.

## Pipeline

```
EDGAR ─► download ─► parse (HTML → text, split by Item, tables kept as rows)
      ─► chunk (fixed | section-aware)  ─► index (dense vectors + BM25)
      ─► retrieve (vector | BM25 | hybrid RRF, ticker/year filters, optional reranker)
      ─► answer (LLM, cites sources, says NOT_FOUND when unsupported)
      ─► evaluate (recall@k, MRR, correctness, faithfulness, refusal rates)
```

| File | What it does |
|---|---|
| `src/tenk_rag/download.py` | Finds each ticker's CIK and downloads its latest 10-Ks from EDGAR |
| `src/tenk_rag/parse.py` | Cleans the HTML, drops hidden XBRL, keeps tables, splits into Items (skipping the table of contents) |
| `src/tenk_rag/chunk.py` | Two chunking strategies; each chunk carries ticker, year, Item and a context header |
| `src/tenk_rag/index.py` | Sentence-transformer embeddings + BM25 keyword index |
| `src/tenk_rag/retrieve.py` | Vector, BM25 and hybrid search (reciprocal rank fusion), metadata filters, cross-encoder reranking |
| `src/tenk_rag/generate.py` | Answer prompt with citations and a refusal path |
| `src/tenk_rag/evaluate.py` | Metrics, LLM judge, per-question-type breakdown, experiment comparison table |
| `src/tenk_rag/questions.py` | Drafts candidate eval questions for you to review |
| `config.yaml` | Companies, models and the list of experiments |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
export ANTHROPIC_API_KEY=...        # only needed for answering, judging, drafting questions
```

Edit `config.yaml`: set `corpus.user_agent` to your name and email (the SEC
requires it) and pick your companies.

## Run it

```bash
python -m tenk_rag.cli download          # ~30 filings for the default 10 companies
python -m tenk_rag.cli parse
python -m tenk_rag.cli index             # builds the default index (downloads the embedding model once)
python -m tenk_rag.cli ask "What were Apple's total net sales in fiscal 2025?" --ticker AAPL --year 2025 --show-chunks
```

Look at the parsed output in `data/parsed/` early. 10-K HTML varies a lot
between companies, and parsing problems quietly cap every later metric.

## Build the eval set (the most important step)

1. Draft candidates: `python -m tenk_rag.cli draft-questions --chunks 40`
   writes to `eval/candidates.jsonl`. Questions whose evidence isn't an exact
   quote from the source are dropped automatically.
2. Review every candidate by hand against the filing. Fix or delete bad ones.
3. Write the hard ones yourself (see `eval/questions.example.jsonl` for the format):
   - **table**: numbers that live only in financial tables
   - **multi_section**: answers that need Item 7 and Item 8 together
   - **cross-year**: "how did X change from 2023 to 2025?" (needs two filings)
   - **unanswerable**: plausible questions the 10-K doesn't answer (forecasts, stock targets, other companies)
4. Save the final set as `eval/questions.jsonl`. Aim for 150+ questions, at
   least 20% unanswerable or cross-section.

Evidence is stored as text, not chunk IDs, so one eval set works across every
chunking experiment.

## Run experiments

```bash
python -m tenk_rag.cli eval --experiment hybrid --retrieval-only   # free and fast: recall@k, MRR
python -m tenk_rag.cli eval --experiment hybrid                    # adds answers + LLM judge
python -m tenk_rag.cli compare                                     # markdown table of all results
```

Experiments live in `config.yaml`. Suggested order:

1. Retrieval only, cheap: `baseline_vector`, `bm25_only`, `hybrid`, `hybrid_rerank`, `hybrid_no_filters`
2. Chunking: `fixed_256`, `fixed_512`, `section_800` against the default `section` 400
3. Full runs (with the LLM) on the 2–3 best retrieval setups
4. Your own ideas: another embedding model, query rewriting, table-specific chunks, contextual chunk headers on/off

Then check the LLM judge: hand-grade ~30 answers and report how often you
agree with it.

## What to write up

- The results table from `compare`, with one paragraph on what moved the numbers
- A failure analysis by question type (the `by_type` breakdown shows where it breaks)
- Cost and latency per query next to quality
- What you'd do next

## Tests

```bash
pytest -q     # offline, uses a synthetic filing
```
