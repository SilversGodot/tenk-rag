"""Command line entry point.

  python -m tenk_rag.cli download
  python -m tenk_rag.cli parse
  python -m tenk_rag.cli index [--experiment NAME]
  python -m tenk_rag.cli chunks --ticker AAPL --year 2025 [--item 7] [--experiment NAME]
  python -m tenk_rag.cli ask "What were Apple's main risk factors in FY2025?" --ticker AAPL --year 2025
  python -m tenk_rag.cli draft-questions --chunks 40
  python -m tenk_rag.cli eval --experiment hybrid [--retrieval-only]
  python -m tenk_rag.cli compare
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

from .config import ROOT, PipelineSettings, load_config

INDEX_ROOT = ROOT / "data" / "index"
RESULTS_DIR = ROOT / "results"


def _paths(cfg):
    c = cfg["corpus"]
    return ROOT / c["raw_dir"], ROOT / c["parsed_dir"]


def _embedder(cfg, settings: PipelineSettings):
    if settings.retrieval == "bm25":
        return None
    from .index import Embedder
    return Embedder(cfg["models"]["embedding"])


def _load_or_build_index(cfg, settings: PipelineSettings, embedder):
    from .index import Index
    index_dir = INDEX_ROOT / settings.index_name
    if (index_dir / "chunks.jsonl").exists():
        idx = Index.load(index_dir)
        if embedder is not None and idx.vectors is None:
            print("[index] existing index has no vectors; rebuilding with embeddings")
        else:
            return idx
    _, parsed = _paths(cfg)
    return Index.build(parsed, index_dir, settings.chunk_strategy, settings.chunk_size,
                       settings.chunk_overlap, embedder)


def _retriever(cfg, settings):
    from .retrieve import Retriever
    embedder = _embedder(cfg, settings)
    index = _load_or_build_index(cfg, settings, embedder)
    return Retriever(index, settings, embedder, cfg["models"]["reranker"])


def main(argv=None):
    # Windows consoles and redirected output (> file.txt) default to a legacy encoding;
    # force UTF-8 so filing text with curly quotes, dashes, etc. never crashes printing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="tenk_rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    sub.add_parser("parse")
    sub.add_parser("preview", help="rewrite readable .txt files in data/preview from parsed JSON")
    pi = sub.add_parser("index"); pi.add_argument("--experiment")
    pc = sub.add_parser("chunks", help="write one filing's chunks to a readable .txt file")
    pc.add_argument("--ticker", required=True); pc.add_argument("--year", type=int, required=True)
    pc.add_argument("--item", help="only this Item, e.g. 7 or 1A"); pc.add_argument("--experiment")
    pa = sub.add_parser("ask")
    pa.add_argument("question"); pa.add_argument("--ticker"); pa.add_argument("--year", type=int)
    pa.add_argument("--experiment"); pa.add_argument("--show-chunks", action="store_true")
    pa.add_argument("--no-llm", action="store_true", help="only retrieve and print chunks; no API call")
    pd = sub.add_parser("draft-questions")
    pd.add_argument("--chunks", type=int, default=40); pd.add_argument("--per-chunk", type=int, default=2)
    pd.add_argument("--out", default="eval/candidates.jsonl")
    pe = sub.add_parser("eval")
    pe.add_argument("--experiment", required=True)
    pe.add_argument("--questions", default="eval/questions.jsonl")
    pe.add_argument("--retrieval-only", action="store_true", help="skip LLM answering/judging (free, fast)")
    pe.add_argument("--limit", type=int)
    sub.add_parser("compare")
    args = p.parse_args(argv)

    cfg = load_config()
    raw, parsed = _paths(cfg)

    if args.cmd == "download":
        from .download import download_filings
        c = cfg["corpus"]
        download_filings(c["tickers"], c["filings_per_company"], raw, c["user_agent"])

    elif args.cmd == "parse":
        from .parse import parse_all
        parse_all(raw, parsed)
        print(f"[preview] readable copies written to {parsed.parent / 'preview'}")

    elif args.cmd == "preview":
        from .parse import write_all_previews
        n = write_all_previews(parsed)
        print(f"[preview] wrote {n} files to {parsed.parent / 'preview'}")

    elif args.cmd == "chunks":
        from .chunk import chunk_document
        from .display import format_chunk_list
        s = PipelineSettings.from_config(cfg, args.experiment)
        doc_id = f"{args.ticker.upper()}_{args.year}"
        path = parsed / f"{doc_id}.json"
        if not path.exists():
            available = sorted(p.stem for p in parsed.glob("*.json"))
            sys.exit(f"No parsed filing {doc_id}. Available: {', '.join(available)}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        chunks = chunk_document(doc, s.chunk_strategy, s.chunk_size, s.chunk_overlap)
        if args.item:
            chunks = [c for c in chunks if c["item"] == args.item.upper()]
            if not chunks:
                sys.exit(f"No chunks for Item {args.item.upper()} in {doc_id}.")
        title = (f"{doc_id}{' Item ' + args.item.upper() if args.item else ''}"
                 f"  |  chunking: {s.chunk_strategy}, {s.chunk_size} words, overlap {s.chunk_overlap}")
        out_dir = parsed.parent / "preview" / "chunks"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{doc_id}{'_item' + args.item.upper() if args.item else ''}_{s.index_name}.txt"
        out.write_text(format_chunk_list(chunks, title), encoding="utf-8")
        print(f"[chunks] {len(chunks)} chunks written to {out}")

    elif args.cmd == "index":
        from .index import Index
        s = PipelineSettings.from_config(cfg, args.experiment)
        Index.build(parsed, INDEX_ROOT / s.index_name, s.chunk_strategy, s.chunk_size,
                    s.chunk_overlap, _embedder(cfg, s))

    elif args.cmd == "ask":
        s = PipelineSettings.from_config(cfg, args.experiment)
        hits = _retriever(cfg, s).search(args.question, args.ticker, args.year)
        from .display import format_results
        if args.no_llm or args.show_chunks:
            print(format_results(args.question, hits, s.to_dict()))
        if args.no_llm:
            return
        from .generate import answer
        from .llm import LLM
        result = answer(LLM(cfg["models"]["llm"]), args.question, hits)
        print("\n" + result["answer"])
        print("\nSources:")
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] {h['header']}  {h['url']}")

    elif args.cmd == "draft-questions":
        from .index import Index
        from .llm import LLM
        from .questions import draft_questions
        s = PipelineSettings.from_config(cfg)
        idx = _load_or_build_index(cfg, s, None)
        n = draft_questions(idx.chunks, LLM(cfg["models"]["llm"], max_tokens=1200),
                            args.chunks, args.per_chunk, ROOT / args.out)
        print(f"[draft] wrote {n} candidate questions to {args.out}. Review them before use.")

    elif args.cmd == "eval":
        from .evaluate import load_questions, run_eval
        s = PipelineSettings.from_config(cfg, args.experiment)
        questions = load_questions(ROOT / args.questions)[: args.limit]
        llm = None
        if not args.retrieval_only:
            from .llm import LLM
            llm = LLM(cfg["models"]["llm"])
        print(f"[eval] {args.experiment}: {len(questions)} questions, settings={s.to_dict()}")
        summary = run_eval(questions, _retriever(cfg, s), llm, RESULTS_DIR / f"{args.experiment}.json",
                           s.to_dict(), generate=not args.retrieval_only)
        print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=2))

    elif args.cmd == "compare":
        from .evaluate import compare
        print(compare(RESULTS_DIR))


if __name__ == "__main__":
    main()
