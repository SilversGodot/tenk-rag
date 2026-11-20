"""Build and load search indexes (dense vectors + BM25) for one chunking setup."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .chunk import chunk_corpus

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def bm25_tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Embedder:
    """Thin wrapper so the embedding model is easy to swap (one experiment axis)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # bge models expect this prefix on queries (not on passages)
        self.query_prefix = "Represent this sentence for searching relevant passages: " \
            if "bge" in model_name.lower() else ""

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, show_progress_bar=True,
                                 normalize_embeddings=True).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0].astype(np.float32)


class Index:
    def __init__(self, chunks: list[dict], vectors: np.ndarray | None):
        self.chunks = chunks
        self.vectors = vectors
        self.bm25 = BM25Okapi([bm25_tokenize(c["header"] + " " + c["text"]) for c in chunks])
        self.by_id = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    @staticmethod
    def build(parsed_dir: str | Path, out_dir: str | Path, strategy: str, size: int,
              overlap: int, embedder: Embedder | None) -> "Index":
        chunks = chunk_corpus(parsed_dir, strategy, size, overlap)
        if not chunks:
            raise RuntimeError(f"No parsed filings in {parsed_dir}. Run download + parse first.")
        vectors = None
        if embedder is not None:
            vectors = embedder.embed_passages([c["header"] + "\n" + c["text"] for c in chunks])
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        if vectors is not None:
            np.save(out_dir / "vectors.npy", vectors)
            (out_dir / "meta.json").write_text(json.dumps({"embedding_model": embedder.model_name}), encoding="utf-8")
        print(f"[index] {len(chunks)} chunks -> {out_dir}")
        return Index(chunks, vectors)

    @staticmethod
    def load(index_dir: str | Path) -> "Index":
        index_dir = Path(index_dir)
        if not (index_dir / "chunks.jsonl").exists():
            raise FileNotFoundError(f"No index at {index_dir}. Run: python -m tenk_rag.cli index")
        chunks = [json.loads(l) for l in open(index_dir / "chunks.jsonl", encoding="utf-8")]
        vec_path = index_dir / "vectors.npy"
        vectors = np.load(vec_path) if vec_path.exists() else None
        return Index(chunks, vectors)
