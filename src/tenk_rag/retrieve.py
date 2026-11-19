"""Retrieval: vector, BM25, or hybrid (reciprocal rank fusion), with optional
metadata filters and cross-encoder reranking."""
from __future__ import annotations

import numpy as np

from .config import PipelineSettings
from .index import Embedder, Index, bm25_tokenize


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: score = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self, index: Index, settings: PipelineSettings,
                 embedder: Embedder | None = None, reranker_name: str | None = None):
        self.index = index
        self.s = settings
        self.embedder = embedder
        self.reranker = None
        if settings.rerank:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_name)
        if settings.retrieval in {"vector", "hybrid"} and (embedder is None or index.vectors is None):
            raise ValueError("Vector/hybrid retrieval needs an embedder and an index built with vectors.")

    def _mask(self, ticker: str | None, year: int | None) -> np.ndarray:
        n = len(self.index.chunks)
        mask = np.ones(n, dtype=bool)
        if not self.s.use_filters:
            return mask
        for i, c in enumerate(self.index.chunks):
            if ticker and c["ticker"] != ticker.upper():
                mask[i] = False
            elif year and c["fiscal_year"] != int(year):
                mask[i] = False
        return mask if mask.any() else np.ones(n, dtype=bool)

    def _vector_rank(self, query: str, mask: np.ndarray, n: int) -> list[int]:
        q = self.embedder.embed_query(query)
        scores = self.index.vectors @ q
        scores[~mask] = -np.inf
        return [int(i) for i in np.argsort(-scores)[:n] if mask[i]]

    def _bm25_rank(self, query: str, mask: np.ndarray, n: int) -> list[int]:
        scores = np.asarray(self.index.bm25.get_scores(bm25_tokenize(query)), dtype=float)
        scores[~mask] = -np.inf
        return [int(i) for i in np.argsort(-scores)[:n] if mask[i]]

    def search(self, query: str, ticker: str | None = None, year: int | None = None) -> list[dict]:
        mask = self._mask(ticker, year)
        pool = self.s.candidates
        if self.s.retrieval not in {"vector", "bm25", "hybrid"}:
            raise ValueError(f"Unknown retrieval mode: {self.s.retrieval}")
        vec = self._vector_rank(query, mask, pool) if self.s.retrieval in {"vector", "hybrid"} else []
        kw = self._bm25_rank(query, mask, pool) if self.s.retrieval in {"bm25", "hybrid"} else []
        if self.s.retrieval == "vector":
            ranked = vec
        elif self.s.retrieval == "bm25":
            ranked = kw
        else:
            ranked = rrf([vec, kw])[:pool]

        # Record where each chunk ranked in each method, for inspection/debugging.
        vec_pos = {idx: r for r, idx in enumerate(vec, 1)}
        kw_pos = {idx: r for r, idx in enumerate(kw, 1)}
        hits = []
        for i in ranked:
            h = dict(self.index.chunks[i])
            h["vector_rank"] = vec_pos.get(i)
            h["bm25_rank"] = kw_pos.get(i)
            hits.append(h)
        if self.reranker is not None and hits:
            scores = self.reranker.predict([(query, h["header"] + "\n" + h["text"]) for h in hits])
            order = np.argsort(-np.asarray(scores))
            hits = [hits[i] | {"rerank_score": float(scores[i])} for i in order]
        return hits[: self.s.top_k]
