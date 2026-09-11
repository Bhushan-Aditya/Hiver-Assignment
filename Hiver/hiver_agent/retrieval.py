"""Hybrid retrieval: dense cosine + lexical BM25, fused with reciprocal rank
fusion and a cheap deterministic reranker.

The dense index alone misses exact-token matches (order numbers, "200 rupees",
"Lexington KY"); BM25 alone misses paraphrase. Fusing the two rankings is more
robust than either, and costs microseconds on a 20k corpus because BM25 only
touches the postings of query terms.
"""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .core import Pair

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an and or of to for in on at is was were be been am are i you he she it we they my "
    "me your his her our their this that these those with as by from so if then than not no do "
    "did does have has had will would can could should just about have im ive dont cant".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _norm(text: str) -> str:
    t = re.sub(r"@\d+|https?://\S+|#\w+", " ", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t)).strip()


def dedupe_pairs(pairs: list[Pair]) -> list[Pair]:
    """Drop exact normalized-duplicate customer messages, keeping first seen."""
    seen: set[str] = set()
    out: list[Pair] = []
    for p in pairs:
        key = _norm(p.customer_text)
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


@dataclass
class Hit:
    pair: Pair
    index: int
    dense: float          # cosine similarity in [-1, 1]
    lexical: float         # raw BM25 score
    score: float           # fused + reranked score used for ordering


class BM25:
    """Okapi BM25 over a fixed corpus, with a term -> postings inverted index."""

    __slots__ = ("k1", "b", "idf", "postings", "doc_len", "avgdl", "n")

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = np.fromiter((len(d) for d in docs), dtype=np.float32, count=self.n)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        df: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            tf = Counter(doc)
            for term, freq in tf.items():
                df[term] += 1
                self.postings[term].append((i, freq))
        self.idf = {
            term: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for term, d in df.items()
        }

    def scores(self, query: list[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, freq in self.postings[term]:
                dl = self.doc_len[doc_id]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                out[doc_id] += idf * (freq * (self.k1 + 1)) / denom
        return out


class HybridRetriever:
    def __init__(self, pairs: list[Pair], vectors: np.ndarray, dense_weight: float = 0.65, rrf_k: int = 60):
        self.pairs = pairs
        self.vectors = vectors
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self._cust_tokens = [tokenize(p.customer_text) for p in pairs]
        self.bm25 = BM25([c + tokenize(p.reply_text) for c, p in zip(self._cust_tokens, pairs)])

    def search(self, query: str, query_vec: np.ndarray, k: int = 8,
               drop_index: int | None = None) -> list[Hit]:
        q_tokens = tokenize(query)

        dense_all = self.vectors @ query_vec
        if drop_index is not None and 0 <= drop_index < len(dense_all):
            dense_all = dense_all.copy()
            dense_all[drop_index] = -1.0  # leave-one-out for offline evaluation
        pool = max(k * 4, 40)
        dense_top = np.argpartition(-dense_all, min(pool, len(dense_all) - 1))[:pool]
        dense_rank = {int(i): r for r, i in enumerate(dense_top[np.argsort(-dense_all[dense_top])])}

        lex = self.bm25.scores(q_tokens)
        if drop_index is not None:
            lex.pop(drop_index, None)
        lex_rank = {i: r for r, (i, _) in enumerate(sorted(lex.items(), key=lambda kv: -kv[1]))}

        candidates = set(dense_rank) | set(list(lex_rank)[:pool])
        w = self.dense_weight
        fused: list[tuple[float, int]] = []
        for i in candidates:
            score = 0.0
            if i in dense_rank:
                score += w / (self.rrf_k + dense_rank[i])
            if i in lex_rank:
                score += (1 - w) / (self.rrf_k + lex_rank[i])
            fused.append((score, i))
        fused.sort(reverse=True)

        hits = [
            Hit(self.pairs[i], i, float(dense_all[i]), float(lex.get(i, 0.0)), fused_score)
            for fused_score, i in fused[: max(k * 3, 24)]
        ]
        return self._rerank(query, q_tokens, hits)[:k]

    def _rerank(self, query: str, q_tokens: list[str], hits: list[Hit]) -> list[Hit]:
        """Deterministic nudge: reward rare-term overlap and shared digits/entities."""
        q_set = set(q_tokens)
        q_digits = set(re.findall(r"\d+", query))
        for h in hits:
            d_tokens = set(self._cust_tokens[h.index])
            rare_overlap = sum(self.bm25.idf.get(t, 0.0) for t in (q_set & d_tokens))
            digit_match = len(q_digits & set(re.findall(r"\d+", h.pair.customer_text)))
            norm_dense = max(0.0, h.dense)
            h.score = (
                0.60 * norm_dense
                + 0.30 * math.tanh(rare_overlap / 6.0)
                + 0.10 * min(digit_match, 2) / 2.0
            )
        hits.sort(key=lambda x: -x.score)
        return hits
