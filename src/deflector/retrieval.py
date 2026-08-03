"""Hybrid retrieval: BM25 + dense embeddings, fused with Reciprocal Rank Fusion.

**Why hybrid rather than dense alone.** Support tickets are full of tokens where lexical match is
exactly what you want and semantic similarity actively hurts: `auth_env_mismatch`, `429`,
`mk_test_`, `Meridian-Signature`. An embedding model maps `401` and `403` to almost the same point
in space; BM25 does not. Conversely, "customers keep getting kicked out when they log in" shares no
terms with the SSO document but is semantically adjacent to it. Each arm covers the other's failure
mode, which is why hybrid beats either alone on real support traffic.

**Why RRF rather than a weighted score blend.** BM25 scores are unbounded and corpus-dependent;
cosine similarities sit in [-1, 1]. Blending them requires normalising two distributions that drift
independently as the corpus changes, and the weights need retuning every time. RRF throws away the
magnitudes and fuses on *rank*, so it needs no normalisation and no retuning:

    RRF(d) = sum over arms of  1 / (k + rank(d))

with k = 60, the value from Cormack et al. (2009) that has held up as a default ever since. k damps
the influence of the very top rank so a single arm cannot dominate the fused list.

**Why no vector database.** Eight documents produce roughly 90 chunks. An exact NumPy-free dot
product over 90 vectors takes well under a millisecond. Introducing Pinecone or pgvector here would
add a network hop, a deployment dependency and an index-consistency problem in exchange for solving
a problem that does not exist at this scale. The README states the point at which that stops being
true and what to switch to.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass

from .config import CONFIG, INDEX_PATH
from .corpus import Chunk, load_chunks
from .providers import Provider

_TOKEN = re.compile(r"[a-z0-9_]+")

# Deliberately short. Standard English stoplists remove "not", "no" and "can", all of which change
# the meaning of a support answer. Only genuinely contentless tokens are dropped.
_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "was", "were", "be", "been",
    "in", "on", "at", "for", "with", "as", "by", "that", "this", "it", "its", "from",
    "i", "we", "you", "my", "our", "your",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split, and additionally emit the parts of snake_case identifiers.

    `rate_limit_exceeded` should match a query saying "rate limit exceeded", so the identifier is
    indexed both whole and split. Keeping the whole form matters too — an exact error-code match is
    the strongest retrieval signal a support corpus offers.
    """
    raw = _TOKEN.findall(text.lower())
    out: list[str] = []
    for token in raw:
        if token in _STOPWORDS:
            continue
        out.append(token)
        if "_" in token:
            out.extend(p for p in token.split("_") if p and p not in _STOPWORDS)
    return out


class BM25:
    """Okapi BM25.

        score(q, d) = SUM over terms t in q of
                      IDF(t) * ( f(t,d) * (k1 + 1) ) / ( f(t,d) + k1 * (1 - b + b * |d|/avgdl) )

        IDF(t) = ln( 1 + (N - n(t) + 0.5) / (n(t) + 0.5) )

    k1 controls how fast term frequency saturates: the tenth occurrence of "webhook" in a document
    adds far less than the second. b controls length normalisation — at b=0.75 a long document is
    penalised, but not as harshly as pure per-length division would.
    """

    def __init__(self, documents: list[list[str]], k1: float, b: float) -> None:
        self.k1 = k1
        self.b = b
        self.docs = documents
        self.n_docs = len(documents) or 1
        self.doc_len = [len(d) for d in documents]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.freqs = [Counter(d) for d in documents]

        df: Counter[str] = Counter()
        for freq in self.freqs:
            df.update(freq.keys())
        self.idf = {
            term: math.log(1 + (self.n_docs - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def score(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.n_docs
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in enumerate(self.freqs):
                f = freq.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Hit:
    chunk: Chunk
    rrf_score: float
    dense_score: float
    bm25_score: float
    bm25_rank: int | None
    dense_rank: int | None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_title": self.chunk.doc_title,
            "section": self.chunk.section,
            "rrf": round(self.rrf_score, 5),
            "dense": round(self.dense_score, 4),
            "bm25": round(self.bm25_score, 3),
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
        }


class Retriever:
    def __init__(self, provider: Provider, chunks: list[Chunk] | None = None) -> None:
        self.provider = provider
        self.chunks = chunks if chunks is not None else load_chunks()
        self.cfg = CONFIG.retrieval
        self.bm25 = BM25(
            [tokenize(c.contextualized) for c in self.chunks], self.cfg.bm25_k1, self.cfg.bm25_b
        )
        self.vectors: list[list[float]] | None = None
        self.dense_available = False

    # -- index -------------------------------------------------------------------------

    def build_index(self, force: bool = False) -> bool:
        """Embed the corpus once and persist it.

        The index is committed to the repo. A reviewer cloning this can run the service without
        spending a single embedding call, and retrieval behaviour is identical to the run that
        produced the README numbers.
        """
        fingerprint = _fingerprint(self.chunks)

        if not force and INDEX_PATH.exists():
            try:
                saved = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
                if saved.get("fingerprint") == fingerprint:
                    self.vectors = saved["vectors"]
                    self.dense_available = True
                    return True
            except (json.JSONDecodeError, KeyError):
                pass

        vectors = self.provider.embed([c.contextualized for c in self.chunks])
        if vectors is None:
            self.dense_available = False
            return False

        self.vectors = vectors
        self.dense_available = True
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        INDEX_PATH.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "model": CONFIG.models.embedder,
                    "n_chunks": len(self.chunks),
                    "vectors": vectors,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return True

    # -- search ------------------------------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or self.cfg.top_k
        n_candidates = self.cfg.candidates_per_arm
        k = self.cfg.rrf_k

        bm25_scores = self.bm25.score(tokenize(query))
        bm25_order = sorted(
            range(len(self.chunks)), key=lambda i: bm25_scores[i], reverse=True
        )[:n_candidates]
        bm25_order = [i for i in bm25_order if bm25_scores[i] > 0]

        dense_scores = [0.0] * len(self.chunks)
        dense_order: list[int] = []
        if self.dense_available and self.vectors:
            query_vec = self.provider.embed([query])
            if query_vec:
                qv = query_vec[0]
                dense_scores = [cosine(qv, v) for v in self.vectors]
                dense_order = sorted(
                    range(len(self.chunks)), key=lambda i: dense_scores[i], reverse=True
                )[:n_candidates]

        bm25_rank = {idx: r for r, idx in enumerate(bm25_order)}
        dense_rank = {idx: r for r, idx in enumerate(dense_order)}

        fused: dict[int, float] = {}
        for idx, rank in bm25_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for idx, rank in dense_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            Hit(
                chunk=self.chunks[idx],
                rrf_score=score,
                dense_score=dense_scores[idx],
                bm25_score=bm25_scores[idx],
                bm25_rank=bm25_rank.get(idx),
                dense_rank=dense_rank.get(idx),
            )
            for idx, score in ranked
        ]

    # -- calibrated strength -----------------------------------------------------------

    def strength(self, hits: list[Hit]) -> tuple[float, float]:
        """Return (strength, margin) in a roughly 0-1 space.

        RRF scores are not interpretable as confidence — they depend only on rank, so the top hit
        always scores about the same whether it is a perfect match or the least-bad of a bad set.
        For the confidence gate we need a number that actually drops when the corpus does not cover
        the question, so strength comes from the underlying similarity, not from the fusion.

        Dense cosine is used when available. Without it, BM25 is squashed through a saturating
        transform: s / (s + 6), where 6 is roughly the BM25 score of a solid single-section match on
        this corpus. It is a crude calibration and is labelled as such — the README records the
        degraded-mode thresholds separately.
        """
        if not hits:
            return 0.0, 0.0

        if self.dense_available:
            values = sorted((h.dense_score for h in hits), reverse=True)
        else:
            values = sorted((h.bm25_score / (h.bm25_score + 6.0) for h in hits), reverse=True)

        top = max(0.0, values[0])
        second = max(0.0, values[1]) if len(values) > 1 else 0.0
        return top, max(0.0, top - second)


def _fingerprint(chunks: list[Chunk]) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(CONFIG.models.embedder.encode())
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.contextualized.encode())
    return h.hexdigest()[:16]
