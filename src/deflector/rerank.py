"""Cross-encoder reranking — optional, and off unless the extra is installed.

**Why a reranker exists at all.** BM25 and embeddings are both *bi-encoders* in spirit: the query and
the document are scored independently and compared by similarity. That is what makes them fast enough
to run over the whole corpus, and it is also their ceiling — neither ever looks at the query and the
document *together*, so neither can tell that "is a billing_admin allowed to cancel?" is answered by
the row that says `cancel plan: No` rather than the row that merely mentions billing_admin most often.

A cross-encoder concatenates the query and one candidate and runs them through a transformer jointly,
so attention spans both. It cannot be run over 262 chunks per query — that is 262 forward passes — but
it is exactly the right tool for reordering the 12 candidates that hybrid retrieval already narrowed
down to. Retrieve wide and cheap, rerank narrow and expensive: the standard two-stage shape.

**Why a local model rather than an LLM reranker.** An LLM listwise reranker is the fashionable
choice and it works, but it adds a model call to every single ticket. At 500 tickets/day that is a
permanent cost line for a reordering task a 22M-parameter cross-encoder does on CPU in milliseconds
for free. The cost discipline that made me skip a vector database applies here too.

**Why it is an optional extra.** `sentence-transformers` pulls in torch, which is a few hundred
megabytes — real weight for a service whose base install is five small packages. So the base install
stays lightweight and the reranker is opt-in:

    uv pip install '.[rerank]'

If the extra is absent, `Reranker.available` is False and retrieval silently keeps its fused order.
The system is designed to be correct without it and better with it, never broken by its absence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# ms-marco-MiniLM-L-6-v2: 22M parameters, trained on MS MARCO passage ranking. Small enough to run on
# CPU per request, and the standard baseline cross-encoder — a deliberate choice over something larger
# because reranking 12 candidates must not become the slowest step in the pipeline.
DEFAULT_MODEL = os.getenv("DEFLECTOR_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")


@dataclass
class RerankResult:
    index: int          # position in the candidate list handed in
    score: float        # cross-encoder relevance logit; higher is better, unbounded


class Reranker:
    """Lazily-loaded cross-encoder. Safe to construct when the dependency is missing."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._checked = False
        self._error: str = ""

    @property
    def available(self) -> bool:
        if not self._checked:
            self._checked = True
            try:
                from sentence_transformers import CrossEncoder  # noqa: F401
            except ImportError as exc:
                self._error = f"sentence-transformers not installed ({exc})"
                return False
        return not self._error

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            # max_length=512 covers a table row plus its context header comfortably. Longer chunks
            # are truncated from the right, which keeps the header — the part that says which
            # document and section this came from — rather than losing it.
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def rank(self, query: str, candidates: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Score every (query, candidate) pair jointly and return them best-first.

        Returns an identity ordering when the model is unavailable, so callers never branch on
        availability for correctness — only for reporting.
        """
        if not candidates:
            return []
        if not self.available:
            return [RerankResult(index=i, score=0.0) for i in range(len(candidates))][: top_k or len(candidates)]

        model = self._load()
        scores = model.predict(
            [(query, text) for text in candidates],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        ranked = sorted(
            (RerankResult(index=i, score=float(s)) for i, s in enumerate(scores)),
            key=lambda r: r.score,
            reverse=True,
        )
        return ranked[: top_k or len(ranked)]

    @staticmethod
    def to_probability(score: float) -> float:
        """Squash a cross-encoder logit into 0-1 so it can feed the confidence gate.

        ms-marco cross-encoders emit an unbounded relevance logit, typically about -11 for an
        unrelated pair and +8 for a strong match. A logistic squash puts that on the same scale as
        the cosine similarity the gate already expects, so the retrieval signal keeps one meaning
        whichever retriever produced it.
        """
        import math

        return 1.0 / (1.0 + math.exp(-score))
