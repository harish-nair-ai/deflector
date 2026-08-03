"""Semantic answer cache.

Support traffic is extremely repetitive. "Why am I getting 429s", "what's my rate limit", "we keep
hitting the API limit" are three tickets and one question, and a naive exact-string cache catches
none of them because no two customers phrase anything the same way. Embedding the question and
matching on cosine similarity catches all three.

At 500 tickets/day this is the single largest cost lever in the system — larger than model choice,
because a cache hit costs one embedding call (~$0.0000004) instead of two chat completions.

**Three things this gets right that naive semantic caches get wrong:**

1. **Only `auto_resolve` answers are cached.** A cached escalation is worthless — the ticket still
   needs a human, and serving a stale "we escalated this" tells the next customer nothing. More
   importantly, caching a *low-confidence* answer and then serving it to a near-miss question would
   launder an uncertain answer into a confident-looking one.

2. **Tickets carrying sensitive data or policy intent are never cached, read or written.** A cache
   keyed on similarity is a cross-customer channel by construction. If ticket A contained a card
   number and produced an answer, serving that answer to customer B on a similar question risks
   leaking specifics of A's situation. Anything the guardrails flagged is excluded from both sides
   of the cache.

3. **The threshold is deliberately high (0.95).** Semantic caching fails in one direction: two
   questions that are *nearly* the same but differ in the one word that matters — "what is the
   Growth rate limit" versus "what is the Starter rate limit" — sit very close in embedding space
   and have completely different answers. 0.95 is tuned to be conservative, because a cache miss
   costs a fraction of a cent and a wrong cache hit is exactly the failure mode this whole system
   is built to prevent.

The store is a JSON file. That is correct at this scale and wrong at ten times it; the README says
what to replace it with and when.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import CACHE_DIR, CONFIG
from .retrieval import cosine


@dataclass
class CacheEntry:
    question: str
    answer: str
    citations: list[dict]
    score: float
    vector: list[float]
    created_at: float
    hits: int = 0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "score": self.score,
            "vector": self.vector,
            "created_at": self.created_at,
            "hits": self.hits,
        }


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    writes: int = 0
    skipped_unsafe: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def to_dict(self) -> dict:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "writes": self.writes,
            "skipped_unsafe": self.skipped_unsafe,
            "hit_rate": round(self.hit_rate, 4),
        }


class SemanticCache:
    def __init__(
        self,
        provider,
        path: Path | None = None,
        threshold: float | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        self.provider = provider
        self.path = path or (CACHE_DIR / "semantic_cache.json")
        self.threshold = threshold if threshold is not None else CONFIG.cache.similarity_threshold
        self.ttl = ttl_seconds if ttl_seconds is not None else CONFIG.cache.ttl_seconds
        self.stats = CacheStats()
        self.entries: list[CacheEntry] = []
        self._load()

    # -- persistence -------------------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        now = time.time()
        self.entries = [
            CacheEntry(**e) for e in raw.get("entries", [])
            if now - e.get("created_at", 0) < self.ttl
        ]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"entries": [e.to_dict() for e in self.entries]}, separators=(",", ":")),
            encoding="utf-8",
        )

    # -- safety ------------------------------------------------------------------------

    @staticmethod
    def is_cacheable(route: str, screening: dict) -> bool:
        """Only confident, clean answers are eligible.

        This is the security boundary of the cache, not a performance tweak — a similarity-keyed
        store is a cross-customer channel, so anything the guardrails touched stays out of it.
        """
        if route != "auto_resolve":
            return False
        if screening.get("policy_intents"):
            return False
        if screening.get("injection_hits"):
            return False
        for detection in screening.get("detections", []):
            if detection.get("tier") == "escalate":
                return False
        return True

    # -- api ---------------------------------------------------------------------------

    def lookup(self, question: str, screening: dict) -> CacheEntry | None:
        if not CONFIG.cache.enabled:
            return None

        # A ticket carrying a secret must not even probe the cache: the answer it gets back would be
        # derived from someone else's ticket, and this one needs a human regardless.
        if not self.is_cacheable("auto_resolve", screening):
            self.stats.skipped_unsafe += 1
            return None

        self.stats.lookups += 1
        if not self.entries:
            return None

        vectors = self.provider.embed([question])
        if not vectors:
            return None
        qv = vectors[0]

        best: CacheEntry | None = None
        best_score = 0.0
        for entry in self.entries:
            score = cosine(qv, entry.vector)
            if score > best_score:
                best, best_score = entry, score

        if best is not None and best_score >= self.threshold:
            best.hits += 1
            self.stats.hits += 1
            self._save()
            return best
        return None

    def store(self, question: str, result, screening: dict) -> bool:
        if not CONFIG.cache.enabled:
            return False
        if not self.is_cacheable(result.route.value, screening):
            return False

        vectors = self.provider.embed([question])
        if not vectors:
            return False

        self.entries.append(
            CacheEntry(
                question=question,
                answer=result.answer,
                citations=result.citations,
                score=result.score,
                vector=vectors[0],
                created_at=time.time(),
            )
        )
        # Keep the store bounded by evicting the least-used oldest entries first.
        if len(self.entries) > CONFIG.cache.max_entries:
            self.entries.sort(key=lambda e: (e.hits, e.created_at))
            self.entries = self.entries[-CONFIG.cache.max_entries:]

        self.stats.writes += 1
        self._save()
        return True
