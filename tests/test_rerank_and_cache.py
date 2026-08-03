"""Tests for the two optional components that ship disabled.

Both the cross-encoder reranker and the semantic answer cache are implemented, measured, and turned
off. These tests exist so that stays a decision rather than drift: they assert the defaults, and they
encode the safety property that made the cache unsafe to enable, so anyone who flips it on has to
confront the reason it was off.

No network, no model downloads.
"""

from __future__ import annotations

import time

import pytest

from deflector.config import CONFIG
from deflector.rerank import Reranker
from deflector.semantic_cache import SemanticCache


class TestDefaults:
    def test_reranker_is_disabled_by_default(self) -> None:
        """Measured on the golden set: MRR 1.000 -> 0.965, hit@1 100% -> 94.7%, p50 31ms -> 143ms."""
        assert CONFIG.retrieval.rerank_enabled is False

    def test_semantic_cache_is_disabled_by_default(self) -> None:
        """No safe threshold exists — see evals/measure_semantic_cache.py."""
        assert CONFIG.cache.enabled is False

    def test_reranker_degrades_to_identity_when_unavailable(self) -> None:
        """Callers must never need to branch on availability for correctness."""
        reranker = Reranker(model_name="definitely/not-a-real-model")
        reranker._checked = True
        reranker._error = "simulated unavailable"
        results = reranker.rank("query", ["a", "b", "c"])
        assert [r.index for r in results] == [0, 1, 2]

    def test_reranker_handles_empty_candidates(self) -> None:
        assert Reranker().rank("query", []) == []

    def test_logit_squash_is_monotonic_and_bounded(self) -> None:
        low, mid, high = (Reranker.to_probability(x) for x in (-11.0, 0.0, 8.0))
        assert 0.0 < low < mid < high < 1.0
        assert mid == pytest.approx(0.5)


class _StubProvider:
    """Deterministic fake embeddings — no network, no model."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.table.get(t, [0.0, 0.0, 1.0]) for t in texts]


class _StubResult:
    def __init__(self, route, answer="answer", score=0.9):
        self.route = route
        self.answer = answer
        self.citations = []
        self.score = score


class TestSemanticCacheSafety:
    """The cache is only ever allowed to see clean, already-auto-resolved traffic."""

    CLEAN = {"detections": [], "injection_hits": [], "policy_intents": []}

    def _cache(self, tmp_path, table=None):
        return SemanticCache(
            provider=_StubProvider(table or {}),
            path=tmp_path / "cache.json",
            threshold=0.95,
        )

    def test_only_auto_resolve_is_cacheable(self) -> None:
        assert SemanticCache.is_cacheable("auto_resolve", self.CLEAN) is True
        assert SemanticCache.is_cacheable("agent_assist", self.CLEAN) is False
        assert SemanticCache.is_cacheable("escalate", self.CLEAN) is False

    def test_escalate_tier_detection_blocks_caching(self) -> None:
        screening = {
            "detections": [{"kind": "credit_card", "tier": "escalate"}],
            "injection_hits": [],
            "policy_intents": [],
        }
        assert SemanticCache.is_cacheable("auto_resolve", screening) is False

    def test_redact_tier_detection_does_not_block_caching(self) -> None:
        """An email address is in almost every ticket; blocking on it would disable the cache."""
        screening = {
            "detections": [{"kind": "email", "tier": "redact"}],
            "injection_hits": [],
            "policy_intents": [],
        }
        assert SemanticCache.is_cacheable("auto_resolve", screening) is True

    def test_policy_intent_blocks_caching(self) -> None:
        screening = {"detections": [], "injection_hits": [], "policy_intents": ["refund_or_credit"]}
        assert SemanticCache.is_cacheable("auto_resolve", screening) is False

    def test_injection_blocks_caching(self) -> None:
        screening = {"detections": [], "injection_hits": ["ignore instructions"], "policy_intents": []}
        assert SemanticCache.is_cacheable("auto_resolve", screening) is False

    def test_ttl_expiry_drops_stale_entries(self, tmp_path) -> None:
        from deflector.semantic_cache import CacheEntry
        import json

        stale = CacheEntry(
            question="old", answer="a", citations=[], score=0.9,
            vector=[1.0, 0.0, 0.0], created_at=time.time() - 10_000,
        )
        path = tmp_path / "cache.json"
        path.write_text(json.dumps({"entries": [stale.to_dict()]}))
        cache = SemanticCache(_StubProvider({}), path=path, ttl_seconds=100)
        assert cache.entries == []


class TestParametricCollision:
    """The finding that keeps the cache disabled.

    Two questions differing only in the plan name are *more* similar to each other than a genuine
    rephrasing is to the original. Measured on real embeddings: wrong-entity 0.816 vs weakest
    paraphrase 0.653. This test encodes the consequence with stub vectors so it runs offline: at any
    threshold low enough to catch the paraphrase, the wrong-entity question hits too.
    """

    def test_no_threshold_separates_paraphrase_from_wrong_entity(self, tmp_path) -> None:
        seed = "what is the Growth rate limit"
        paraphrase = "rpm cap on Growth before 429s"
        wrong_entity = "what is the Starter rate limit"

        # Vectors chosen so cos(seed, wrong) > cos(seed, paraphrase), mirroring the real measurement.
        table = {
            seed: [1.0, 0.0, 0.0],
            wrong_entity: [0.816, 0.578, 0.0],      # cos to seed = 0.816
            paraphrase: [0.653, 0.757, 0.0],        # cos to seed = 0.653
        }
        from deflector.confidence import Route

        for threshold in (0.60, 0.70, 0.80):
            cache = SemanticCache(
                _StubProvider(table), path=tmp_path / f"c{threshold}.json", threshold=threshold
            )
            cache.store(seed, _StubResult(Route.AUTO_RESOLVE), TestSemanticCacheSafety.CLEAN)

            para_hit = cache.lookup(paraphrase, TestSemanticCacheSafety.CLEAN) is not None
            wrong_hit = cache.lookup(wrong_entity, TestSemanticCacheSafety.CLEAN) is not None

            # The property: you never get the paraphrase without also getting the wrong entity.
            assert not (para_hit and not wrong_hit), (
                f"at threshold {threshold} the paraphrase hit while the wrong-entity question did "
                "not — if this ever passes, re-run evals/measure_semantic_cache.py, the embedding "
                "model may have changed and the cache may now be safe to enable"
            )
