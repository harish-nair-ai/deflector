"""Tests for the two findings that came out of the first full golden-set run.

Both are about the same distinction: a confident, well-grounded answer is not automatically an
answer we are allowed to send. One is a routing cap; the other is a signal we deliberately do not
repair. Neither needs the network.
"""

from __future__ import annotations

import contextlib

import pytest

from deflector.config import CONFIG
from deflector.confidence import (
    Route,
    audit_citations,
    repair_single_source_markers,
    score_confidence,
)
from deflector.guardrails import match_assist_only_intents


@contextlib.contextmanager
def citation_repair_enabled():
    """The config is a frozen dataclass, so flipping a flag needs the back door and a restore."""
    field = "repair_single_source_citations"
    original = getattr(CONFIG.confidence, field)
    object.__setattr__(CONFIG.confidence, field, True)
    try:
        yield
    finally:
        object.__setattr__(CONFIG.confidence, field, original)


@pytest.fixture
def repair_on():
    with citation_repair_enabled():
        yield


class TestAssistOnlyIntents:
    """Requests to *do* something cap at agent_assist rather than escalating."""

    def test_action_request_is_detected(self) -> None:
        body = (
            "Could you send a copy of last month's invoice to priya.sharma@northwind.example "
            "and cc me? What roles can access invoices?"
        )
        assert match_assist_only_intents(body) == ["action_requested"]

    def test_informational_question_is_not_an_action_request(self) -> None:
        """The bare verb must not fire — these are questions about how the product works."""
        for body in (
            "How do I send a webhook payload to my endpoint?",
            "What happens when we send more than 1200 requests a minute?",
            "The docs say the server will send a retry after 30s — is that right?",
            "What is the retry schedule for webhooks?",
        ):
            assert match_assist_only_intents(body) == [], body

    def _decide(self, assist_only: list[str]) -> Route:
        return score_confidence(
            retrieval_strength=0.8,
            retrieval_margin=0.2,
            citation=audit_citations("Only Owners can access invoices [S1].", n_sources=1),
            verifier_supported=3,
            verifier_unsupported=0,
            verifier_ran=True,
            verifier_skipped=False,
            self_report=0.9,
            answerable=True,
            policy_intents=[],
            sensitive_escalate=[],
            injection=False,
            n_sources=1,
            assist_only_intents=assist_only,
        ).route

    def test_high_confidence_auto_resolves_without_an_action_request(self) -> None:
        assert self._decide([]) is Route.AUTO_RESOLVE

    def test_action_request_caps_high_confidence_at_agent_assist(self) -> None:
        """The answer is good enough to send; the ticket is not one we may close."""
        assert self._decide(["action_requested"]) is Route.AGENT_ASSIST

    def test_cap_is_not_an_escalation(self) -> None:
        """Escalating would waste a human on a ticket whose question we answered correctly."""
        assert self._decide(["action_requested"]) is not Route.ESCALATE

    def test_band_stays_honest_when_the_route_is_capped(self) -> None:
        """We cap what we do, not what we believe. The score is still High."""
        decision = score_confidence(
            retrieval_strength=0.8,
            retrieval_margin=0.2,
            citation=audit_citations("Only Owners can access invoices [S1].", n_sources=1),
            verifier_supported=3,
            verifier_unsupported=0,
            verifier_ran=True,
            verifier_skipped=False,
            self_report=0.9,
            answerable=True,
            policy_intents=[],
            sensitive_escalate=[],
            injection=False,
            n_sources=1,
            assist_only_intents=["action_requested"],
        )
        assert decision.band.value == "High"
        assert decision.score >= CONFIG.confidence.high


class TestCitationRepairStaysOff:
    """Missing citation markers are a quality signal, not a formatting slip.

    Repairing them raised the auto-resolve rate from 10.3% to 17.9% and dropped precision from
    100% to 71.4%, letting through two answers that must not have been auto-sent. The signal is
    correlated with the model synthesising loosely instead of lifting a fact off a source, so
    normalising the format throws away the evidence and keeps the score.
    """

    ANSWER = "The Growth plan sustained limit is 1,200 requests per minute."

    def test_repair_is_disabled_by_default(self) -> None:
        assert CONFIG.confidence.repair_single_source_citations is False

    def test_uncited_answer_keeps_its_zero_coverage(self) -> None:
        repaired, changed = repair_single_source_markers(self.ANSWER, ["S1"])
        assert changed is False
        assert repaired == self.ANSWER
        assert audit_citations(repaired, n_sources=1).coverage == 0.0

    def test_marker_goes_inside_the_sentence_when_enabled(self, repair_on) -> None:
        """Appending after the full stop re-splits into the next sentence and mis-scores coverage."""
        two = "The limit is 1,200 requests per minute. Bursts of 3,000 are allowed for 10 seconds."
        repaired, changed = repair_single_source_markers(two, ["S1"])
        assert changed is True
        assert "minute [S1]." in repaired
        assert audit_citations(repaired, n_sources=1).coverage == 1.0

    def test_multi_source_answers_are_never_repaired(self, repair_on) -> None:
        """Which claim came from which document is a judgement — guessing it invents attribution."""
        repaired, changed = repair_single_source_markers(self.ANSWER, ["S1", "S2"])
        assert changed is False
        assert repaired == self.ANSWER
