"""Tests for layout-aware parsing and the confidence gate. All offline, no model calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from deflector.confidence import (
    Band,
    Route,
    audit_citations,
    cheap_signals_already_decided,
    score_confidence,
)
from deflector.ingest import (
    looks_like_continuation,
    normalize_table,
    parse_markdown,
    row_blocks,
    summarize_table,
    table_to_markdown,
)

ROOT = Path(__file__).resolve().parents[1]


# =======================================================================================
# Table handling — the part that most affects real accuracy
# =======================================================================================

class TestTables:
    def test_ragged_rows_are_padded_not_dropped(self) -> None:
        rows = normalize_table([["Plan", "RPM", "Burst"], ["Growth", "1200"], []])
        assert rows == [["Plan", "RPM", "Burst"], ["Growth", "1200", ""]]

    def test_row_blocks_attach_headers_to_every_value(self) -> None:
        rows = [["Plan", "RPM"], ["Growth", "1200"], ["Starter", "300"]]
        blocks = row_blocks(rows, "Limits", "Plan limits", None, 1)
        assert len(blocks) == 2
        assert "Plan: Growth" in blocks[0].display
        assert "RPM: 1200" in blocks[0].display
        # A row without its headers is a meaningless tuple; this is the whole point.
        assert "RPM: 300" in blocks[1].display

    def test_single_row_table_yields_no_row_blocks(self) -> None:
        assert row_blocks([["Header"]], "s", "", None, 1) == []

    def test_summary_names_columns_and_keys_for_the_dense_arm(self) -> None:
        rows = [["Plan", "RPM"], ["Growth", "1200"], ["Starter", "300"]]
        summary = summarize_table(rows, "Plan limits", "Limits")
        assert "Columns: Plan, RPM" in summary
        assert "Growth" in summary and "Starter" in summary

    def test_markdown_round_trip_keeps_every_row(self) -> None:
        rows = [["a", "b"], ["1", "2"], ["3", "4"]]
        rendered = table_to_markdown(rows)
        assert rendered.count("\n") == 3          # header, separator, two body rows
        assert "| 3 | 4 |" in rendered


class TestCrossPageStitching:
    def test_repeated_header_near_top_is_a_continuation(self) -> None:
        prev = [["Code", "HTTP"], ["a", "400"]]
        nxt = [["Code", "HTTP"], ["b", "401"]]
        assert looks_like_continuation(prev, nxt, bbox_top=60)

    def test_table_lower_down_the_page_is_a_new_table(self) -> None:
        prev = [["Code", "HTTP"], ["a", "400"]]
        nxt = [["Code", "HTTP"], ["b", "401"]]
        assert not looks_like_continuation(prev, nxt, bbox_top=500)

    def test_different_column_count_is_never_a_continuation(self) -> None:
        prev = [["Code", "HTTP"], ["a", "400"]]
        nxt = [["Severity", "Definition", "Target"], ["Sev-1", "x", "y"]]
        assert not looks_like_continuation(prev, nxt, bbox_top=60)


class TestMarkdownParsing:
    def test_tables_become_atomic_blocks_plus_rows(self) -> None:
        doc = parse_markdown(ROOT / "corpus" / "api-rate-limits.md")
        kinds = {b.kind for b in doc.blocks}
        assert "table" in kinds and "table_row" in kinds and "prose" in kinds

    def test_table_content_is_not_duplicated_into_prose(self) -> None:
        """A table must appear as a table, not also smeared into the surrounding prose."""
        doc = parse_markdown(ROOT / "corpus" / "api-rate-limits.md")
        prose = "\n".join(b.text for b in doc.blocks if b.kind == "prose")
        assert "| Developer (free) | 60 |" not in prose

    def test_growth_row_is_individually_retrievable(self) -> None:
        doc = parse_markdown(ROOT / "corpus" / "api-rate-limits.md")
        rows = [b for b in doc.blocks if b.kind == "table_row" and "Growth" in b.text]
        assert rows, "the Growth plan row should be its own retrievable block"
        assert any("1,200" in b.text for b in rows)


# =======================================================================================
# Citation auditing
# =======================================================================================

class TestCitationAudit:
    def test_every_claim_cited_scores_full_coverage(self) -> None:
        answer = "The limit is 1,200 RPM [S1]. Exceeding it returns 429 [S2]."
        audit = audit_citations(answer, n_sources=2)
        assert audit.claim_sentences == 2
        assert audit.coverage == 1.0
        assert audit.invalid_ids == []

    def test_uncited_claim_lowers_coverage(self) -> None:
        answer = "The limit is 1,200 RPM [S1]. Exceeding it returns 429."
        audit = audit_citations(answer, n_sources=2)
        assert audit.coverage == pytest.approx(0.5)

    def test_citation_beyond_the_supplied_sources_is_fabricated(self) -> None:
        audit = audit_citations("The limit is 1,200 RPM [S7].", n_sources=3)
        assert audit.invalid_ids == ["S7"]

    def test_pure_prose_is_not_penalised(self) -> None:
        audit = audit_citations("I will need to check this with the team.", n_sources=2)
        assert audit.coverage == 1.0


# =======================================================================================
# The gate itself
# =======================================================================================

def _score(**overrides):
    base = dict(
        retrieval_strength=0.8,
        retrieval_margin=0.1,
        citation=audit_citations("The limit is 1,200 RPM [S1].", 2),
        verifier_supported=4,
        verifier_unsupported=0,
        verifier_ran=True,
        verifier_skipped=False,
        self_report=0.9,
        answerable=True,
        policy_intents=[],
        sensitive_escalate=[],
        injection=False,
        n_sources=3,
    )
    base.update(overrides)
    return score_confidence(**base)


class TestConfidenceGate:
    def test_all_signals_strong_gives_auto_resolve(self) -> None:
        decision = _score()
        assert decision.band is Band.HIGH
        assert decision.route is Route.AUTO_RESOLVE
        assert decision.gates == []

    @pytest.mark.parametrize(
        "override,expected_gate",
        [
            ({"injection": True}, "prompt_injection_suspected"),
            ({"sensitive_escalate": ["credit_card"]}, "sensitive_data:credit_card"),
            ({"policy_intents": ["refund_or_credit"]}, "policy_intent:refund_or_credit"),
            ({"answerable": False}, "model_declined_no_supporting_source"),
            ({"n_sources": 0}, "no_sources_retrieved"),
            ({"verifier_unsupported": 2}, "unsupported_claims:2"),
        ],
    )
    def test_hard_gates_force_escalation_despite_strong_signals(self, override, expected_gate) -> None:
        decision = _score(**override)
        assert decision.route is Route.ESCALATE
        assert decision.band is Band.LOW
        assert any(g.startswith(expected_gate.split(":")[0]) for g in decision.gates)

    def test_fabricated_citation_is_a_hard_gate(self) -> None:
        decision = _score(citation=audit_citations("The limit is 1,200 [S9].", 2))
        assert decision.route is Route.ESCALATE
        assert any("fabricated_citation" in g for g in decision.gates)

    def test_out_of_corpus_question_cannot_auto_resolve(self) -> None:
        decision = _score(retrieval_strength=0.05)
        assert decision.route is Route.ESCALATE
        assert any("retrieval_below_floor" in g for g in decision.gates)

    def test_confidence_is_not_authority(self) -> None:
        """A perfectly grounded refund answer still must not be auto-sent."""
        decision = _score(policy_intents=["refund_or_credit"], self_report=1.0)
        assert decision.route is Route.ESCALATE

    def test_self_report_alone_cannot_reach_auto_resolve(self) -> None:
        """The model claiming certainty must not be enough on its own."""
        decision = _score(
            self_report=1.0,
            verifier_ran=False,
            verifier_skipped=False,
            verifier_supported=None,
            verifier_unsupported=None,
            retrieval_strength=0.45,
            citation=audit_citations("It returns 429.", 2),
        )
        assert decision.route is not Route.AUTO_RESOLVE

    def test_missing_verifier_scores_neutral_not_generous(self) -> None:
        decision = _score(
            verifier_ran=False, verifier_skipped=False,
            verifier_supported=None, verifier_unsupported=None,
        )
        assert decision.signals["verifier"] == 0.5

    def test_ambiguous_retrieval_margin_is_discounted(self) -> None:
        tight = _score(retrieval_margin=0.001)
        wide = _score(retrieval_margin=0.2)
        assert tight.signals["retrieval"] < wide.signals["retrieval"]


class TestVerifierShortCircuit:
    def test_skips_when_a_gate_already_decided(self) -> None:
        assert cheap_signals_already_decided(
            retrieval_strength=0.9,
            citation=audit_citations("x [S1].", 2),
            answerable=True,
            policy_intents=["refund_or_credit"],
            sensitive_escalate=[],
            injection=False,
        )

    def test_runs_when_the_outcome_is_still_open(self) -> None:
        assert not cheap_signals_already_decided(
            retrieval_strength=0.9,
            citation=audit_citations("The limit is 1,200 [S1].", 2),
            answerable=True,
            policy_intents=[],
            sensitive_escalate=[],
            injection=False,
        )
