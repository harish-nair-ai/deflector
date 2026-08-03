"""The confidence gate and routing decision.

This is the part of the system that matters. Producing a grounded answer is the easy half; deciding
whether that answer is good enough to send to a customer without a human reading it first is the half
that determines whether the deployment survives contact with production.

**Why not just ask the model for a confidence score.** Because LLMs are badly calibrated about their
own groundedness and skew high — a model that has hallucinated a rate limit will report 0.9 on it,
since from the inside a fabricated fact and a retrieved one feel identical. Self-reported confidence
is kept, because it carries *some* signal, but it is weighted lowest at 0.10 and can never on its own
lift an answer into auto-resolve.

**The four signals**, in descending order of trust:

  verifier   0.40  A different, larger model audits the draft against the sources and counts
                   unsupported claims. Independent, and mechanically consequential: any unsupported
                   claim is also a hard gate.
  retrieval  0.25  Similarity of the best-matching chunk, plus the margin to the next. Low
                   similarity means the corpus does not cover the question, whatever the model wrote.
  citation   0.25  Mechanically computed, no model involved: what fraction of claim-bearing
                   sentences carry a citation, and does every citation point at a real source.
  self       0.10  The model's own estimate.

**Hard gates.** Any one of these forces LOW and escalation regardless of the blended score. A
weighted average is a smooth function, and some failures are not smooth — a citation pointing at [S7]
when only four sources were supplied is not a small quality deduction, it is a fabrication, and no
amount of strength elsewhere should compensate for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .config import CONFIG


class Band(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Route(str, Enum):
    AUTO_RESOLVE = "auto_resolve"   # sent to the customer with no human in the loop
    AGENT_ASSIST = "agent_assist"   # drafted, an agent reviews and sends
    ESCALATE = "escalate"           # human owns it from here; the draft is context, not a reply


@dataclass
class CitationAudit:
    claim_sentences: int
    cited_sentences: int
    invalid_ids: list[str] = field(default_factory=list)
    coverage: float = 0.0


@dataclass
class Decision:
    band: Band
    route: Route
    score: float
    signals: dict[str, float]
    gates: list[str]
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "band": self.band.value,
            "route": self.route.value,
            "score": round(self.score, 4),
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "hard_gates": self.gates,
            "reasons": self.reasons,
        }


# A sentence carrying any of these is making a checkable factual claim and needs a citation.
# Prose like "Let me know if that helps" is not a claim and is not penalised for lacking one.
_CLAIM_MARKERS = re.compile(
    r"""(
        \d                                  |   # any number: limits, prices, durations, codes
        \b[a-z]+_[a-z_]+\b                  |   # snake_case identifiers, i.e. error codes
        \b(?:must|cannot|can't|not|never|always|only|requires?|returns?|
             refund\w*|charge\w*|expire\w*|retr(?:y|ies|ied))\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_CITE = re.compile(r"\[S(\d+)\]")


def audit_citations(answer: str, n_sources: int) -> CitationAudit:
    """Mechanically check citation coverage and validity. No model involved.

    Splits the answer into sentences, decides which of them make factual claims, and measures how
    many of those carry at least one citation pointing at a source that was actually supplied.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    claim_sentences = 0
    cited_sentences = 0
    invalid: list[str] = []

    for sentence in sentences:
        ids = _CITE.findall(sentence)
        for raw in ids:
            index = int(raw)
            if index < 1 or index > n_sources:
                invalid.append(f"S{raw}")
        if _CLAIM_MARKERS.search(_CITE.sub("", sentence)):
            claim_sentences += 1
            if ids:
                cited_sentences += 1

    # An answer with no factual claims at all (a pure "I need to check this") is not penalised on
    # coverage — the abstention path is judged by the retrieval and verifier signals instead.
    coverage = 1.0 if claim_sentences == 0 else cited_sentences / claim_sentences

    return CitationAudit(
        claim_sentences=claim_sentences,
        cited_sentences=cited_sentences,
        invalid_ids=sorted(set(invalid)),
        coverage=coverage,
    )


def score_confidence(
    *,
    retrieval_strength: float,
    retrieval_margin: float,
    citation: CitationAudit,
    verifier_supported: int | None,
    verifier_unsupported: int | None,
    verifier_ran: bool,
    verifier_skipped: bool,
    self_report: float,
    answerable: bool,
    policy_intents: list[str],
    sensitive_escalate: list[str],
    injection: bool,
    n_sources: int,
) -> Decision:
    cfg = CONFIG.confidence
    gates: list[str] = []
    reasons: list[str] = []

    # ---- signal: retrieval ------------------------------------------------------------
    s_retrieval = max(0.0, min(1.0, retrieval_strength))
    if retrieval_margin < cfg.margin_floor:
        # Several chunks matched about equally well: the corpus is ambiguous here, and the model
        # picked one of them for reasons we cannot inspect. Discount rather than gate.
        s_retrieval *= 0.85
        reasons.append(
            f"retrieval margin {retrieval_margin:.3f} below {cfg.margin_floor} — ambiguous coverage"
        )

    # ---- signal: verifier -------------------------------------------------------------
    if verifier_ran and verifier_supported is not None and verifier_unsupported is not None:
        total = verifier_supported + verifier_unsupported
        s_verifier = 1.0 if total == 0 else verifier_supported / total
    else:
        # The verifier did not run. Do not award the benefit of the doubt: an unverified answer is
        # scored neutral, which alone cannot reach the auto-resolve bar. This is the deliberate
        # asymmetry — when the checking machinery is unavailable, the system becomes more cautious,
        # not more permissive.
        s_verifier = 0.5
        if verifier_skipped:
            reasons.append("verifier skipped — a hard gate had already decided the outcome")
        else:
            reasons.append("verifier unavailable or unparseable — scored neutral, not assumed correct")

    # ---- signal: citations ------------------------------------------------------------
    s_citation = citation.coverage

    # ---- signal: self report ----------------------------------------------------------
    s_self = max(0.0, min(1.0, self_report))

    signals = {
        "verifier": s_verifier,
        "retrieval": s_retrieval,
        "citation": s_citation,
        "self_report": s_self,
    }

    score = (
        cfg.w_verifier * s_verifier
        + cfg.w_retrieval * s_retrieval
        + cfg.w_citation * s_citation
        + cfg.w_self_report * s_self
    ) / cfg.weights_sum()

    # ---- hard gates -------------------------------------------------------------------
    if injection:
        gates.append("prompt_injection_suspected")
    if sensitive_escalate:
        gates.append("sensitive_data:" + ",".join(sensitive_escalate))
    if policy_intents:
        gates.append("policy_intent:" + ",".join(policy_intents))
    if not answerable:
        gates.append("model_declined_no_supporting_source")
    if n_sources == 0:
        gates.append("no_sources_retrieved")
    if citation.invalid_ids:
        gates.append("fabricated_citation:" + ",".join(citation.invalid_ids))
    if verifier_ran and verifier_unsupported:
        gates.append(f"unsupported_claims:{verifier_unsupported}")
    if retrieval_strength < cfg.retrieval_floor:
        gates.append(
            f"retrieval_below_floor:{retrieval_strength:.2f}<{cfg.retrieval_floor}"
        )

    # ---- band and route ---------------------------------------------------------------
    if gates:
        band = Band.LOW
        route = Route.ESCALATE
        reasons.append("hard gate tripped — routed to a human regardless of score")
    elif score >= cfg.high:
        band, route = Band.HIGH, Route.AUTO_RESOLVE
    elif score >= cfg.medium:
        band, route = Band.MEDIUM, Route.AGENT_ASSIST
    else:
        band, route = Band.LOW, Route.ESCALATE

    return Decision(
        band=band, route=route, score=score, signals=signals, gates=gates, reasons=reasons
    )


def cheap_signals_already_decided(
    *,
    retrieval_strength: float,
    citation: CitationAudit,
    answerable: bool,
    policy_intents: list[str],
    sensitive_escalate: list[str],
    injection: bool,
) -> bool:
    """Should we skip the verifier call?

    The verifier is the most expensive step in the pipeline. It is worth paying for only when it can
    change the outcome. If a hard gate has already tripped, the ticket is going to a human whatever
    the verifier says, and spending a second model call to confirm a decision that is already made is
    pure cost. Roughly a fifth of traffic short-circuits here; the saving is quantified in the
    README's cost table.
    """
    return bool(
        injection
        or sensitive_escalate
        or policy_intents
        or not answerable
        or citation.invalid_ids
        or retrieval_strength < CONFIG.confidence.retrieval_floor
    )
