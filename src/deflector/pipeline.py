"""The deflection pipeline.

    screen -> retrieve -> answer -> audit citations -> verify -> score -> route

Two properties are non-negotiable and shape the code:

**It never raises on a ticket.** Every failure mode — no sources, provider down, unparseable model
output, verifier timeout — converges on an escalation with a stated reason. A support system whose
error path is a 500 has simply moved the queue, not shortened it. `deflect()` returns a Decision in
all cases.

**Every decision is fully explainable.** The response carries the retrieved chunks with their scores,
the four confidence signals, the gates that tripped, the prompt version and the models used. When
support leadership asks why ticket 4471 was auto-answered, the answer is in the record, not in a
reconstruction.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .config import CONFIG
from .confidence import (
    Band,
    CitationAudit,
    Decision,
    Route,
    audit_citations,
    cheap_signals_already_decided,
    score_confidence,
)
from .corpus import Chunk, load_chunks
from .guardrails import match_policy_intents, screen
from .providers import Provider, Usage
from .retrieval import Hit, Retriever


@dataclass
class DeflectionResult:
    ticket_id: str
    question: str
    answer: str
    band: Band
    route: Route
    score: float
    citations: list[dict[str, str]]
    sources: list[dict[str, Any]]
    decision: Decision
    screening: dict[str, Any]
    usage: Usage
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "route": self.route.value,
            "confidence_band": self.band.value,
            "confidence_score": round(self.score, 4),
            "answer": self.answer,
            "citations": self.citations,
            "decision": self.decision.to_dict(),
            "screening": self.screening,
            "retrieved": self.sources,
            "usage": self.usage.to_dict(),
            "meta": self.meta,
        }

    def customer_facing(self) -> str | None:
        """Only a HIGH-band auto-resolve is ever sent without a human."""
        return self.answer if self.route is Route.AUTO_RESOLVE else None


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict | None:
    """Models wrap JSON in fences, prefix it with prose, or emit it after reasoning.

    Rather than demand perfection from a free-tier model, extract the outermost object and parse it.
    A failure here returns None and the caller escalates — a ticket is never dropped because a model
    was chatty.
    """
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(cleaned)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        # Trailing commas are the single most common malformation.
        try:
            return json.loads(re.sub(r",(\s*[}\]])", r"\1", match.group()))
        except json.JSONDecodeError:
            return None


class Deflector:
    def __init__(self, provider: Provider | None = None, chunks: list[Chunk] | None = None) -> None:
        self.provider = provider or Provider()
        self.retriever = Retriever(self.provider, chunks=chunks)
        self.retriever.build_index()

    # -----------------------------------------------------------------------------------

    def deflect(
        self, body: str, subject: str = "", ticket_id: str | None = None
    ) -> DeflectionResult:
        ticket_id = ticket_id or f"tkt_{uuid.uuid4().hex[:10]}"
        started = time.perf_counter()
        usage = Usage()

        # -- 1. screen before anything else ---------------------------------------------
        # Redaction happens here so raw secrets never reach the provider, not after the fact.
        # Subject and body are screened separately: they are separate fields in the prompt, and
        # screening them as one blob duplicates the subject into the body.
        combined = f"{subject}\n{body}".strip()
        screened = screen(combined)          # used only for the detection verdict
        screened_body = screen(body)
        screened_subject = screen(subject).text if subject else "(none)"
        policy_intents = match_policy_intents(combined)
        safe_query = f"{screened_subject if subject else ''} {screened_body.text}".strip()

        # -- 2. retrieve ----------------------------------------------------------------
        hits: list[Hit] = self.retriever.search(safe_query)
        strength, margin = self.retriever.strength(hits)
        sources_block = prompts.format_sources(hits)

        # -- 3. answer ------------------------------------------------------------------
        answer_text = ""
        answerable = False
        self_report = 0.0
        model_injection_flag = False
        missing_info = ""
        cited_ids: list[str] = []
        parse_failed = False

        if hits:
            result = self.provider.chat(
                system=prompts.ANSWER_SYSTEM,
                user=prompts.ANSWER_USER.format(
                    sources=sources_block,
                    subject=screened.text.split("\n")[0] if subject else "(none)",
                    body=screened.text,
                ),
                model=CONFIG.models.answerer,
                fallbacks=CONFIG.models.answerer_fallbacks,
                max_tokens=CONFIG.models.answerer_max_tokens,
            )
            usage.add(result.usage)

            payload = parse_json_object(result.text) if result.ok else None
            if payload is None:
                parse_failed = True
            else:
                answerable = bool(payload.get("answerable", False))
                answer_text = str(payload.get("answer") or "").strip()
                self_report = _as_float(payload.get("confidence"), 0.0)
                model_injection_flag = bool(payload.get("injection_suspected", False))
                missing_info = str(payload.get("missing_information") or "")
                cited_ids = [str(c) for c in (payload.get("citations") or [])]

        # -- 4. audit citations (mechanical, no model) ----------------------------------
        citation = audit_citations(answer_text, len(hits))

        # -- 5. verify, but only when it can change the outcome -------------------------
        injection = bool(screened.injection_hits) or model_injection_flag
        skip_verifier = cheap_signals_already_decided(
            retrieval_strength=strength,
            citation=citation,
            answerable=answerable and not parse_failed,
            policy_intents=policy_intents,
            sensitive_escalate=screened.escalate_kinds,
            injection=injection,
        )

        verifier_ran = False
        v_supported: int | None = None
        v_unsupported: int | None = None
        v_detail: list[str] = []

        if not skip_verifier and answer_text:
            verdict = self.provider.chat(
                system=prompts.VERIFY_SYSTEM,
                user=prompts.VERIFY_USER.format(sources=sources_block, answer=answer_text),
                model=CONFIG.models.verifier,
                fallbacks=CONFIG.models.verifier_fallbacks,
                max_tokens=CONFIG.models.verifier_max_tokens,
            )
            usage.add(verdict.usage)
            payload = parse_json_object(verdict.text) if verdict.ok else None
            if payload is not None:
                verifier_ran = True
                v_supported = int(_as_float(payload.get("supported_claims"), 0))
                v_unsupported = int(_as_float(payload.get("unsupported_claims"), 0))
                v_detail = [str(d) for d in (payload.get("unsupported_detail") or [])][:5]
                if payload.get("citations_valid") is False:
                    citation.invalid_ids = sorted(set(citation.invalid_ids + ["verifier_flagged"]))

        # -- 6. score and route ---------------------------------------------------------
        decision = score_confidence(
            retrieval_strength=strength,
            retrieval_margin=margin,
            citation=citation,
            verifier_supported=v_supported,
            verifier_unsupported=v_unsupported,
            verifier_ran=verifier_ran,
            verifier_skipped=skip_verifier,
            self_report=self_report,
            answerable=answerable and not parse_failed,
            policy_intents=policy_intents,
            sensitive_escalate=screened.escalate_kinds,
            injection=injection,
            n_sources=len(hits),
        )

        if parse_failed:
            decision.gates.append("model_output_unparseable")
            decision.band, decision.route = Band.LOW, Route.ESCALATE
        if v_detail:
            decision.reasons.extend(f"unsupported: {d}" for d in v_detail)
        if missing_info:
            decision.reasons.append(f"model flagged missing: {missing_info}")

        # -- 7. shape the reply ---------------------------------------------------------
        if decision.route is Route.ESCALATE and not answer_text:
            answer_text = _escalation_note(decision, missing_info)

        citations = [
            {
                "marker": f"S{i}",
                "doc_id": hit.chunk.doc_id,
                "title": hit.chunk.doc_title,
                "section": hit.chunk.section,
                "last_reviewed": hit.chunk.last_reviewed,
            }
            for i, hit in enumerate(hits, start=1)
            if f"S{i}" in set(cited_ids) or f"[S{i}]" in answer_text
        ]

        return DeflectionResult(
            ticket_id=ticket_id,
            question=body,
            answer=answer_text,
            band=decision.band,
            route=decision.route,
            score=decision.score,
            citations=citations,
            sources=[hit.to_dict() for hit in hits],
            decision=decision,
            screening=screened.to_dict() | {"policy_intents": policy_intents},
            usage=usage,
            meta={
                "prompt_version": prompts.PROMPT_VERSION,
                "answerer": CONFIG.models.answerer,
                "verifier": CONFIG.models.verifier if verifier_ran else None,
                "verifier_skipped": skip_verifier,
                "dense_retrieval": self.retriever.dense_available,
                "retrieval_strength": round(strength, 4),
                "retrieval_margin": round(margin, 4),
                "citation_coverage": round(citation.coverage, 3),
                "claim_sentences": citation.claim_sentences,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )


def _escalation_note(decision: Decision, missing: str) -> str:
    """What the human sees when nothing was auto-answerable.

    Written for the agent picking the ticket up, not for the customer. It states why the system
    stepped back, which is the difference between a useful handoff and a ticket that lands with no
    context and has to be diagnosed from scratch.
    """
    lines = ["Routed to a human. No reply was sent to the customer."]
    if decision.gates:
        lines.append("Reason: " + "; ".join(decision.gates))
    if missing:
        lines.append(f"Not covered by the knowledge base: {missing}")
    return "\n".join(lines)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
