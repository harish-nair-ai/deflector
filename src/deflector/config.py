"""Every tunable number in the system lives here.

Deliberate: the operating point of a deflection system is a business decision, not a code change.
Support leadership should be able to move the auto-resolve bar without touching logic, and the eval
harness should be able to sweep it. Nothing else in the codebase hardcodes a threshold.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "corpus"
CACHE_DIR = ROOT / ".cache"
INDEX_PATH = CACHE_DIR / "index.json"
LLM_CACHE_DIR = CACHE_DIR / "llm"


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Two different models, on purpose.

    The answerer writes; a *different, larger* model verifies. Using the same model to grade its own
    output invites self-preference bias — a model rates its own text as well-supported far more
    often than an independent grader does. The verifier is the expensive call, so it only runs when
    the cheap signals have not already decided the outcome.
    """

    answerer: str = os.getenv("DEFLECTOR_ANSWERER", "openai/gpt-oss-20b:free")
    verifier: str = os.getenv("DEFLECTOR_VERIFIER", "inclusionai/ling-3.0-flash:free")
    embedder: str = os.getenv("DEFLECTOR_EMBEDDER", "openai/text-embedding-3-small")

    # Free-tier models rate-limit upstream. Fall through rather than fail the ticket.
    # Fallbacks are deliberately from different families than the primary: when a provider degrades
    # it degrades for everyone on that provider, so a same-family fallback fails at the same moment.
    answerer_fallbacks: tuple[str, ...] = (
        "google/gemma-4-31b-it:free",
        "inclusionai/ling-3.0-flash:free",
    )
    verifier_fallbacks: tuple[str, ...] = (
        "google/gemma-4-26b-a4b-it:free",
        "openai/gpt-oss-20b:free",
    )

    answerer_max_tokens: int = 900
    verifier_max_tokens: int = 700
    temperature: float = 0.0


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalConfig:
    chunk_target_words: int = 220      # split oversized sections at ~this size
    chunk_overlap_words: int = 40
    candidates_per_arm: int = 12       # BM25 and dense each return this many
    rrf_k: int = 60                    # standard RRF constant
    top_k: int = 4                     # chunks that reach the prompt

    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Two-stage retrieval: fusion narrows the corpus cheaply, a cross-encoder reorders that pool.
    #
    # OFF BY DEFAULT, AND THAT IS A MEASURED DECISION, NOT AN OVERSIGHT.
    # `make eval-retrieval` compares all four strategies on the golden set. On this corpus reranking
    # is a net loss: MRR 1.000 -> 0.965, hit@1 100% -> 94.7%, p50 31ms -> 143ms, in exchange for
    # +1.3pp precision@4. Two reasons. First, hybrid retrieval is already saturated here — recall@4
    # is 100% and MRR is 1.000, so there is no headroom for a reranker to win and every reordering it
    # makes is a chance to lose. Second, ms-marco cross-encoders are trained on web-search prose,
    # and half this corpus is table rows shaped like "Plan: Growth | Sustained RPM: 1,200", which is
    # nothing like their training distribution.
    #
    # It stays in the codebase because it is the right answer at a larger corpus, where recall stops
    # being free and fusion starts burying relevant chunks below the cut. Enable with
    # DEFLECTOR_RERANK=1 and re-measure before trusting it.
    rerank_enabled: bool = os.getenv("DEFLECTOR_RERANK", "") != ""
    rerank_candidates: int = 12


# --------------------------------------------------------------------------------------
# Confidence and routing
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfidenceConfig:
    """Weights for the blended confidence score.

    Note the deliberate ordering. The model's *self-reported* confidence is the least trusted signal
    and carries the smallest weight: LLMs are poorly calibrated about their own groundedness and
    skew high. The signals that can be checked mechanically — did every claim carry a citation, did
    the citation point at a real retrieved chunk — carry more weight than anything self-reported.
    """

    # Attach the inline [S1] marker when the model cited exactly one source and used no markers.
    # OFF, and the reason is the most useful thing I measured on this project.
    #
    # The hypothesis was that a missing marker is a formatting slip: same answer, same sources, the
    # model just forgot the brackets. It looked obviously true — 11 of 17 answerable cases scored 0.0
    # citation coverage while the verifier scored those same answers 1.0. Repairing it should have
    # recovered deflection for free.
    #
    # It did recover deflection, and it broke precision:
    #
    #     repair      auto-resolve rate   precision   wrongly auto-resolved
    #     off              10.3%            100.0%            0
    #     on               17.9%             71.4%            2
    #
    # The extra auto-resolves were not good answers that had been unfairly held back. They were
    # `table-service-credit`, which must never auto-resolve, and `webhook-retry-window`, which
    # auto-sent an answer missing the retry count it was asked for. So the missing marker was not
    # noise — it was *correlated with the answer being weak*. The model drops citations precisely
    # when it is synthesising loosely rather than lifting a fact off a source, which is exactly when
    # it should not be trusted. Repairing the format destroyed a real signal and kept the score.
    #
    # Left in the codebase and switchable because the finding is worth more than the code: flip this
    # to True and re-run `make eval` to reproduce the precision drop.
    repair_single_source_citations: bool = False

    w_verifier: float = 0.40      # independent model's support judgement
    w_retrieval: float = 0.25     # did we actually find relevant material
    w_citation: float = 0.25      # mechanically verified citation coverage
    w_self_report: float = 0.10   # the model's own opinion, deliberately last

    high: float = 0.72            # >= this and no hard gate  -> AUTO_RESOLVE
    medium: float = 0.45          # >= this                   -> AGENT_ASSIST

    retrieval_floor: float = 0.28  # below this, treat as out-of-corpus regardless of the answer
    margin_floor: float = 0.015    # top1 - top2; below this the corpus is ambiguous

    def weights_sum(self) -> float:
        return self.w_verifier + self.w_retrieval + self.w_citation + self.w_self_report


# --------------------------------------------------------------------------------------
# Policy: things that escalate no matter how confident the model is
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyConfig:
    """Confidence is not authority.

    A perfectly grounded, perfectly cited answer about refund policy still must not be auto-sent,
    because the *action* behind it moves money and is hard to reverse. These intents route to a human
    at any confidence. This is the distinction between "the model knows the answer" and "the model
    is allowed to end the conversation" — the two are constantly conflated, and conflating them is
    how support automation ends up issuing refunds it should not have.
    """

    escalate_intents: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "refund_or_credit": (
                "refund", "refunded", "money back", "chargeback", "dispute the charge",
                "credit my account", "reverse the charge", "waive",
            ),
            "cancellation": (
                "cancel my", "cancel our", "terminate our contract", "close my account",
                "downgrade immediately", "end our subscription",
            ),
            "legal_or_compliance": (
                "gdpr", "dpa", "data processing agreement", "our lawyer", "legal team",
                "breach of contract", "sue", "liability", "subpoena",
            ),
            "security_incident": (
                "leaked", "compromised", "unauthorized access", "security incident",
                "someone else accessed", "hacked", "exposed our key",
            ),
            "data_deletion": (
                "delete all our data", "erase my data", "right to be forgotten",
                "permanently delete",
            ),
            "outage_or_escalation": (
                "production is down", "everything is failing", "complete outage",
                "speak to a manager", "escalate this", "this is urgent and",
            ),
        }
    )

    # Requests for someone to *do* something, as opposed to questions about how something works.
    #
    # These do not escalate — escalating them would waste a human on a ticket whose question part the
    # system can answer perfectly well. They cap the route at agent_assist: draft the answer, let a
    # person perform the action and send it.
    #
    # The failure this prevents is subtle and was caught by the golden set. "Send last month's
    # invoice to priya@… and cc me. What roles can access invoices?" is two requests wearing one
    # ticket. The roles question is answerable and the system answered it, cited it, and scored 0.887
    # — comfortably auto-resolve. Auto-sending that reply tells a customer their invoice request was
    # handled when nothing was sent and no invoice went anywhere. A confident, correct, well-grounded
    # answer to the wrong half of the ticket is still a false resolution.
    #
    # Patterns are deliberately anchored on an imperative or a direct request ("please send",
    # "can you send") rather than the bare verb, so "how do I send a webhook" stays auto-resolvable.
    assist_only_intents: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "action_requested": (
                r"\b(?:please|kindly|could you|can you|can we|would you|pls)\s+"
                r"(?:\w+\s+){0,2}?(?:send|forward|email|mail|resend|share|attach|issue|"
                r"update|change|reset|add|remove|enable|disable|extend|raise|increase)\b",
                r"\bcc\s+me\b",
                # Indirect object must be a person, not an article. An earlier draft accepted
                # "send a|the", which fired on "how do I send a webhook payload" and on the
                # corpus's own prose ("the server will send a retry") — informational sentences,
                # not requests. Restricting to me/us/them, or to a named deliverable, removes both.
                r"\b(?:send|forward|email|mail|resend)\s+(?:me|us|them)\b",
                r"\b(?:send|forward|email|mail|resend)\b[^.?!]{0,30}?"
                r"\b(?:copy|invoice|receipt|statement|report|export|transcript)\b",
                r"\bon\s+my\s+behalf\b",
            ),
        }
    )

    # Detected prompt-injection attempts always go to a human, and never get auto-answered.
    injection_patterns: tuple[str, ...] = (
        r"ignore (?:all |any |your )?(?:previous |prior |above )?instructions",
        r"disregard (?:the |your )?(?:above|previous|prior|system)",
        r"you are now (?:a|an|in)\b",
        r"system prompt",
        r"reveal (?:your |the )?(?:prompt|instructions|system)",
        r"act as (?:if you are )?(?:an? )?(?:admin|administrator|developer|root)",
        r"override (?:the )?(?:policy|rules|guardrails?)",
        r"\bDAN\b mode",
        r"print (?:your |the )(?:instructions|prompt|rules)",
    )


@dataclass(frozen=True)
class SemanticCacheConfig:
    """Answer reuse across differently-worded tickets.

    The threshold is the whole design. Semantic caching fails in exactly one direction: two questions
    that are nearly identical but differ in the one word that decides the answer — "the Growth rate
    limit" versus "the Starter rate limit" — sit very close together in embedding space. 0.95 is
    deliberately conservative, because a miss costs a fraction of a cent and a wrong hit is the
    failure this entire system exists to prevent.
    """

    # OFF BY DEFAULT. MEASURED, NOT ASSUMED — see `evals/measure_semantic_cache.py`.
    #
    # Cosine similarity to a seeded "what is the Growth rate limit?" answer:
    #     genuine paraphrase  "hitting the API limit on Growth, what's the rpm cap"   0.653
    #     genuine paraphrase  "Growth plan: how many requests a minute before 429s"   0.784
    #     WRONG PLAN          "we're on Starter ... what is our sustained rate limit"  0.816   <-- highest
    #
    # The near-miss outscores both real paraphrases, because it differs from the seed by one word
    # while the paraphrases differ by many. There is therefore NO threshold that admits the
    # paraphrases and rejects the wrong-plan question, and a hit there would serve a Starter customer
    # the Growth limits — a confidently wrong answer about their account, which is the exact failure
    # this whole system exists to prevent.
    #
    # Keying on the retrieved chunk set instead does not rescue it: paraphrase-2 and the wrong-plan
    # question both overlap the seed's chunk set at Jaccard 0.333, so that signal does not
    # discriminate either. Requiring an identical chunk set degenerates to an exact-string cache.
    #
    # The failure is specific to *parametric* questions — same shape, different entity — which is
    # most of support. Making this safe needs entity-aware keying (extract the plan/tier/region and
    # require an exact match), not a higher threshold. Until that exists, it stays off.
    enabled: bool = os.getenv("DEFLECTOR_SEMANTIC_CACHE", "") != ""
    similarity_threshold: float = 0.95
    ttl_seconds: float = 60 * 60 * 24 * 7      # a week; knowledge-base edits should win eventually
    max_entries: int = 5000


@dataclass(frozen=True)
class Config:
    models: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    cache: SemanticCacheConfig = field(default_factory=SemanticCacheConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    api_base: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    request_timeout: float = float(os.getenv("DEFLECTOR_TIMEOUT", "90"))

    # Record/replay. On by default so the eval is reproducible without a key.
    cache_enabled: bool = os.getenv("DEFLECTOR_NO_CACHE", "") == ""
    offline: bool = os.getenv("DEFLECTOR_OFFLINE", "") != ""

    @property
    def api_key(self) -> str | None:
        return os.getenv("OPENROUTER_API_KEY") or None


CONFIG = Config()
