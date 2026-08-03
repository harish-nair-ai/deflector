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
class Config:
    models: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
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
