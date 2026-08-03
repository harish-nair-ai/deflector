"""Sensitive-data detection, redaction, and prompt-injection screening.

**The central design decision: two tiers, not one.**

The naive version of "escalate when sensitive data is detected" flags every ticket, because every
support ticket contains an email address and most contain a phone number. A guardrail that fires on
100% of traffic is not a guardrail; it is an outage. So detections are split by what the right
response actually is:

  REDACT   — PII that is normal in a support ticket and is not itself a risk signal.
             Email, phone, IP address. These are masked before the text reaches the LLM (so they
             stay out of the provider's logs) and the ticket proceeds normally.

  ESCALATE — data that should never have been pasted into a ticket at all.
             Card numbers, API secrets, private keys, bank details, government IDs. Their presence
             means the customer has exposed a credential or the ticket carries payment data, and
             both need a human plus, usually, a rotation or a compliance step. These are redacted
             *and* routed to a human.

**Why the Luhn check matters.** A bare 16-digit regex flags invoice numbers, order IDs, request IDs
and tracking numbers. Validating the check digit removes essentially all of those false positives at
the cost of eleven lines of code. Getting this wrong is how a guardrail earns a reputation for crying
wolf, after which agents start ignoring it — which is worse than not having it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .config import CONFIG


class Tier(str, Enum):
    REDACT = "redact"
    ESCALATE = "escalate"


@dataclass
class Detection:
    kind: str
    tier: Tier
    count: int
    sample: str  # already masked — never carries the raw value


@dataclass
class ScreenResult:
    text: str                                   # redacted text, safe to send to the model
    detections: list[Detection] = field(default_factory=list)
    injection_hits: list[str] = field(default_factory=list)

    @property
    def must_escalate(self) -> bool:
        return any(d.tier is Tier.ESCALATE for d in self.detections) or bool(self.injection_hits)

    @property
    def escalate_kinds(self) -> list[str]:
        return sorted({d.kind for d in self.detections if d.tier is Tier.ESCALATE})

    def to_dict(self) -> dict:
        return {
            "detections": [
                {"kind": d.kind, "tier": d.tier.value, "count": d.count, "sample": d.sample}
                for d in self.detections
            ],
            "injection_hits": self.injection_hits,
        }


def luhn_valid(digits: str) -> bool:
    """Standard mod-10 checksum used by every major card scheme."""
    digits = re.sub(r"\D", "", digits)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Ordered: the most specific patterns run first so a JWT is not also matched as a generic token.
_PATTERNS: list[tuple[str, Tier, re.Pattern[str]]] = [
    ("private_key", Tier.ESCALATE,
     re.compile(r"-----BEGIN[ A-Z]*PRIVATE KEY-----")),
    ("jwt", Tier.ESCALATE,
     re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\b")),
    ("aws_access_key", Tier.ESCALATE,
     re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("provider_secret", Tier.ESCALATE,
     re.compile(r"\b(?:sk|rk)-(?:live|test|proj|or-v1)?[-_]?[A-Za-z0-9]{20,}\b")),
    ("github_token", Tier.ESCALATE,
     re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b")),
    ("slack_token", Tier.ESCALATE,
     re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("meridian_live_key", Tier.ESCALATE,
     re.compile(r"\bmk_live_[A-Za-z0-9]{8,}\b")),
    ("password_in_clear", Tier.ESCALATE,
     re.compile(r"(?i)\b(?:my |our |the )?(?:password|passwd|pwd|passphrase)\s*(?:is|=|:)\s*\S{4,}")),
    ("iban", Tier.ESCALATE,
     re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("us_ssn", Tier.ESCALATE,
     re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    # The trailing guard matters: without it this pattern swallows the first twelve digits of a
    # space-separated card number ("4539 5678 9012" out of "4539 5678 9012 3458") and reports it as
    # a national ID. The leading guard stops it starting mid-way through a longer digit run.
    ("aadhaar", Tier.ESCALATE,
     re.compile(r"(?<!\d)(?<!\d\s)[2-9]\d{3}\s?\d{4}\s?\d{4}(?!\s?\d)")),
    # Redact tier: expected in normal tickets.
    ("email", Tier.REDACT,
     re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", Tier.REDACT,
     re.compile(r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\(\d{2,4}\)[ -]?)?\d{3,5}[ -]?\d{3,4}[ -]?\d{3,4}(?!\w)")),
    ("ip_address", Tier.REDACT,
     re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _mask(value: str, keep: int = 4) -> str:
    stripped = value.strip()
    if len(stripped) <= keep:
        return "*" * len(stripped)
    return "*" * (len(stripped) - keep) + stripped[-keep:]


def screen(text: str) -> ScreenResult:
    """Detect, redact, and flag. Returns text that is safe to put in a prompt."""
    detections: dict[str, Detection] = {}
    redacted = text

    # Cards first: a Luhn-valid number would otherwise be partly eaten by the phone pattern.
    card_spans: list[tuple[int, int]] = []
    for match in _CARD_CANDIDATE.finditer(redacted):
        if luhn_valid(match.group()):
            card_spans.append(match.span())
    if card_spans:
        detections["credit_card"] = Detection(
            kind="credit_card",
            tier=Tier.ESCALATE,
            count=len(card_spans),
            sample=_mask(redacted[card_spans[0][0]:card_spans[0][1]]),
        )
        for start, end in reversed(card_spans):
            redacted = redacted[:start] + "[REDACTED:credit_card]" + redacted[end:]

    for kind, tier, pattern in _PATTERNS:
        found = list(pattern.finditer(redacted))
        if not found:
            continue
        # A bare 4-6 digit run is an invoice or error number, not a phone number.
        if kind == "phone":
            found = [m for m in found if len(re.sub(r"\D", "", m.group())) >= 9]
            if not found:
                continue
        detections[kind] = Detection(
            kind=kind, tier=tier, count=len(found), sample=_mask(found[0].group())
        )
        for match in reversed(found):
            redacted = redacted[: match.start()] + f"[REDACTED:{kind}]" + redacted[match.end():]

    injection_hits = [
        pattern for pattern in CONFIG.policy.injection_patterns
        if re.search(pattern, text, re.IGNORECASE)
    ]

    return ScreenResult(
        text=redacted, detections=list(detections.values()), injection_hits=injection_hits
    )


def match_policy_intents(text: str) -> list[str]:
    """Intents that route to a human regardless of how well-grounded the answer is."""
    lowered = text.lower()
    return sorted(
        intent
        for intent, phrases in CONFIG.policy.escalate_intents.items()
        if any(phrase in lowered for phrase in phrases)
    )
