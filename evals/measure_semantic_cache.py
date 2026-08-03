"""Measure whether a semantic answer cache is safe on this corpus.

This script exists to justify a *negative* decision. The semantic cache is implemented and shipped
disabled, and this is the evidence for that — kept runnable so the claim can be checked rather than
taken on trust, and re-checked if the corpus or embedding model changes.

The question it answers: is there a similarity threshold that admits genuine rephrasings of a
question while rejecting a question that looks almost identical but is about a different entity?

Run: python evals/measure_semantic_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deflector.providers import Provider, load_dotenv  # noqa: E402
from deflector.retrieval import Retriever, cosine  # noqa: E402

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"
)

SEED = (
    "429s on Growth We're on the Growth plan and started getting 429 responses. "
    "What is our sustained rate limit?"
)

PROBES = [
    ("paraphrase", "We keep hitting the API limit on Growth - what is the requests per minute cap?"),
    ("paraphrase", "Growth plan: how many requests a minute are we allowed before 429s?"),
    ("paraphrase", "What's the rpm ceiling for a Growth account before we start seeing 429 errors?"),
    ("WRONG ENTITY", "We're on the Starter plan and started getting 429 responses. What is our sustained rate limit?"),
    ("WRONG ENTITY", "We're on the Enterprise plan and started getting 429 responses. What is our sustained rate limit?"),
    ("unrelated", "How long do you retain webhook delivery logs?"),
]


def main() -> int:
    load_dotenv()
    provider = Provider()
    retriever = Retriever(provider)
    retriever.build_index()

    seed_vec = provider.embed([SEED])[0]
    seed_chunks = frozenset(h.chunk.chunk_id for h in retriever.search(SEED))

    print(f"\n{BOLD}{'='*84}{RESET}")
    print(f"{BOLD}  SEMANTIC CACHE SAFETY{RESET}")
    print(f"{BOLD}{'='*84}{RESET}\n")
    print(f"{DIM}seed: {SEED[:78]}{RESET}\n")
    print(f"  {BOLD}{'kind':<14} {'cosine':>7}  {'chunk Jaccard':>14}   question{RESET}")
    print("  " + "─" * 80)

    rows = []
    for kind, probe in PROBES:
        sim = cosine(seed_vec, provider.embed([probe])[0])
        chunks = frozenset(h.chunk.chunk_id for h in retriever.search(probe))
        jaccard = len(seed_chunks & chunks) / len(seed_chunks | chunks)
        rows.append((kind, sim, jaccard, probe))
        colour = RED if kind == "WRONG ENTITY" else (GREEN if kind == "paraphrase" else DIM)
        print(f"  {colour}{kind:<14}{RESET} {sim:>7.3f}  {jaccard:>14.3f}   {probe[:44]}…")

    paraphrases = [r[1] for r in rows if r[0] == "paraphrase"]
    wrong = [r[1] for r in rows if r[0] == "WRONG ENTITY"]

    print("\n  " + "─" * 80)
    print(f"  lowest genuine paraphrase : {min(paraphrases):.3f}")
    print(f"  highest WRONG-entity probe: {max(wrong):.3f}")
    print()

    if max(wrong) >= min(paraphrases):
        print(
            f"  {RED}{BOLD}NO SAFE THRESHOLD EXISTS.{RESET} A wrong-entity question scores "
            f"{max(wrong):.3f}, above the weakest\n  genuine paraphrase at {min(paraphrases):.3f}. "
            "Any threshold that produces cache hits on real\n  rephrasings would also serve one "
            f"customer's answer to a different plan's question.\n\n  {BOLD}Verdict: semantic cache "
            f"stays disabled.{RESET} Fixing it needs entity-aware keying,\n  not a higher threshold.\n"
        )
        return 1

    print(
        f"  {GREEN}A safe threshold exists between {max(wrong):.3f} and {min(paraphrases):.3f}."
        f"{RESET} Re-enable and set it there.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
