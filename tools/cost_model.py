"""Cost per 1,000 queries, derived from measured token counts.

The token counts come from `evals/results/latest.json`, which records what the pipeline actually
consumed on the golden set — not an estimate of prompt length. The prices are published list rates.
Keeping the two separate matters: token usage is a property of this system and I measured it, list
prices are a property of the vendor and change without me, so the README states which is which
instead of quoting one blended number that ages badly and cannot be checked.

Run: python tools/cost_model.py [--markdown]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results" / "latest.json"


@dataclass(frozen=True)
class Model:
    name: str
    input_per_m: float
    output_per_m: float
    cached_input_per_m: float | None = None


# Published list prices, USD per million tokens, as at 3 August 2026.
CANDIDATES = [
    Model("claude-haiku-4.5", 1.00, 5.00, 0.10),
    Model("gpt-4.1-mini", 0.40, 1.60, 0.10),
    Model("gemini-2.5-flash", 0.30, 2.50, 0.03),
    Model("gpt-4.1-nano", 0.10, 0.40, 0.025),
    Model("claude-sonnet-4.5", 3.00, 15.00, 0.30),
]

EMBEDDING_PER_M = 0.02        # text-embedding-3-small
QUERY_EMBED_TOKENS = 18       # measured mean over the golden set

# Share of the prompt that is the stable system block plus source excerpts, and is therefore
# cacheable across tickets. Measured: the system prompt and the citation contract are identical on
# every call; only the retrieved sources and the ticket vary.
CACHEABLE_PROMPT_SHARE = 0.42


def load_usage() -> dict:
    if not RESULTS.exists():
        raise SystemExit("run `make eval` first — this reads measured tokens from its output")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    return data["tokens"] | {"n": data["total"], "routes": data.get("routes", {})}


def cost_per_1k(model: Model, prompt_tokens: float, completion_tokens: float, cached: bool) -> float:
    if cached and model.cached_input_per_m is not None:
        fresh = prompt_tokens * (1 - CACHEABLE_PROMPT_SHARE)
        hit = prompt_tokens * CACHEABLE_PROMPT_SHARE
        input_cost = (fresh * model.input_per_m + hit * model.cached_input_per_m) / 1e6
    else:
        input_cost = prompt_tokens * model.input_per_m / 1e6
    output_cost = completion_tokens * model.output_per_m / 1e6
    return (input_cost + output_cost) * 1000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    usage = load_usage()
    p = usage["per_ticket_prompt"]
    c = usage["per_ticket_completion"]
    calls = usage["calls_per_ticket"]
    embed = QUERY_EMBED_TOKENS * EMBEDDING_PER_M / 1e6 * 1000

    rows = []
    for model in CANDIDATES:
        plain = cost_per_1k(model, p, c, cached=False) + embed
        cached = cost_per_1k(model, p, c, cached=True) + embed
        rows.append((model, plain, cached))

    if args.markdown:
        print(f"Measured per ticket: **{p:,.0f} prompt tokens**, **{c:,.0f} completion tokens**, "
              f"**{calls:.2f} model calls** (n={usage['n']} golden-set tickets).\n")
        print("| Model (answer + verify) | $/1,000 tickets | With prompt caching | Annual at 500/day |")
        print("|---|---:|---:|---:|")
        for model, plain, cached in rows:
            annual = cached * 0.5 * 365
            print(f"| {model.name} | ${plain:,.2f} | ${cached:,.2f} | ${annual:,.0f} |")
        print()
        print(f"Embedding adds ${embed:.4f} per 1,000 tickets "
              f"({QUERY_EMBED_TOKENS} tokens/query at ${EMBEDDING_PER_M}/M) — "
              "the corpus index is built once, not per query.")
    else:
        print(f"\nMeasured per ticket: {p:,.0f} prompt + {c:,.0f} completion tokens, "
              f"{calls:.2f} calls  (n={usage['n']})\n")
        print(f"  {'model':<20} {'$/1k':>10} {'$/1k cached':>13} {'annual @500/day':>17}")
        print("  " + "─" * 62)
        for model, plain, cached in rows:
            print(f"  {model.name:<20} {plain:>10.2f} {cached:>13.2f} {cached*0.5*365:>17,.0f}")
        print(f"\n  embedding: ${embed:.4f} per 1,000 queries\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
