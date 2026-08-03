"""Retrieval-only evaluation.

The end-to-end eval tells you a case failed. It does not tell you *where*, and that ambiguity is
expensive: a retrieval miss and a generation miss look identical at the outcome level but need
completely different fixes. Measuring retrieval on its own separates them.

It is also the only honest way to evaluate a retrieval change. Swapping in a reranker and watching
end-to-end pass rates move tells you almost nothing, because the generator can paper over a mediocre
retrieval set and a hard gate can mask a good one. Here the generator is not involved at all.

Labels come from `expect_docs` in the same golden set — already written, no extra annotation.

Metrics, and why each is here:

  Recall@k     Did the right document appear in the top k at all? This is the ceiling on
               everything downstream — a document not retrieved cannot be cited.
  MRR          How high up? 1.0 means first place, 0.5 means second. Rank matters because the
               generator attends to earlier sources more, and because top_k=4 is a hard cut.
  Precision@4  What fraction of what we actually send the model is relevant? Low precision is
               wasted tokens and a distraction surface.
  Hit@1        The strictest view: was the very first result right?

Run:
    python evals/run_retrieval_eval.py                 # current config
    python evals/run_retrieval_eval.py --compare       # BM25 / dense / hybrid / hybrid+rerank
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deflector.config import CONFIG  # noqa: E402
from deflector.providers import Provider, load_dotenv  # noqa: E402
from deflector.retrieval import Retriever  # noqa: E402

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"
)


def docs_of(hits) -> list[str]:
    return [h.chunk.doc_id for h in hits]


def evaluate(retriever: Retriever, cases: list[dict], mode: str, k: int) -> dict:
    """Score one retrieval configuration over every labelled case."""
    recall_hits = 0
    hit_at_1 = 0
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    latencies: list[float] = []
    failures: list[tuple[str, list[str], list[str]]] = []

    for case in cases:
        expected = case["expect_docs"]
        started = time.perf_counter()
        hits = retriever.search(
            f"{case.get('subject','')} {case['body']}".strip(), top_k=k, mode=mode
        )
        latencies.append((time.perf_counter() - started) * 1000)

        retrieved = docs_of(hits)
        relevant = [d in expected for d in retrieved]

        if any(relevant):
            recall_hits += 1
            first = relevant.index(True)
            reciprocal_ranks.append(1.0 / (first + 1))
            if first == 0:
                hit_at_1 += 1
        else:
            reciprocal_ranks.append(0.0)
            failures.append((case["id"], expected, sorted(set(retrieved))))

        precisions.append(sum(relevant) / len(relevant) if relevant else 0.0)

    n = len(cases)
    latencies.sort()
    return {
        "mode": mode,
        "n": n,
        "recall_at_k": recall_hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "precision_at_k": sum(precisions) / n,
        "hit_at_1": hit_at_1 / n,
        "p50_ms": latencies[len(latencies) // 2],
        "failures": failures,
    }


def print_row(result: dict, baseline: dict | None = None) -> None:
    def delta(key: str) -> str:
        if baseline is None or baseline["mode"] == result["mode"]:
            return ""
        diff = result[key] - baseline[key]
        if abs(diff) < 0.0005:
            return f"  {DIM}  ·   {RESET}"
        colour = GREEN if diff > 0 else RED
        return f"  {colour}{diff:+.3f}{RESET}"

    print(
        f"  {result['mode']:<16} "
        f"{result['recall_at_k']:>7.1%}{delta('recall_at_k')}  "
        f"{result['mrr']:>6.3f}{delta('mrr')}  "
        f"{result['precision_at_k']:>7.1%}{delta('precision_at_k')}  "
        f"{result['hit_at_1']:>6.1%}{delta('hit_at_1')}  "
        f"{result['p50_ms']:>7.1f}ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Compare retrieval strategies")
    parser.add_argument("--k", type=int, default=CONFIG.retrieval.top_k)
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "evals" / "results" / "retrieval.json"
    )
    args = parser.parse_args()

    load_dotenv()
    cases = yaml.safe_load((ROOT / "evals" / "golden.yaml").read_text(encoding="utf-8"))["cases"]
    labelled = [c for c in cases if c.get("expect_docs")]

    provider = Provider()
    retriever = Retriever(provider)
    retriever.build_index()

    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}  RETRIEVAL-ONLY EVALUATION{RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")
    print(
        f"{DIM}{len(labelled)} labelled cases · {len(retriever.chunks)} chunks · "
        f"top_k={args.k} · dense={'on' if retriever.dense_available else 'off'}{RESET}\n"
    )

    modes = ["bm25", "dense", "hybrid"]
    if args.compare and retriever.reranker_available:
        modes.append("hybrid+rerank")
    elif args.compare:
        print(f"{YELLOW}  reranker unavailable — install with: uv pip install '.[rerank]'{RESET}\n")

    if not args.compare:
        modes = ["hybrid+rerank" if retriever.reranker_available else "hybrid"]

    print(f"  {BOLD}{'strategy':<16} {'recall@k':>7}  {'MRR':>6}  {'prec@k':>7}  {'hit@1':>6}  {'p50':>9}{RESET}")
    print("  " + "─" * 74)

    results = []
    baseline = None
    for mode in modes:
        result = evaluate(retriever, labelled, mode, args.k)
        if baseline is None:
            baseline = result
        print_row(result, baseline)
        results.append(result)
    print()

    worst = results[-1]
    if worst["failures"]:
        print(f"  {BOLD}Cases where no expected document was retrieved at all:{RESET}")
        for case_id, expected, got in worst["failures"]:
            print(f"    {RED}✗{RESET} {case_id}")
            print(f"        expected one of: {expected}")
            print(f"        got:             {got}")
        print()
    else:
        print(f"  {GREEN}every labelled case retrieved at least one expected document{RESET}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            [{k: v for k, v in r.items() if k != "failures"} for r in results], indent=2
        ),
        encoding="utf-8",
    )
    print(f"{DIM}wrote {args.out.relative_to(ROOT)}{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
