"""Evaluation harness.

The metric this system is judged on is **not** answer accuracy. It is:

    of the tickets we auto-resolved, how many were we right to auto-resolve?

That is auto-resolve precision, and it is the number that decides whether the deployment survives.
Deflection rate is the number the business wants, and the two trade against each other — you can
have any deflection rate you like if you stop caring about being right. So both are always reported
together, and `--sweep` shows the whole trade-off curve rather than one flattering point on it.

A wrong auto-resolve and an unnecessary escalation are not equal errors. The first sends a customer
a confidently wrong answer about their money; the second costs an agent four minutes. The harness
weights them accordingly by reporting a **false auto-resolve count** separately and never letting it
be averaged away into an aggregate score.

Run:
    python evals/run_eval.py                 # replays from cache — reproducible, no API key needed
    python evals/run_eval.py --fresh         # hits the API
    python evals/run_eval.py --sweep         # threshold sweep, prints the operating-point table
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deflector.config import CONFIG  # noqa: E402
from deflector.pipeline import Deflector  # noqa: E402
from deflector.providers import Provider, load_dotenv  # noqa: E402

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"
)

# A route is "acceptable" if it is at least as cautious as the expectation, except where the
# expectation is exactly auto_resolve — being over-cautious there is a miss on deflection, which we
# count separately rather than hiding.
ROUTE_ORDER = {"auto_resolve": 0, "agent_assist": 1, "escalate": 2}


@dataclass
class CaseResult:
    case_id: str
    type: str
    expected: str
    actual: str
    band: str
    score: float
    passed: bool
    failures: list[str] = field(default_factory=list)
    retrieved_docs: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    model_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    answer: str = ""


def route_ok(expected: str, actual: str) -> bool:
    if expected == "not_auto":
        return actual in ("agent_assist", "escalate")
    if expected == "escalate":
        return actual == "escalate"
    if expected == "auto_resolve":
        return actual == "auto_resolve"
    if expected == "agent_assist":
        return actual in ("agent_assist", "escalate")
    return True


def evaluate_case(deflector: Deflector, case: dict) -> CaseResult:
    started = time.perf_counter()
    result = deflector.deflect(
        body=case["body"], subject=case.get("subject", ""), ticket_id=case["id"]
    )
    elapsed = (time.perf_counter() - started) * 1000

    expected = case.get("expect_route", "")
    actual = result.route.value
    failures: list[str] = []

    if expected and not route_ok(expected, actual):
        failures.append(f"route: expected {expected}, got {actual}")

    answer_lower = result.answer.lower()

    # Content assertions only apply when something was actually produced for a customer or agent.
    if actual != "escalate":
        for needle in case.get("must_contain", []) or []:
            if needle.lower() not in answer_lower:
                failures.append(f"missing required text: {needle!r}")

    for needle in case.get("must_not_contain", []) or []:
        if needle.lower() in answer_lower:
            failures.append(f"contains forbidden text: {needle!r}")

    retrieved_docs = [s["chunk_id"].split("::")[0] for s in result.sources]
    expect_docs = case.get("expect_docs") or []
    if expect_docs and not any(d in retrieved_docs for d in expect_docs):
        failures.append(f"retrieval: none of {expect_docs} in {sorted(set(retrieved_docs))}")

    detected = {d["kind"] for d in result.screening.get("detections", [])}
    for kind in case.get("expect_detections", []) or []:
        if kind not in detected:
            failures.append(f"detector missed: {kind}")

    if case.get("expect_no_escalate_detections"):
        escalating = {
            d["kind"] for d in result.screening.get("detections", []) if d["tier"] == "escalate"
        }
        if escalating:
            failures.append(f"false positive on escalate tier: {sorted(escalating)}")

    usage = result.usage
    return CaseResult(
        case_id=case["id"],
        type=case.get("type", "unspecified"),
        expected=expected,
        actual=actual,
        band=result.band.value,
        score=result.score,
        passed=not failures,
        failures=failures,
        retrieved_docs=sorted(set(retrieved_docs)),
        elapsed_ms=elapsed,
        # Model time, not harness time. On a replayed run `elapsed` is disk-read speed, which is
        # not a latency any user experiences; the cache preserves what each call originally took.
        model_ms=usage.latency_ms,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        calls=usage.calls,
        answer=result.answer,
    )


def report(results: list[CaseResult], wall_seconds: float) -> dict:
    total = len(results)
    passed = sum(r.passed for r in results)

    by_type: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_type[r.type].append(r)

    auto = [r for r in results if r.actual == "auto_resolve"]
    false_auto = [r for r in auto if not r.passed]
    deflected = [r for r in results if r.actual in ("auto_resolve", "agent_assist")]

    should_not_auto = [
        r for r in results if r.expected in ("escalate", "not_auto")
    ]
    correctly_held = [r for r in should_not_auto if r.actual != "auto_resolve"]

    print(f"\n{BOLD}{'='*78}{RESET}")
    print(f"{BOLD}  DEFLECTOR — GOLDEN SET{RESET}")
    print(f"{BOLD}{'='*78}{RESET}\n")

    print(f"{BOLD}{'case':<38} {'type':<18} {'expected':<13} {'actual':<13} {'score':>6}{RESET}")
    print("─" * 92)
    for r in results:
        mark = f"{GREEN}✓{RESET}" if r.passed else f"{RED}✗{RESET}"
        colour = {"High": GREEN, "Medium": YELLOW, "Low": RED}.get(r.band, "")
        print(
            f"{mark} {r.case_id[:36]:<36} {r.type:<18} {r.expected:<13} "
            f"{colour}{r.actual:<13}{RESET} {r.score:>6.3f}"
        )
        for failure in r.failures:
            print(f"    {RED}└─ {failure}{RESET}")

    print("\n" + "─" * 92)
    print(f"{BOLD}BY CASE TYPE{RESET}")
    for name in sorted(by_type):
        group = by_type[name]
        ok = sum(g.passed for g in group)
        bar_colour = GREEN if ok == len(group) else (YELLOW if ok >= len(group) * 0.7 else RED)
        print(f"  {name:<22} {bar_colour}{ok}/{len(group)}{RESET}")

    print("\n" + "─" * 92)
    print(f"{BOLD}THE NUMBERS THAT MATTER{RESET}\n")

    auto_precision = (len(auto) - len(false_auto)) / len(auto) if auto else 1.0
    hold_recall = len(correctly_held) / len(should_not_auto) if should_not_auto else 1.0

    print(
        f"  {BOLD}Auto-resolve precision{RESET}   "
        f"{GREEN if auto_precision >= 0.95 else RED}{auto_precision:.1%}{RESET}"
        f"   ({len(auto) - len(false_auto)}/{len(auto)} auto-resolved answers were correct)"
    )
    print(
        f"  {BOLD}Wrongly auto-resolved{RESET}    "
        f"{GREEN if not false_auto else RED}{len(false_auto)}{RESET}"
        f"   {DIM}← the count that decides whether this ships{RESET}"
    )
    print(
        f"  {BOLD}Held back correctly{RESET}      {hold_recall:.1%}"
        f"   ({len(correctly_held)}/{len(should_not_auto)} of must-not-auto cases were held)"
    )
    print(
        f"  {BOLD}Deflection rate{RESET}          {len(deflected)/total:.1%}"
        f"   ({len(deflected)}/{total} handled without a human owning the ticket)"
    )
    print(f"  {BOLD}Auto-resolve rate{RESET}        {len(auto)/total:.1%}")
    print(f"  {BOLD}Overall case pass{RESET}        {passed}/{total} ({passed/total:.1%})")

    routes = Counter(r.actual for r in results)
    print(
        f"\n  routing mix: "
        f"{GREEN}auto {routes['auto_resolve']}{RESET} · "
        f"{YELLOW}assist {routes['agent_assist']}{RESET} · "
        f"{RED}escalate {routes['escalate']}{RESET}"
    )

    prompt_tokens = sum(r.prompt_tokens for r in results)
    completion_tokens = sum(r.completion_tokens for r in results)
    calls = sum(r.calls for r in results)
    latencies = sorted(r.model_ms for r in results if r.model_ms > 0)
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0

    print(f"\n{BOLD}COST AND LATENCY (measured, not estimated){RESET}\n")
    print(f"  model calls            {calls}  ({calls/total:.2f} per ticket)")
    print(f"  prompt tokens          {prompt_tokens:,}  ({prompt_tokens/total:,.0f} per ticket)")
    print(f"  completion tokens      {completion_tokens:,}  ({completion_tokens/total:,.0f} per ticket)")
    print(f"  latency p50 / p95      {p50/1000:.1f}s / {p95/1000:.1f}s")
    print(f"  wall clock             {wall_seconds:.0f}s for {total} cases")

    summary = {
        "total": total,
        "passed": passed,
        "auto_resolve_precision": auto_precision,
        "wrongly_auto_resolved": len(false_auto),
        "held_back_correctly": hold_recall,
        "deflection_rate": len(deflected) / total,
        "auto_resolve_rate": len(auto) / total,
        "routes": dict(routes),
        "by_type": {k: {"passed": sum(g.passed for g in v), "total": len(v)} for k, v in by_type.items()},
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "calls": calls,
            "per_ticket_prompt": prompt_tokens / total,
            "per_ticket_completion": completion_tokens / total,
            "calls_per_ticket": calls / total,
        },
        "latency_ms": {"p50": p50, "p95": p95},
        "thresholds": {
            "high": CONFIG.confidence.high,
            "medium": CONFIG.confidence.medium,
            "retrieval_floor": CONFIG.confidence.retrieval_floor,
        },
        "models": {
            "answerer": CONFIG.models.answerer,
            "verifier": CONFIG.models.verifier,
            "embedder": CONFIG.models.embedder,
        },
        "cases": [
            {
                "id": r.case_id, "type": r.type, "expected": r.expected, "actual": r.actual,
                "band": r.band, "score": round(r.score, 4), "passed": r.passed,
                "failures": r.failures,
            }
            for r in results
        ],
    }
    print()
    return summary


def sweep(results: list[CaseResult]) -> None:
    """Re-band the recorded scores at different thresholds.

    Hard gates are already reflected in the recorded route, so a gated case stays escalated at every
    threshold — which is correct, and is why the curve flattens rather than reaching 100% deflection.
    """
    print(f"\n{BOLD}OPERATING POINT — moving the auto-resolve threshold{RESET}\n")
    print(f"  {'high':<8} {'auto rate':>10} {'precision':>11} {'wrong auto':>11}")
    print("  " + "─" * 42)
    for threshold in [0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85]:
        auto, wrong = 0, 0
        for r in results:
            # A case that was escalated by a hard gate cannot be promoted by lowering a threshold.
            gated = r.actual == "escalate" and r.score >= CONFIG.confidence.high
            would_auto = r.score >= threshold and not gated and r.actual != "escalate"
            if r.actual == "escalate" and r.score >= threshold and not gated:
                would_auto = False
            if would_auto:
                auto += 1
                if r.expected in ("escalate", "not_auto") or not r.passed:
                    wrong += 1
        precision = (auto - wrong) / auto if auto else 1.0
        colour = GREEN if wrong == 0 else RED
        marker = f"  {DIM}← shipped{RESET}" if abs(threshold - CONFIG.confidence.high) < 1e-9 else ""
        print(
            f"  {threshold:<8.2f} {auto/len(results):>9.1%} {precision:>10.1%} "
            f"{colour}{wrong:>11}{RESET}{marker}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Bypass the response cache")
    parser.add_argument("--sweep", action="store_true", help="Print the threshold sweep")
    parser.add_argument("--only", default="", help="Run only case ids containing this substring")
    parser.add_argument("--out", type=Path, default=ROOT / "evals" / "results" / "latest.json")
    args = parser.parse_args()

    load_dotenv()
    if args.fresh:
        import os

        os.environ["DEFLECTOR_NO_CACHE"] = "1"

    cases = yaml.safe_load((ROOT / "evals" / "golden.yaml").read_text(encoding="utf-8"))["cases"]
    if args.only:
        cases = [c for c in cases if args.only in c["id"]]

    print(f"{DIM}loading corpus and building index…{RESET}")
    deflector = Deflector(provider=Provider())
    print(
        f"{DIM}{len(deflector.retriever.chunks)} chunks · dense="
        f"{'on' if deflector.retriever.dense_available else 'off'} · "
        f"answerer={CONFIG.models.answerer} · verifier={CONFIG.models.verifier}{RESET}"
    )

    started = time.perf_counter()
    results: list[CaseResult] = []
    for i, case in enumerate(cases, start=1):
        print(f"{DIM}  [{i}/{len(cases)}] {case['id']}…{RESET}", flush=True)
        results.append(evaluate_case(deflector, case))
    wall = time.perf_counter() - started

    summary = report(results, wall)
    if args.sweep:
        sweep(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{DIM}wrote {args.out.relative_to(ROOT)}{RESET}\n")

    return 0 if summary["wrongly_auto_resolved"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
