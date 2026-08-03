"""Inject measured eval results into README.md between marker comments.

The README quotes numbers. Numbers typed by hand drift from the numbers the code produces, usually
within a day, and a stale metric in a README is worse than no metric because it looks authoritative.
So the results section is generated: run the eval, run this, and the document cannot disagree with
the harness.

Run: python tools/update_readme_results.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "evals" / "results" / "latest.json"

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
COST_START = "<!-- COST:START -->"
COST_END = "<!-- COST:END -->"

TYPE_LABELS = {
    "answerable": "Answerable from prose",
    "table_lookup": "Answerable only from a table",
    "figure_lookup": "Answerable only from a figure",
    "unanswerable": "Not in the corpus (must abstain)",
    "policy_escalation": "Correct answer, action needs a human",
    "sensitive": "Sensitive data present",
    "sensitive_benign": "Benign PII (must not over-escalate)",
    "injection": "Prompt injection",
    "ambiguous": "Ambiguous or partially covered",
}


def render_results(data: dict) -> str:
    lines: list[str] = []
    routes = data["routes"]
    total = data["total"]

    lines.append(
        f"Measured on the {total}-case golden set. "
        f"Answerer `{data['models']['answerer']}`, "
        f"independent verifier `{data['models']['verifier']}`.\n"
    )
    lines.append("| Metric | Value | What it means |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| **Auto-resolve precision** | **{data['auto_resolve_precision']:.1%}** | "
        "Of answers sent with no human involved, the share that were correct. **The number that "
        "decides whether this ships.** |"
    )
    lines.append(
        f"| **Wrongly auto-resolved** | **{data['wrongly_auto_resolved']}** | "
        "Tickets where a wrong answer reached the customer. |"
    )
    lines.append(
        f"| Held back correctly | {data['held_back_correctly']:.1%} | "
        "Of cases that must not be auto-answered, the share the gate actually held. |"
    )
    lines.append(
        f"| Deflection rate | {data['deflection_rate']:.1%} | "
        "Tickets resolved or drafted without a human owning them. |"
    )
    lines.append(
        f"| Auto-resolve rate | {data['auto_resolve_rate']:.1%} | Sent with no human at all. |"
    )
    lines.append(
        f"| Case pass rate | {data['passed']}/{total} | "
        "All assertions, including required facts and forbidden phrases. |"
    )
    lines.append("")
    lines.append(
        f"Routing mix: **{routes.get('auto_resolve', 0)} auto-resolve** · "
        f"**{routes.get('agent_assist', 0)} agent-assist** · "
        f"**{routes.get('escalate', 0)} escalate**.\n"
    )

    lines.append("| Case type | Passed |")
    lines.append("|---|---:|")
    for key, group in sorted(data["by_type"].items(), key=lambda kv: -kv[1]["total"]):
        label = TYPE_LABELS.get(key, key)
        lines.append(f"| {label} | {group['passed']}/{group['total']} |")
    lines.append("")

    latency = data["latency_ms"]
    tokens = data["tokens"]
    lines.append(
        f"Latency p50 **{latency['p50']/1000:.1f}s**, p95 **{latency['p95']/1000:.1f}s**. "
        f"**{tokens['calls_per_ticket']:.2f}** model calls per ticket "
        f"(under two because the verifier is skipped once a hard gate has already decided the "
        f"outcome)."
    )
    return "\n".join(lines)


def render_cost() -> str:
    output = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "cost_model.py"), "--markdown"],
        capture_output=True, text=True, check=True,
    )
    return output.stdout.strip()


def replace(text: str, start: str, end: str, body: str) -> str:
    a = text.index(start) + len(start)
    b = text.index(end)
    return text[:a] + "\n" + body + "\n" + text[b:]


def main() -> int:
    if not RESULTS.exists():
        print("no eval results yet — run `make eval` first")
        return 1
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    text = README.read_text(encoding="utf-8")
    text = replace(text, START, END, render_results(data))
    text = replace(text, COST_START, COST_END, render_cost())
    README.write_text(text, encoding="utf-8")
    print(f"README.md updated from {RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
