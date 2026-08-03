"""Command line entry point: `deflector ask "..."`, `deflector serve`, `deflector index`."""

from __future__ import annotations

import argparse
import json
import sys

from .config import CONFIG
from .pipeline import Deflector
from .providers import load_dotenv

BAND_COLOUR = {"High": "\033[92m", "Medium": "\033[93m", "Low": "\033[91m"}
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"


def _render(result) -> None:
    band = result.band.value
    colour = BAND_COLOUR.get(band, "")
    print()
    print(f"{BOLD}route{RESET}       {colour}{result.route.value.upper()}{RESET}")
    print(f"{BOLD}confidence{RESET}  {colour}{band}{RESET}  ({result.score:.3f})")
    print()

    if result.route.value == "auto_resolve":
        print(f"{BOLD}Reply sent to customer:{RESET}")
    elif result.route.value == "agent_assist":
        print(f"{BOLD}Draft for agent review — not sent:{RESET}")
    else:
        print(f"{BOLD}Escalated — nothing sent:{RESET}")
    print(result.answer or "(no answer produced)")
    print()

    if result.citations:
        print(f"{BOLD}Sources cited{RESET}")
        for c in result.citations:
            print(f"  {c['marker']}  {c['title']} › {c['section']}  {DIM}({c['doc_id']}){RESET}")
        print()

    sig = result.decision.signals
    print(f"{BOLD}Signals{RESET}")
    for name, weight in (
        ("verifier", CONFIG.confidence.w_verifier),
        ("retrieval", CONFIG.confidence.w_retrieval),
        ("citation", CONFIG.confidence.w_citation),
        ("self_report", CONFIG.confidence.w_self_report),
    ):
        value = sig.get(name, 0.0)
        bar = "█" * int(round(value * 20))
        print(f"  {name:<12} {value:>5.2f}  w={weight:<4} {DIM}{bar}{RESET}")
    print()

    if result.decision.gates:
        print(f"{BOLD}Hard gates tripped{RESET}")
        for gate in result.decision.gates:
            print(f"  \033[91m✗\033[0m {gate}")
        print()

    screening = result.screening
    if screening.get("detections") or screening.get("policy_intents") or screening.get("injection_hits"):
        print(f"{BOLD}Screening{RESET}")
        for d in screening.get("detections", []):
            mark = "\033[91mESCALATE\033[0m" if d["tier"] == "escalate" else f"{DIM}redact{RESET}"
            print(f"  {mark:<20} {d['kind']} ×{d['count']}  {DIM}{d['sample']}{RESET}")
        for intent in screening.get("policy_intents", []):
            print(f"  \033[91mESCALATE\033[0m            policy intent: {intent}")
        if screening.get("injection_hits"):
            print(f"  \033[91mESCALATE\033[0m            prompt injection suspected")
        print()

    if result.decision.reasons:
        print(f"{DIM}Notes{RESET}")
        for reason in result.decision.reasons:
            print(f"  {DIM}· {reason}{RESET}")
        print()

    meta, usage = result.meta, result.usage.to_dict()
    print(
        f"{DIM}{meta['elapsed_ms']:.0f} ms · {usage['calls']} model call(s)"
        f"{' (cached)' if usage['cache_hits'] else ''}"
        f" · {usage['prompt_tokens']}+{usage['completion_tokens']} tokens"
        f" · dense={'on' if meta['dense_retrieval'] else 'off (BM25 only)'}"
        f" · {meta['prompt_version']}{RESET}"
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="deflector", description="Support deflection service")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Run one ticket through the pipeline")
    ask.add_argument("question", help="The customer's message")
    ask.add_argument("--subject", default="", help="Ticket subject")
    ask.add_argument("--json", action="store_true", help="Emit the full decision record as JSON")

    sub.add_parser("index", help="Build or refresh the embedding index")

    ingest = sub.add_parser("ingest", help="Parse the corpus and report what the parser found")
    ingest.add_argument(
        "--dir", type=str, default=None, help="Directory to parse instead of the product corpus"
    )
    ingest.add_argument(
        "--no-figures", action="store_true", help="Skip vision captioning (fast structural pass)"
    )

    serve = sub.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "ingest":
        from collections import Counter
        from pathlib import Path

        from .corpus import RAW_DIR, chunks_from_doc
        from .config import CORPUS_DIR
        from .ingest import ingest_directory

        dirs = [Path(args.dir)] if args.dir else [CORPUS_DIR, RAW_DIR]
        docs = ingest_directory(dirs, caption_figures=not args.no_figures)

        header = f"{'document':<38} {'source':<12} {'pg':>3} {'prose':>6} {'tbl':>4} {'rows':>5} {'fig':>4} {'chunks':>7}"
        print(f"\n{BOLD}{header}{RESET}")
        print("─" * len(header))
        totals = Counter()
        for doc in docs:
            kinds = Counter(b.kind for b in doc.blocks)
            n_chunks = len(chunks_from_doc(doc))
            pages = doc.stats.get("pages_born_digital", 0) + doc.stats.get("pages_scanned", 0)
            scanned = doc.stats.get("pages_scanned", 0)
            marker = f"{DIM}(scanned×{scanned}){RESET}" if scanned else ""
            print(
                f"{doc.doc_id[:37]:<38} {doc.source_kind:<12} {pages or '-':>3} "
                f"{kinds.get('prose', 0):>6} {kinds.get('table', 0):>4} "
                f"{kinds.get('table_row', 0):>5} {kinds.get('figure', 0):>4} {n_chunks:>7} {marker}"
            )
            totals.update(kinds)
            totals["chunks"] += n_chunks
            totals["docs"] += 1
            totals["scanned_pages"] += scanned

        print("─" * len(header))
        print(
            f"{BOLD}{totals['docs']} documents{RESET} · {totals['prose']} prose · "
            f"{totals['table']} tables · {totals['table_row']} table rows · "
            f"{totals['figure']} figures · {totals['scanned_pages']} scanned pages "
            f"→ {BOLD}{totals['chunks']} chunks{RESET}\n"
        )
        return 0

    if args.command == "index":
        deflector = Deflector()
        ok = deflector.retriever.build_index(force=True)
        print(
            f"indexed {len(deflector.retriever.chunks)} chunks from "
            f"{len({c.doc_id for c in deflector.retriever.chunks})} documents "
            f"({'dense + BM25' if ok else 'BM25 only — embeddings unavailable'})"
        )
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run("deflector.api:app", host=args.host, port=args.port, log_level="info")
        return 0

    deflector = Deflector()
    result = deflector.deflect(body=args.question, subject=args.subject)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _render(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
