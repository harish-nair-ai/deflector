"""Fetch the real-world stress corpus.

These are third-party documents used to prove the parser handles layouts I did not author. They are
downloaded rather than committed — redistributing someone else's PDF inside a repo is not mine to do,
and the point is that the parser survives documents it has never seen, which a committed fixture
quietly undermines.

Run: python tools/fetch_stress_corpus.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "corpus_stress"

SOURCES = [
    (
        "attention-is-all-you-need.pdf",
        "https://arxiv.org/pdf/1706.03762",
        "Two-column academic layout, 8 tables, figures. Nothing like the product corpus.",
    ),
    (
        "irs-form-w9.pdf",
        "https://www.irs.gov/pub/irs-pdf/fw9.pdf",
        "A real government form: dense boxed layout, no clean prose flow.",
    ),
]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)

    for name, url, why in SOURCES:
        target = DEST / name
        if target.exists():
            print(f"  have    {name}")
            continue
        print(f"  fetch   {name}  — {why}")
        try:
            response = httpx.get(url, timeout=120, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"          failed: {exc}")
            continue
        target.write_bytes(response.content)
        print(f"          {target.stat().st_size:,} bytes")

    # The scanned fixture is derived locally, so the degradation is reproducible.
    scanned = DEST / "scanned-w9-form.pdf"
    source = DEST / "irs-form-w9.pdf"
    if not scanned.exists() and source.exists():
        print("  derive  scanned-w9-form.pdf — rasterised, deskewed, JPEG-degraded, no text layer")
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "make_scanned_pdf.py"),
             str(source), str(scanned), "--pages", "1-2"],
            check=True,
        )

    print("\n  now run:  make ingest-stress\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
