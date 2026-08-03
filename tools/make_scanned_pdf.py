"""Produce a genuinely scanned PDF — image-only, no text layer — from a born-digital one.

Rasterising and then re-wrapping is the honest way to build a scanned fixture: the output really has
no extractable text, so the ingester's scanned-page branch is exercised for real rather than being
simulated with a flag. Mild degradation is applied on purpose — a slight skew, grayscale conversion
and JPEG artefacts — because a clean 300 dpi render is not what arrives from a customer's office
scanner, and a pipeline tuned on clean renders falls over on the real thing.

Usage: python tools/make_scanned_pdf.py <source.pdf> <out.pdf> [--pages 1-3]
"""

from __future__ import annotations

import argparse
import io
import random
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def rasterize(src: Path, dpi: int, first: int, last: int) -> list[Image.Image]:
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "pg"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-f", str(first), "-l", str(last),
             str(src), str(prefix)],
            check=True, capture_output=True,
        )
        return [Image.open(p).convert("RGB").copy() for p in sorted(Path(tmp).glob("pg*.png"))]


def degrade(image: Image.Image, seed: int) -> Image.Image:
    """Make it look like it came off a real scanner rather than a renderer."""
    rng = random.Random(seed)

    # A page never sits perfectly square in the feeder.
    image = image.rotate(rng.uniform(-0.7, 0.7), resample=Image.BICUBIC, fillcolor=(255, 255, 255))

    # Most office scanners default to grayscale.
    image = image.convert("L").convert("RGB")

    # Slight paper tint, so the background is not pure white.
    tint = Image.new("RGB", image.size, (252, 251, 246))
    image = Image.blend(image, tint, 0.12)

    # JPEG compression artefacts, the signature of a scanned attachment.
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=62)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pages", default="1-3", help="page range, e.g. 1-3")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    first, _, last = args.pages.partition("-")
    pages = rasterize(args.source, args.dpi, int(first), int(last or first))
    if not pages:
        print("no pages rendered")
        return 1

    degraded = [degrade(p, seed=i) for i, p in enumerate(pages)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    degraded[0].save(
        args.output, "PDF", save_all=True, append_images=degraded[1:], resolution=args.dpi
    )
    print(f"wrote {args.output} — {len(degraded)} page(s), {args.output.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
