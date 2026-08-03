"""Layout-aware ingestion for markdown, HTML and PDF — including scanned PDFs.

A support knowledge base is not a folder of clean markdown. It is a product manual exported to PDF,
a help-centre article with nested tables, and a decade-old runbook someone scanned. The parsing layer
is where most RAG systems quietly lose their accuracy, long before the model is involved, and the
losses are invisible: nothing errors, the answers just get subtly wrong.

Four failure modes are handled explicitly.

**1. Tables get destroyed by naive extraction.**
`page.extract_text()` on a PDF flattens a table into a stream of cells with no row structure:
"Growth 1,200 2,000 for 60 seconds 100 Enterprise Negotiated". Ask "what is the Growth burst
ceiling" against that and the model will confidently return a number from the wrong row — and it
will cite the correct document while doing it, which makes the error very hard to catch.

So tables are located first, extracted as structure, and their bounding boxes are then *excluded*
from the prose pass, so the same content never appears twice in two different shapes.

**2. Tables retrieve terribly when embedded whole.**
A markdown table is mostly pipes and digits. Its embedding carries almost no semantic signal, so the
dense arm cannot find it, and the lexical arm only finds it if the user happened to type a literal
cell value. Two things fix this: a generated natural-language summary of what the table contains,
and row-level chunks.

**3. The answer is usually one row, not the whole table.**
For a lookup table — plan limits, error codes — the useful unit of retrieval is a single row with its
headers attached: "Code: region_mismatch | HTTP: 403 | Retryable: No | Operator action: Account
pinned to another region". Emitting both the whole table (for "compare the plans") and every row
(for "what does region_mismatch mean") is what makes table QA actually work. This is the single
highest-impact decision in this file.

**4. Tables span pages, and scanned pages have no text at all.**
Cross-page tables are stitched back together by matching column geometry and detecting a repeated
header row. Pages with no text layer are detected by character density and routed to a vision model,
which reads structure better than classical OCR does — an OCR engine returns a scanned table as
unaligned text, whereas a VLM can be asked for the rows.
"""

from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import CACHE_DIR, CONFIG

# Below this many characters of extractable text, a PDF page is treated as scanned.
SCANNED_CHAR_THRESHOLD = 90
# A table starting this close to the top of a page is a candidate continuation of the previous page.
CONTINUATION_TOP_MARGIN = 130


@dataclass
class Block:
    """One semantic unit of a document, before chunking."""

    kind: str                       # "prose" | "table" | "table_row" | "figure"
    section: str
    text: str                       # what gets indexed and searched
    display: str = ""               # what gets shown to the model; defaults to text
    page: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.display:
            self.display = self.text


@dataclass
class ParsedDoc:
    doc_id: str
    title: str
    category: str
    source_path: str
    source_kind: str                # "markdown" | "html" | "pdf" | "pdf_scanned"
    blocks: list[Block]
    last_reviewed: str = ""
    stats: dict[str, Any] = field(default_factory=dict)


# =======================================================================================
# Table helpers
# =======================================================================================

def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_table(rows: list[list[Any]]) -> list[list[str]]:
    """Trim, collapse whitespace, and drop rows that are entirely empty."""
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [_clean_cell(c) for c in row]
        if any(cells):
            cleaned.append(cells)
    if not cleaned:
        return []
    width = max(len(r) for r in cleaned)
    return [r + [""] * (width - len(r)) for r in cleaned]


def table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header, *body = rows
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(out)


def summarize_table(rows: list[list[str]], caption: str, section: str) -> str:
    """A deterministic natural-language description of a table, for the dense retrieval arm.

    An LLM-written summary would read better, but it costs a call per table on every re-index and
    introduces a hallucination surface in the *index itself* — a wrong summary silently mis-routes
    every future query. The deterministic version names the columns and enumerates the key column,
    which is what a user's question actually resembles: "what is the rate limit for Growth" contains
    the column concept and the row key, and both are now present in the indexed text.
    """
    if not rows:
        return ""
    header = [h for h in rows[0] if h]
    keys = [r[0] for r in rows[1:] if r and r[0]][:24]
    parts = []
    if caption:
        parts.append(caption.rstrip("."))
    if section:
        parts.append(f"from section {section}")
    text = "Table: " + (". ".join(parts) if parts else section or "untitled")
    if header:
        text += f". Columns: {', '.join(header)}"
    if keys:
        text += f". Rows: {', '.join(keys)}"
    return text + "."


def row_blocks(
    rows: list[list[str]], section: str, caption: str, page: int | None, table_index: int
) -> list[Block]:
    """One retrievable block per table row, with the column headers attached to every value.

    Without the headers a row is a meaningless tuple of values. With them, each row becomes a
    self-contained fact that survives being retrieved in isolation — which is the entire point,
    since it will be retrieved in isolation.
    """
    if len(rows) < 2:
        return []
    header = rows[0]
    blocks: list[Block] = []
    for i, row in enumerate(rows[1:]):
        pairs = [
            f"{h}: {v}" for h, v in zip(header, row) if v and h
        ] or [v for v in row if v]
        if not pairs:
            continue
        label = row[0] or f"row {i + 1}"
        blocks.append(
            Block(
                kind="table_row",
                section=section,
                text=f"{caption or section} — " + " | ".join(pairs),
                display=" | ".join(pairs),
                page=page,
                meta={"table_index": table_index, "row_key": label, "row_number": i + 1},
            )
        )
    return blocks


def looks_like_continuation(
    prev_rows: list[list[str]], rows: list[list[str]], bbox_top: float
) -> bool:
    """Is this table the tail of the one that ended on the previous page?"""
    if not prev_rows or not rows:
        return False
    if bbox_top > CONTINUATION_TOP_MARGIN:
        return False
    if len(prev_rows[0]) != len(rows[0]):
        return False
    # Either the header repeats verbatim, or the first row is clearly data under the same schema.
    return [c.lower() for c in prev_rows[0]] == [c.lower() for c in rows[0]] or all(
        h for h in prev_rows[0]
    )


# =======================================================================================
# Vision: figure captioning and scanned-page reading
# =======================================================================================

VISION_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"

FIGURE_PROMPT = (
    "You are indexing a technical figure from a software product manual so it can be found by "
    "search. Write a dense factual description. Name every box, label, arrow target, axis, legend "
    "entry, number and error code you can read, verbatim. State what the figure shows overall in "
    "one sentence first. Do not speculate about anything not visible. No preamble."
)

SCANNED_PAGE_PROMPT = (
    "Transcribe this scanned document page completely and literally. Preserve headings. Render any "
    "table as a markdown table with its header row. Do not summarise, do not omit rows, do not add "
    "commentary. Output only the transcription."
)


def _vision_cache_path(image_bytes: bytes, prompt: str) -> Path:
    import hashlib

    digest = hashlib.sha256(image_bytes + prompt.encode()).hexdigest()[:32]
    directory = CACHE_DIR / "vision"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{digest}.json"


def describe_image(image_bytes: bytes, prompt: str = FIGURE_PROMPT, max_tokens: int = 500) -> str:
    """Caption an image with a vision model, cached to disk.

    Cached aggressively and committed to the repo: this runs at ingest time, so a reviewer never
    pays for it and the index is reproducible. If the model is unavailable the caller falls back to
    the document's own alt text and caption, which is worse but never blocks ingestion.
    """
    path = _vision_cache_path(image_bytes, prompt)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        except (json.JSONDecodeError, KeyError):
            pass

    if CONFIG.offline or not CONFIG.api_key:
        return ""

    try:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.thumbnail((1100, 1100))
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        payload = buffer.getvalue()
    except Exception:
        payload = image_bytes

    b64 = base64.b64encode(payload).decode()

    import httpx

    try:
        response = httpx.post(
            f"{CONFIG.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {CONFIG.api_key}"},
            json={
                "model": VISION_MODEL,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    }
                ],
            },
            timeout=CONFIG.request_timeout,
        )
        data = response.json()
        if "error" in data:
            return ""
        text = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        return ""

    if text:
        path.write_text(json.dumps({"model": VISION_MODEL, "text": text}), encoding="utf-8")
    return text


def render_pdf_page(pdf_path: Path, page_number: int, dpi: int = 150) -> bytes:
    """Rasterise one PDF page via poppler. Used for the scanned path."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            [
                "pdftoppm", "-png", "-r", str(dpi),
                "-f", str(page_number), "-l", str(page_number),
                str(pdf_path), str(prefix),
            ],
            check=True,
            capture_output=True,
        )
        files = sorted(Path(tmp).glob("page*.png"))
        return files[0].read_bytes() if files else b""


# =======================================================================================
# Markdown
# =======================================================================================

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, raw[match.end():]


_MD_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def parse_markdown(path: Path) -> ParsedDoc:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    doc_id = meta.get("doc_id", path.stem)
    blocks: list[Block] = []
    section = "Overview"
    buffer: list[str] = []
    table_index = 0

    def flush_prose() -> None:
        text = "\n".join(buffer).strip()
        if text:
            blocks.append(Block(kind="prose", section=section, text=text))
        buffer.clear()

    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            flush_prose()
            section = line[3:].strip()
            i += 1
            continue
        if line.startswith("# "):
            i += 1
            continue

        # A markdown table: consume the whole run of pipe-lines as one unit.
        if _MD_TABLE_LINE.match(line) and i + 1 < len(lines) and _MD_TABLE_SEP.match(lines[i + 1]):
            flush_prose()
            raw_rows: list[list[str]] = []
            while i < len(lines) and _MD_TABLE_LINE.match(lines[i]):
                if not _MD_TABLE_SEP.match(lines[i]):
                    raw_rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            rows = normalize_table(raw_rows)
            if rows:
                table_index += 1
                markdown = table_to_markdown(rows)
                blocks.append(
                    Block(
                        kind="table",
                        section=section,
                        text=summarize_table(rows, "", section) + "\n" + markdown,
                        display=markdown,
                        meta={"table_index": table_index, "n_rows": len(rows) - 1},
                    )
                )
                blocks.extend(row_blocks(rows, section, "", None, table_index))
            continue

        buffer.append(line)
        i += 1

    flush_prose()

    return ParsedDoc(
        doc_id=doc_id,
        title=meta.get("title", path.stem.replace("-", " ").title()),
        category=meta.get("category", "general"),
        source_path=str(path.name),
        source_kind="markdown",
        blocks=blocks,
        last_reviewed=meta.get("last_reviewed", ""),
        stats={"tables": table_index},
    )


# =======================================================================================
# HTML
# =======================================================================================

def parse_html(path: Path, caption_figures: bool = True) -> ParsedDoc:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    def meta_value(name: str, default: str = "") -> str:
        tag = soup.find("meta", attrs={"name": name})
        return tag.get("content", default) if tag else default

    title = (soup.title.string if soup.title else path.stem).split("·")[0].strip()
    blocks: list[Block] = []
    section = "Overview"
    table_index = 0

    for nav in soup.select("nav, script, style"):
        nav.decompose()

    article = soup.find("article") or soup.body or soup

    for element in article.find_all(["h1", "h2", "h3", "p", "table", "figure", "dl"], recursive=True):
        # Skip nodes nested inside a figure we will handle as a unit.
        if element.find_parent("figure") and element.name != "figure":
            continue

        if element.name in ("h1", "h2", "h3"):
            section = element.get_text(" ", strip=True)

        elif element.name == "p":
            text = element.get_text(" ", strip=True)
            if text:
                blocks.append(Block(kind="prose", section=section, text=text))

        elif element.name == "dl":
            pairs = []
            for dt in element.find_all("dt"):
                dd = dt.find_next_sibling("dd")
                pairs.append(
                    f"Q: {dt.get_text(' ', strip=True)}\nA: "
                    f"{dd.get_text(' ', strip=True) if dd else ''}"
                )
            if pairs:
                blocks.append(Block(kind="prose", section=section, text="\n\n".join(pairs)))

        elif element.name == "table":
            caption_tag = element.find("caption")
            caption = caption_tag.get_text(" ", strip=True) if caption_tag else ""
            raw_rows = [
                [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                for tr in element.find_all("tr")
            ]
            rows = normalize_table(raw_rows)
            if rows:
                table_index += 1
                markdown = table_to_markdown(rows)
                blocks.append(
                    Block(
                        kind="table",
                        section=section,
                        text=summarize_table(rows, caption, section) + "\n" + markdown,
                        display=(f"{caption}\n" if caption else "") + markdown,
                        meta={"table_index": table_index, "caption": caption, "n_rows": len(rows) - 1},
                    )
                )
                blocks.extend(row_blocks(rows, section, caption, None, table_index))

        elif element.name == "figure":
            img = element.find("img")
            figcaption = element.find("figcaption")
            alt = img.get("alt", "") if img else ""
            caption = figcaption.get_text(" ", strip=True) if figcaption else ""

            described = ""
            if caption_figures and img and img.get("src"):
                image_path = (path.parent / img["src"]).resolve()
                if image_path.exists():
                    described = describe_image(image_path.read_bytes())

            text = "\n".join(x for x in [caption, alt, described] if x)
            if text:
                blocks.append(
                    Block(
                        kind="figure",
                        section=section,
                        text=f"Figure: {text}",
                        display=f"[Figure] {caption or alt}\n{described or alt}",
                        meta={
                            "alt": alt,
                            "caption": caption,
                            "src": img.get("src") if img else "",
                            "vision_described": bool(described),
                        },
                    )
                )

    return ParsedDoc(
        doc_id=meta_value("article-id", path.stem),
        title=title,
        category=meta_value("category", "general"),
        source_path=str(path.name),
        source_kind="html",
        blocks=blocks,
        last_reviewed=meta_value("last-reviewed", ""),
        stats={"tables": table_index},
    )


# =======================================================================================
# PDF
# =======================================================================================

def _bbox_contains(obj: dict, bbox: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = bbox
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom


def _pdf_headings_and_prose(
    page, table_bboxes: list[tuple], body_size: float
) -> list[tuple[str, str, float]]:
    """Return (kind, text, top) triples where kind is 'heading' or 'prose'.

    The vertical position is carried through because tables are emitted after the prose pass, and a
    table must be attributed to the heading physically above it — not to whichever heading happened
    to be last when the page finished parsing.
    """
    words = [
        w for w in page.extract_words(extra_attrs=["size"])
        if not any(_bbox_contains(w, b) for b in table_bboxes)
    ]
    if not words:
        return []

    lines: dict[float, list[dict]] = {}
    for w in words:
        key = round(w["top"] / 3) * 3
        lines.setdefault(key, []).append(w)

    out: list[tuple[str, str, float]] = []
    for key in sorted(lines):
        row = sorted(lines[key], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in row).strip()
        if not text:
            continue
        size = max(float(w.get("size", body_size)) for w in row)
        top = min(float(w["top"]) for w in row)
        out.append(("heading" if size >= body_size + 1.4 else "prose", text, top))
    return out


def parse_pdf(path: Path, caption_figures: bool = True) -> ParsedDoc:
    import pdfplumber

    blocks: list[Block] = []
    table_index = 0
    scanned_pages = 0
    born_digital_pages = 0
    figures = 0
    section = "Overview"

    pending: dict | None = None   # a table awaiting a possible continuation on the next page

    def flush_pending() -> None:
        nonlocal pending, table_index
        if pending is None:
            return
        rows = normalize_table(pending["rows"])
        if rows:
            table_index += 1
            markdown = table_to_markdown(rows)
            blocks.append(
                Block(
                    kind="table",
                    section=pending["section"],
                    page=pending["page"],
                    text=summarize_table(rows, "", pending["section"]) + "\n" + markdown,
                    display=markdown,
                    meta={
                        "table_index": table_index,
                        "n_rows": len(rows) - 1,
                        "spans_pages": pending["pages"],
                    },
                )
            )
            blocks.extend(
                row_blocks(rows, pending["section"], "", pending["page"], table_index)
            )
        pending = None

    with pdfplumber.open(path) as pdf:
        # The body-text size must be measured from prose only. Measuring it across the whole page
        # lets a dense 8.5pt table dominate the distribution, which drags the baseline down until
        # every 10pt body line clears the heading threshold and the prose buffer never fills.
        from collections import Counter

        size_counts: Counter[float] = Counter()
        for pg in pdf.pages[:4]:
            table_boxes = [t.bbox for t in pg.find_tables()]
            for w in pg.extract_words(extra_attrs=["size"]):
                if any(_bbox_contains(w, b) for b in table_boxes):
                    continue
                if w.get("size"):
                    size_counts[round(float(w["size"]), 1)] += len(w["text"])
        body_size = size_counts.most_common(1)[0][0] if size_counts else 10.0

        for page_number, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""

            # ---- scanned-page path ----------------------------------------------------
            if len(raw_text.strip()) < SCANNED_CHAR_THRESHOLD:
                scanned_pages += 1
                flush_pending()
                transcript = ""
                image_bytes = render_pdf_page(path, page_number)
                if image_bytes:
                    transcript = describe_image(
                        image_bytes, prompt=SCANNED_PAGE_PROMPT, max_tokens=1400
                    )
                if transcript:
                    blocks.append(
                        Block(
                            kind="prose",
                            section=f"Page {page_number} (scanned)",
                            page=page_number,
                            text=transcript,
                            meta={"scanned": True, "vision_transcribed": True},
                        )
                    )
                continue

            born_digital_pages += 1
            found = page.find_tables()
            bboxes = [t.bbox for t in found]

            # ---- prose, with table regions excluded ------------------------------------
            buffer: list[str] = []

            def flush_prose(sec: str) -> None:
                text = "\n".join(buffer).strip()
                if len(text) > 2:
                    blocks.append(
                        Block(kind="prose", section=sec, page=page_number, text=text)
                    )
                buffer.clear()

            headings_by_top: list[tuple[float, str]] = []
            for kind, text, top in _pdf_headings_and_prose(page, bboxes, body_size):
                if kind == "heading":
                    flush_prose(section)
                    section = text
                    headings_by_top.append((top, text))
                else:
                    buffer.append(text)
            flush_prose(section)

            def section_above(y: float, fallback: str) -> str:
                """The heading immediately above this vertical position on the page."""
                candidates = [name for top, name in headings_by_top if top < y]
                return candidates[-1] if candidates else fallback

            # ---- tables, with cross-page stitching -------------------------------------
            for table in sorted(found, key=lambda t: t.bbox[1]):
                rows = normalize_table(table.extract())
                if not rows:
                    continue
                if pending is not None and looks_like_continuation(
                    pending["rows"], rows, table.bbox[1]
                ):
                    header_repeated = [c.lower() for c in pending["rows"][0]] == [
                        c.lower() for c in rows[0]
                    ]
                    pending["rows"].extend(rows[1:] if header_repeated else rows)
                    pending["pages"].append(page_number)
                    continue
                flush_pending()
                pending = {
                    "rows": rows,
                    "section": section_above(table.bbox[1], section),
                    "page": page_number,
                    "pages": [page_number],
                }

            # ---- embedded figures -------------------------------------------------------
            if caption_figures and page.images:
                for image in page.images:
                    try:
                        crop = page.crop(
                            (
                                max(0, image["x0"] - 2),
                                max(0, image["top"] - 2),
                                min(page.width, image["x1"] + 2),
                                min(page.height, image["bottom"] + 2),
                            )
                        )
                        buf = io.BytesIO()
                        crop.to_image(resolution=150).save(buf, format="PNG")
                        described = describe_image(buf.getvalue())
                    except Exception:
                        described = ""
                    if described:
                        figures += 1
                        blocks.append(
                            Block(
                                kind="figure",
                                section=section,
                                page=page_number,
                                text=f"Figure on page {page_number}: {described}",
                                display=f"[Figure, page {page_number}]\n{described}",
                                meta={"vision_described": True},
                            )
                        )

        flush_pending()

    title = path.stem.replace("-", " ").title()
    try:
        import pdfplumber as _pp

        with _pp.open(path) as pdf:
            if pdf.metadata and pdf.metadata.get("Title"):
                title = pdf.metadata["Title"]
    except Exception:
        pass

    return ParsedDoc(
        doc_id=path.stem,
        title=title,
        category="operations",
        source_path=str(path.name),
        source_kind="pdf_scanned" if scanned_pages > born_digital_pages else "pdf",
        blocks=blocks,
        stats={
            "tables": table_index,
            "figures": figures,
            "pages_born_digital": born_digital_pages,
            "pages_scanned": scanned_pages,
        },
    )


# =======================================================================================
# Entry point
# =======================================================================================

def parse_any(path: Path, caption_figures: bool = True) -> ParsedDoc | None:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return parse_markdown(path)
    if suffix in (".html", ".htm"):
        return parse_html(path, caption_figures=caption_figures)
    if suffix == ".pdf":
        return parse_pdf(path, caption_figures=caption_figures)
    return None


def ingest_directory(
    directories: Iterable[Path], caption_figures: bool = True
) -> list[ParsedDoc]:
    docs: list[ParsedDoc] = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_dir() or path.name.startswith("."):
                continue
            parsed = parse_any(path, caption_figures=caption_figures)
            if parsed and parsed.blocks:
                docs.append(parsed)
    return docs
