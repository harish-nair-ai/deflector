"""Turning parsed blocks into retrievable chunks.

Parsing (`ingest.py`) decides *what the document says*. This module decides *what the unit of
retrieval is*, and the two decisions are deliberately separate — swapping the chunking policy should
not require touching a PDF parser.

The policy:

* **Prose** is packed to a target size along paragraph boundaries, never mid-sentence, with a small
  overlap so a fact split across a boundary survives in one of the two halves.
* **Tables are never split.** A table is one chunk regardless of size, because half a table is worse
  than no table — it retrieves confidently and answers wrongly.
* **Table rows are their own chunks**, carrying their headers, because the answer to a lookup
  question is one row.
* **Figures** are chunks whose text is the vision model's description plus the document's own
  caption.

Every chunk carries a **context header** — document title, then section path — prepended before
embedding. This is the cheap form of contextual retrieval: the expensive form asks an LLM to write a
bespoke situating sentence per chunk, which helps measurably but costs a full pass over the corpus on
every re-index and puts a hallucination surface inside the index itself. The deterministic header
captures much of the benefit for zero tokens. The trade-off is stated in the README rather than
hidden.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import CACHE_DIR, CONFIG, CORPUS_DIR, ROOT
from .ingest import Block, ParsedDoc, ingest_directory

RAW_DIR = ROOT / "corpus_raw"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    category: str
    section: str
    text: str
    kind: str = "prose"
    page: int | None = None
    last_reviewed: str = ""
    source_kind: str = "markdown"

    @property
    def contextualized(self) -> str:
        """What actually gets embedded and lexically searched."""
        locator = f"{self.doc_title} > {self.section}"
        if self.page:
            locator += f" > page {self.page}"
        return f"[{locator}]\n{self.text}"

    def citation_label(self) -> str:
        label = f"{self.doc_title} — {self.section}"
        if self.page:
            label += f" (p.{self.page})"
        return label

    def to_dict(self) -> dict:
        return asdict(self)


def _slug(value: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:limit] or "section"


def _pack_prose(text: str, target: int, overlap: int) -> list[str]:
    """Pack paragraphs up to a target word count, carrying a short tail forward as overlap."""
    words = len(text.split())
    if words <= target * 1.35:
        return [text]

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 1:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    count = 0
    for para in paragraphs:
        para_words = len(para.split())
        if count + para_words > target and current:
            parts.append("\n\n".join(current))
            tail: list[str] = []
            tail_words = 0
            for prev in reversed(current):
                tail.insert(0, prev)
                tail_words += len(prev.split())
                if tail_words >= overlap:
                    break
            current = list(tail)
            count = tail_words
        current.append(para)
        count += para_words
    if current:
        parts.append("\n\n".join(current))
    return parts or [text]


def chunks_from_doc(doc: ParsedDoc) -> list[Chunk]:
    cfg = CONFIG.retrieval
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}

    def add(block: Block, text: str) -> None:
        base = f"{doc.doc_id}::{_slug(block.section)}"
        if block.kind == "table_row":
            base += f"::row{block.meta.get('row_number', 0)}"
        elif block.kind == "table":
            base += f"::table{block.meta.get('table_index', 0)}"
        elif block.kind == "figure":
            base += "::figure"
        count = seen.get(base, 0)
        seen[base] = count + 1
        chunk_id = base if count == 0 else f"{base}#{count}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                doc_title=doc.title,
                category=doc.category,
                section=block.section,
                text=text,
                kind=block.kind,
                page=block.page,
                last_reviewed=doc.last_reviewed,
                source_kind=doc.source_kind,
            )
        )

    for block in doc.blocks:
        if block.kind == "prose":
            for part in _pack_prose(
                block.text, cfg.chunk_target_words, cfg.chunk_overlap_words
            ):
                add(block, part)
        else:
            # Tables, table rows and figures are atomic by policy.
            add(block, block.text)

    return chunks


def _source_fingerprint(directories: list[Path]) -> str:
    """Hash of every source file's path, size and mtime, plus the chunking policy.

    Ingestion is a build step: parsing four PDFs and captioning their figures takes tens of seconds
    and, for figures, real model calls. Paying that on every process start is wrong — it turns a
    deploy-time cost into a request-path cost. The cache is keyed on the sources *and* the chunking
    parameters, so editing a document or changing the target chunk size invalidates it, but
    restarting the service does not.

    The vision cache is hashed in too, and that is not cosmetic. Figure captions and scanned-page
    transcriptions are *model output* that becomes chunk text, so they are an input to chunking just
    as much as the PDFs are. They also arrive late: a captioning call can fail on one run (rate
    limit, timeout) and succeed on the next, leaving the sources byte-identical while the text they
    produce changes. Keying on sources alone means that second run silently keeps serving chunks
    built from the failed attempt. Caught exactly that way — a figure caption landed on a retry and
    the index stayed one chunk short.
    """
    import hashlib

    cfg = CONFIG.retrieval
    h = hashlib.sha256()
    h.update(f"{cfg.chunk_target_words}:{cfg.chunk_overlap_words}:v3".encode())
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                stat = path.stat()
                h.update(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}".encode())

    vision_dir = CACHE_DIR / "vision"
    if vision_dir.exists():
        # Content, not mtime: these files are committed fixtures, so a fresh clone gets whatever
        # mtime the checkout assigns. Hashing bytes keeps the fingerprint stable across machines.
        for path in sorted(vision_dir.glob("*.json")):
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


CHUNK_CACHE = CACHE_DIR / "chunks.json"


def load_chunks(
    corpus_dir: Path | None = None,
    raw_dir: Path | None = None,
    caption_figures: bool = True,
    use_cache: bool = True,
) -> list[Chunk]:
    directories = [corpus_dir or CORPUS_DIR, raw_dir or RAW_DIR]
    fingerprint = _source_fingerprint(directories)

    if use_cache and CHUNK_CACHE.exists():
        try:
            cached = json.loads(CHUNK_CACHE.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                return [Chunk(**c) for c in cached["chunks"]]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    docs = ingest_directory(directories, caption_figures=caption_figures)
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunks_from_doc(doc))

    if use_cache:
        CHUNK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CHUNK_CACHE.write_text(
            json.dumps(
                {"fingerprint": fingerprint, "chunks": [c.to_dict() for c in chunks]},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
    return chunks


def load_documents(
    corpus_dir: Path | None = None, raw_dir: Path | None = None, caption_figures: bool = True
) -> list[ParsedDoc]:
    return ingest_directory(
        [corpus_dir or CORPUS_DIR, raw_dir or RAW_DIR], caption_figures=caption_figures
    )
