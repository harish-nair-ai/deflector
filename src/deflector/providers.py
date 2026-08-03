"""LLM and embedding access, with record/replay and graceful degradation.

Three things here are load-bearing beyond "call the API":

1. **A disk-backed response cache keyed by the exact request.** This makes the eval suite
   reproducible — a reviewer with no API key can run `make eval` and get byte-identical results to
   the ones in the README, because every call replays from `.cache/llm/`. It is also how you debug a
   regression in a non-deterministic system: you freeze the model's side of the conversation and
   change only your own code.

2. **A fallback chain.** Free-tier models rate-limit upstream without warning. A support system that
   drops a ticket because one provider returned 429 is not a support system. On failure we walk the
   chain, and only if every model fails do we return a null result — which the pipeline treats as an
   escalation, not as a crash.

3. **Real token accounting.** Every call records prompt and completion tokens, so the cost figure in
   the README is measured from actual traffic rather than estimated from a guess about prompt length.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import CONFIG, LLM_CACHE_DIR


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0
    embed_tokens: int = 0
    latency_ms: float = 0.0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls
        self.cache_hits += other.cache_hits
        self.embed_tokens += other.embed_tokens
        self.latency_ms += other.latency_ms

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embed_tokens": self.embed_tokens,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class LLMResult:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    ok: bool = True
    error: str = ""


class ProviderError(RuntimeError):
    pass


def _cache_key(kind: str, payload: dict) -> str:
    blob = json.dumps({"kind": kind, **payload}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class Provider:
    """OpenRouter-backed chat + embeddings."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or LLM_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = Usage()
        self._client: httpx.Client | None = None
        self.degraded: list[str] = []

    # -- cache -------------------------------------------------------------------------

    def _read_cache(self, key: str) -> dict | None:
        if not CONFIG.cache_enabled:
            return None
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def _write_cache(self, key: str, value: dict) -> None:
        if not CONFIG.cache_enabled:
            return
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps(value, indent=2), encoding="utf-8"
        )

    # -- transport ---------------------------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=CONFIG.request_timeout)
        return self._client

    def _post(self, path: str, body: dict) -> dict:
        key = CONFIG.api_key
        if not key:
            raise ProviderError("OPENROUTER_API_KEY is not set")
        response = self.client.post(
            f"{CONFIG.api_base}{path}",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/harish-nair-ai/deflector",
                "X-Title": "Deflector",
            },
            json=body,
        )
        if response.status_code >= 400:
            raise ProviderError(f"HTTP {response.status_code}: {response.text[:220]}")
        data = response.json()
        if "error" in data:
            raise ProviderError(str(data["error"])[:220])
        return data

    # -- chat --------------------------------------------------------------------------

    def chat(
        self,
        *,
        system: str,
        user: str,
        model: str,
        fallbacks: tuple[str, ...] = (),
        max_tokens: int = 800,
        temperature: float | None = None,
    ) -> LLMResult:
        temperature = CONFIG.models.temperature if temperature is None else temperature
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        cache_payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        key = _cache_key("chat", cache_payload)

        cached = self._read_cache(key)
        if cached is not None:
            usage = Usage(
                prompt_tokens=cached.get("prompt_tokens", 0),
                completion_tokens=cached.get("completion_tokens", 0),
                calls=1,
                cache_hits=1,
            )
            self.usage.add(usage)
            return LLMResult(
                text=cached["text"], model=cached.get("model", model), usage=usage, ok=True
            )

        if CONFIG.offline:
            return LLMResult(text="", model=model, ok=False, error="offline mode, no cache entry")

        last_error = ""
        for candidate in (model, *fallbacks):
            started = time.perf_counter()
            try:
                data = self._post(
                    "/chat/completions",
                    {
                        "model": candidate,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
            except (ProviderError, httpx.HTTPError) as exc:
                last_error = f"{candidate}: {exc}"
                self.degraded.append(last_error)
                continue

            elapsed = (time.perf_counter() - started) * 1000
            choice = data["choices"][0]["message"]
            text = (choice.get("content") or "").strip()

            # Some reasoning models put everything in `reasoning` and leave content empty.
            if not text:
                text = (choice.get("reasoning") or "").strip()
            if not text:
                last_error = f"{candidate}: empty completion"
                self.degraded.append(last_error)
                continue

            raw_usage = data.get("usage") or {}
            usage = Usage(
                prompt_tokens=raw_usage.get("prompt_tokens", 0),
                completion_tokens=raw_usage.get("completion_tokens", 0),
                calls=1,
                latency_ms=elapsed,
            )
            if candidate != model:
                self.degraded.append(f"fell back to {candidate}")

            self._write_cache(
                key,
                {
                    "text": text,
                    "model": candidate,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "latency_ms": round(elapsed, 1),
                },
            )
            self.usage.add(usage)
            return LLMResult(text=text, model=candidate, usage=usage, ok=True)

        return LLMResult(text="", model=model, ok=False, error=last_error or "all models failed")

    # -- embeddings --------------------------------------------------------------------

    # Batches larger than this are corpus index builds, which persist to .cache/index.json and must
    # not also be written to the response cache — that stores the same megabytes twice.
    # Query embeddings are single-item and *are* cached, because that is what lets the eval replay
    # without an API key.
    EMBED_CACHE_MAX_BATCH = 32

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Returns None when embeddings are unavailable — the caller degrades to BM25 only."""
        if not texts:
            return []

        cacheable = len(texts) <= self.EMBED_CACHE_MAX_BATCH
        key = _cache_key("embed", {"model": CONFIG.models.embedder, "texts": texts})
        cached = self._read_cache(key) if cacheable else None
        if cached is not None:
            self.usage.cache_hits += 1
            return cached["vectors"]

        if CONFIG.offline or not CONFIG.api_key:
            return None

        try:
            data = self._post(
                "/embeddings", {"model": CONFIG.models.embedder, "input": texts}
            )
        except (ProviderError, httpx.HTTPError) as exc:
            self.degraded.append(f"embeddings unavailable ({exc}); using BM25 only")
            return None

        ordered = sorted(data["data"], key=lambda d: d.get("index", 0))
        # Six decimals is far below the precision at which cosine similarity changes, and roughly
        # halves the stored size of every vector.
        vectors = [[round(x, 6) for x in d["embedding"]] for d in ordered]
        self.usage.embed_tokens += (data.get("usage") or {}).get("prompt_tokens", 0)
        if cacheable:
            self._write_cache(key, {"vectors": vectors})
        return vectors

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader — avoids a dependency for six lines of work."""
    path = path or (Path(__file__).resolve().parents[2] / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))
