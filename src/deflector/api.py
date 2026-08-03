"""HTTP surface.

The response shape is chosen for the two consumers that actually exist:

- the **ticketing system**, which needs `route` and nothing else to decide where the ticket goes; and
- the **support agent**, who needs to see why, with sources they can open.

Hence `route` at the top level rather than buried in a nested object, and the full decision record
alongside it rather than in a separate audit endpoint nobody wires up.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import CONFIG
from .pipeline import Deflector
from .providers import load_dotenv

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    # Build the index once at startup, not per request. Cold-start cost is paid by the deploy,
    # not by the first customer to open a ticket.
    _state["deflector"] = Deflector()
    _state["counters"] = {"auto_resolve": 0, "agent_assist": 0, "escalate": 0}
    yield
    _state.clear()


app = FastAPI(
    title="Deflector",
    version="1.0.0",
    description="Grounded support deflection with a calibrated confidence gate.",
    lifespan=lifespan,
)


class TicketRequest(BaseModel):
    body: str = Field(..., description="The customer's message.", min_length=1)
    subject: str = Field("", description="Ticket subject, if the channel has one.")
    ticket_id: str | None = Field(None, description="Your ID, echoed back for correlation.")


class TicketResponse(BaseModel):
    ticket_id: str
    route: str
    confidence_band: str
    confidence_score: float
    answer: str
    send_to_customer: bool
    citations: list[dict[str, str]]
    decision: dict[str, Any]
    screening: dict[str, Any]
    retrieved: list[dict[str, Any]]
    usage: dict[str, Any]
    meta: dict[str, Any]


@app.post("/v1/deflect", response_model=TicketResponse)
def deflect(request: TicketRequest) -> TicketResponse:
    deflector: Deflector = _state["deflector"]
    result = deflector.deflect(
        body=request.body, subject=request.subject, ticket_id=request.ticket_id
    )
    _state["counters"][result.route.value] += 1
    payload = result.to_dict()
    payload["send_to_customer"] = result.customer_facing() is not None
    return TicketResponse(**payload)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    deflector: Deflector | None = _state.get("deflector")
    return {
        "status": "ok",
        "chunks_indexed": len(deflector.retriever.chunks) if deflector else 0,
        "dense_retrieval": deflector.retriever.dense_available if deflector else False,
        "answerer": CONFIG.models.answerer,
        "verifier": CONFIG.models.verifier,
    }


@app.get("/v1/metrics")
def metrics() -> dict[str, Any]:
    """Deliberately minimal.

    In production this is a Prometheus exporter and the numbers below are labelled counters, but the
    metric that matters is the same one either way: what fraction of tickets the system took off the
    queue, and what it cost to do so.
    """
    counters = _state.get("counters", {})
    total = sum(counters.values()) or 1
    deflector: Deflector | None = _state.get("deflector")
    usage = deflector.provider.usage.to_dict() if deflector else {}
    return {
        "counts": counters,
        "deflection_rate": round(
            (counters.get("auto_resolve", 0) + counters.get("agent_assist", 0)) / total, 4
        ),
        "auto_resolve_rate": round(counters.get("auto_resolve", 0) / total, 4),
        "usage": usage,
        "thresholds": {
            "high": CONFIG.confidence.high,
            "medium": CONFIG.confidence.medium,
            "retrieval_floor": CONFIG.confidence.retrieval_floor,
        },
    }
