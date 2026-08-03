---
doc_id: api-rate-limits
title: API Rate Limits
category: api_limits
owner: platform-team
last_reviewed: 2026-06-18
---

# API Rate Limits

Meridian enforces rate limits per API key, measured in requests per minute (RPM) using a sliding
window. Limits are applied independently to each key, not to the account as a whole.

## Limits by plan

| Plan | Sustained RPM | Burst ceiling | Concurrent connections |
|---|---|---|---|
| Developer (free) | 60 | 100 for 10 seconds | 5 |
| Starter | 300 | 500 for 30 seconds | 25 |
| Growth | 1,200 | 2,000 for 60 seconds | 100 |
| Enterprise | Negotiated, default 6,000 | Negotiated | 500 |

Burst capacity refills continuously. A key that has been idle accumulates burst allowance up to the
ceiling, then stops accumulating. Sustained traffic above the plan RPM will exhaust burst within the
stated window and begin returning `429`.

## Reading the rate limit headers

Every response includes the current state of your window:

```
X-RateLimit-Limit: 1200
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 1750000060
X-RateLimit-Window: 60
```

`X-RateLimit-Reset` is a Unix epoch timestamp in seconds. Clients should read `Remaining` and slow
down before it reaches zero rather than waiting for a `429`.

## What happens when you exceed the limit

Requests over the limit return HTTP `429 Too Many Requests` with a `Retry-After` header in seconds:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit of 1200 requests/minute exceeded for key mk_live_...4f2a",
    "retry_after": 12
  }
}
```

Rate-limited requests are **not** billed and do not count toward your monthly request allowance.

## Recommended client behaviour

Use exponential backoff with jitter, honouring `Retry-After` when present:

1. On `429`, sleep for `Retry-After` seconds if the header is present.
2. Otherwise sleep `min(2^attempt * 250ms, 30s)` plus random jitter of 0–250 ms.
3. Give up after 6 attempts and surface the failure to your own queue for later retry.

Do not retry immediately in a tight loop. Repeated tight-loop retries against a limited key can
trigger a temporary 15-minute block, which returns `429` with `code: "temporary_block"`.

## Per-endpoint sub-limits

Three endpoint groups carry their own stricter limits, independent of your plan RPM:

| Endpoint group | Limit |
|---|---|
| `POST /v1/exports` | 10 per hour |
| `POST /v1/batch` | 60 per hour, max 500 records per call |
| `POST /v1/auth/tokens` | 30 per hour |

Exceeding a sub-limit returns `429` with `code: "endpoint_limit_exceeded"` and names the group in
the message. Sub-limits are not raised on plan upgrade; they are raised only by request.

## Requesting a limit increase

Growth and Enterprise customers can request an increase from Settings → API → Request limit change.
Increases require a stated peak RPM and a short description of the traffic pattern. Standard review
takes two business days. Temporary increases for a launch or migration can be granted for up to 30
days and revert automatically.
