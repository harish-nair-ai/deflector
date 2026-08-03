---
doc_id: getting-started-and-sdks
title: Getting Started and Official SDKs
category: integration_errors
owner: devrel
last_reviewed: 2026-06-30
---

# Getting Started and Official SDKs

## First request

```bash
curl https://api.meridian.dev/v1/records \
  -H "Authorization: Bearer mk_test_..." \
  -H "Content-Type: application/json" \
  -d '{"type": "invoice", "amount": 4200, "currency": "USD"}'
```

`amount` is always in **minor units** — 4200 means $42.00. This is the most frequent cause of
`validation_failed` on a first integration, and of support tickets about amounts being "100x wrong".

## Base URLs and versioning

| Environment | Base URL |
|---|---|
| Production | `https://api.meridian.dev` |
| Sandbox | `https://api.meridian.dev` with an `mk_test_` key |

There is no separate sandbox hostname. The key prefix selects the environment.

API versions are dated, e.g. `2026-03-01`. Your account is pinned to the version current at signup.
Send `Meridian-Version: 2026-03-01` to override per request. Versions remain supported for a minimum
of 24 months after superseding. Breaking changes never ship to a pinned version.

## Official SDKs

| Language | Package | Minimum runtime |
|---|---|---|
| Python | `meridian-python` | 3.9 |
| Node / TypeScript | `@meridian/node` | Node 18 |
| Go | `github.com/meridian-dev/meridian-go` | Go 1.21 |
| Ruby | `meridian-ruby` | Ruby 3.1 |
| Java | `dev.meridian:meridian-java` | Java 11 |

SDKs implement retries with backoff, idempotency key generation for writes, and cursor
auto-pagination. Community SDKs exist for PHP and .NET but are not maintained by Meridian and are
not covered by support.

## Timeouts in the SDKs

Default client timeout is 30 seconds with 3 retries on `5xx` and `429`. Override per client:

```python
from meridian import Meridian
client = Meridian(api_key="mk_live_...", timeout=10.0, max_retries=5)
```

Retries only apply to idempotent requests. A `POST` without an idempotency key is not retried
automatically, because the SDK cannot know whether the first attempt took effect.

## Testing

The sandbox accepts deterministic trigger values that force specific outcomes:

| Value | Effect |
|---|---|
| `amount: 1` | Forces `validation_failed` |
| `amount: 402` | Forces `payment_required` |
| `amount: 429` | Forces `rate_limit_exceeded` |
| `amount: 500` | Forces `internal_error` |
| `amount: 504` | Forces `upstream_timeout` |

These triggers work only with `mk_test_` keys.

## Support channels

| Plan | Channel | First response target |
|---|---|---|
| Developer | Community forum | None |
| Starter | Email | 1 business day |
| Growth | Email and chat | 4 business hours |
| Enterprise | Email, chat, shared Slack channel | 1 hour for `urgent` |

Include the `request_id` from the error envelope in every technical ticket. Without it, diagnosis
usually requires a round trip to ask for it.
