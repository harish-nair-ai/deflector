---
doc_id: integration-errors-reference
title: Integration Error Reference
category: integration_errors
owner: platform-team
last_reviewed: 2026-06-25
---

# Integration Error Reference

All errors share one envelope. `request_id` is the first thing to quote when contacting support.

```json
{
  "error": {
    "code": "validation_failed",
    "message": "Field 'amount' must be a positive integer in minor units",
    "field": "amount",
    "doc_url": "https://docs.meridian.dev/errors/validation_failed",
    "request_id": "req_01J8ZB4K7Q2M"
  }
}
```

## 4xx — the caller must change something

| Code | HTTP | Meaning | Resolution |
|---|---|---|---|
| `validation_failed` | 400 | A field failed schema validation | Read `field`; check types and required fields |
| `malformed_json` | 400 | Body is not parseable JSON | Usually a trailing comma or unescaped newline |
| `missing_content_type` | 400 | `Content-Type` absent | Send `application/json` |
| `auth_invalid` | 401 | Key not recognised | See the authentication doc |
| `scope_insufficient` | 403 | Key lacks the required scope | Issue a key with the scope |
| `resource_not_found` | 404 | ID does not exist, or belongs to another account | Confirm you are using the right environment |
| `method_not_allowed` | 405 | Wrong HTTP verb | Check the endpoint reference |
| `conflict_version` | 409 | `If-Match` ETag no longer current | Re-read the record and retry |
| `idempotency_key_reused` | 409 | Same key, different body | Use a fresh key, or resend the identical body |
| `payload_too_large` | 413 | Body over 1 MB | Use `/v1/batch` or chunk the upload |
| `unsupported_media_type` | 415 | Non-JSON content type | Send JSON |
| `unprocessable_entity` | 422 | Schema valid but semantically impossible | e.g. end date before start date |
| `rate_limit_exceeded` | 429 | Over plan RPM | Back off; see the rate limits doc |

## 5xx — retry is appropriate

| Code | HTTP | Meaning | Resolution |
|---|---|---|---|
| `internal_error` | 500 | Unhandled fault on our side | Retry with backoff; quote `request_id` if persistent |
| `upstream_timeout` | 504 | A downstream dependency timed out | Safe to retry; use idempotency keys on writes |
| `service_unavailable` | 503 | Deploy or degraded capacity | Honour `Retry-After` |

`500` responses are always logged on our side against the same `request_id` you receive.

## Idempotency

Every `POST` accepts an `Idempotency-Key` header, any string up to 255 characters. We store the
first response for **24 hours** and replay it for repeats of the same key.

- Same key + identical body → the original response is replayed, including its status code.
- Same key + different body → `409 idempotency_key_reused`.
- Keys are scoped per API key. Two different API keys can use the same idempotency key value safely.

Use a UUID per logical operation, not per HTTP attempt. Generating a new key on retry defeats the
purpose and can create duplicate records.

## Pagination

List endpoints are cursor-paginated:

```
GET /v1/records?limit=100&starting_after=rec_01J8ZB...
```

`limit` defaults to 25 and caps at 100. The response carries `has_more` and `next_cursor`. Do not
build cursors yourself; they are opaque and their format can change. Offset pagination is not
supported.

Cursors expire after 24 hours. A stale cursor returns `400` with `code: "cursor_expired"`; restart
the iteration.

## Timeouts and connection settings

- Server-side request timeout: 30 seconds. Long-running work should go through `/v1/exports`.
- Keep-alive idle timeout: 90 seconds.
- We require TLS 1.2 or higher. TLS 1.0 and 1.1 were disabled on 2025-09-01.
- Connections are refused after the plan's concurrent connection ceiling; the client sees a
  connection reset rather than an HTTP error.

## Sandbox differences

`mk_test_` keys operate on isolated data. Two behaviours differ deliberately: webhooks fire
immediately with no retry backoff, and exports complete synchronously. Do not use sandbox timing to
size production expectations.
