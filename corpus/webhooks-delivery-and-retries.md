---
doc_id: webhooks-delivery-and-retries
title: Webhook Delivery, Retries and Signatures
category: integration_errors
owner: platform-team
last_reviewed: 2026-06-11
---

# Webhook Delivery, Retries and Signatures

## Delivery expectations

Meridian sends webhooks as `POST` with a JSON body. Your endpoint must respond with any `2xx` status
within **10 seconds**. Anything else — a non-2xx status, a timeout, a TLS failure, or a connection
reset — is treated as a failed delivery and enters the retry schedule.

Do not do real work inside the request. Acknowledge with `200` immediately, enqueue the payload, and
process it asynchronously. Slow handlers are the single most common cause of webhook incidents.

## Retry schedule

Failed deliveries are retried 8 times with exponential backoff:

| Attempt | Delay after previous |
|---|---|
| 1 | 10 seconds |
| 2 | 30 seconds |
| 3 | 2 minutes |
| 4 | 10 minutes |
| 5 | 1 hour |
| 6 | 3 hours |
| 7 | 12 hours |
| 8 | 24 hours |

Total retry window is approximately 40 hours. After the eighth failure the event is marked
`undeliverable` and no further attempts are made. Undeliverable events remain replayable from the
dashboard for 30 days.

If an endpoint returns non-2xx for **20 consecutive** deliveries, it is automatically disabled and a
`webhook.endpoint_disabled` notification is emailed to account owners. Re-enable it from
Settings → Webhooks after fixing the handler.

## Signature verification

Every delivery carries:

```
Meridian-Signature: t=1750000000,v1=5257a869e7ecebeda32affa62cdca3fa51cad7e77a0e56ff536d0ce8e108d8bd
Meridian-Event-Id: evt_01J8Z9Q...
Meridian-Delivery-Attempt: 3
```

Verify by computing `HMAC-SHA256` over the literal string `{timestamp}.{raw_request_body}` using
your endpoint's signing secret, then comparing to `v1` in constant time.

Two failure modes account for most signature problems:

1. **Using a parsed and re-serialised body.** The signature is over the raw bytes. Re-serialising
   JSON changes key order and whitespace and will never match. Capture the raw body before parsing.
2. **Rejecting on timestamp skew.** We recommend rejecting deliveries where `t` is more than 5
   minutes from your clock, but a server with drifting time will then reject everything. Check NTP
   before assuming the signature logic is wrong.

## Idempotency and ordering

Delivery is **at-least-once**, not exactly-once. A retry after a timeout can deliver an event your
handler already processed. Deduplicate on `Meridian-Event-Id`, which is stable across all retries of
the same event.

Ordering is **not guaranteed**. Two events generated milliseconds apart may arrive in either order,
and a retried event will arrive after later events. Where order matters, use the `sequence` field in
the payload body rather than arrival order.

## Common HTTP-level failures

| Symptom in the delivery log | Usual cause |
|---|---|
| `connection_timeout` | Handler doing synchronous work; move to a queue |
| `tls_handshake_failed` | Expired or incomplete certificate chain; test with `openssl s_client` |
| `dns_resolution_failed` | Endpoint hostname no longer resolves |
| `http_405` | Endpoint route accepts GET only |
| `http_401` / `http_403` | A proxy or WAF in front of the handler is blocking us |
| `payload_too_large_rejected` | Handler body limit below 1 MB |

Webhook payloads can reach 1 MB. Configure your server's body limit accordingly.

## Replay

Individual events can be replayed from Settings → Webhooks → Deliveries for 30 days. Bulk replay of a
time range is available on Growth and Enterprise and is rate limited to one bulk job at a time.
