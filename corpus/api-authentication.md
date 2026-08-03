---
doc_id: api-authentication
title: API Authentication and Keys
category: api_errors
owner: platform-team
last_reviewed: 2026-05-30
---

# API Authentication and Keys

## Key format

Meridian API keys carry an environment prefix:

- `mk_live_…` — production, operates on real data
- `mk_test_…` — sandbox, isolated data, no billing impact

Keys are 40 characters after the prefix. The full key value is shown exactly once, at creation. We
store only a hash, so a lost key cannot be recovered — it must be rotated.

## Sending the key

Pass the key as a bearer token:

```
Authorization: Bearer mk_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Query-string authentication (`?api_key=`) was removed in API version `2025-11-01`. Requests using it
against a current version receive `401` with `code: "auth_scheme_unsupported"`.

## Scopes

Each key carries an explicit scope set. A key with insufficient scope receives `403`, not `401`.

| Scope | Grants |
|---|---|
| `read:records` | GET on `/v1/records` and `/v1/records/{id}` |
| `write:records` | POST, PATCH, DELETE on `/v1/records` |
| `read:billing` | GET on `/v1/invoices`, `/v1/usage` |
| `manage:webhooks` | Full access to `/v1/webhooks` |
| `admin` | All of the above plus `/v1/auth/tokens` |

Scopes cannot be edited after creation. To change scopes, create a new key and retire the old one.

## Distinguishing 401 from 403

| Status | Code | Meaning | Fix |
|---|---|---|---|
| 401 | `auth_missing` | No `Authorization` header | Send the header |
| 401 | `auth_invalid` | Key not recognised or revoked | Check for whitespace or truncation; rotate if revoked |
| 401 | `auth_scheme_unsupported` | Using query-string auth | Move to bearer token |
| 401 | `auth_env_mismatch` | `mk_test_` key against a production endpoint | Use the matching environment |
| 403 | `scope_insufficient` | Valid key, wrong scope | Create a key with the needed scope |
| 403 | `ip_not_allowed` | Source IP outside the allowlist | Add the IP under Settings → API → IP allowlist |
| 403 | `quota_exhausted` | Developer plan allowance spent | Upgrade or wait for monthly reset |

The most common cause of `auth_invalid` in practice is a truncated key copied from a terminal that
wrapped the line, or a trailing newline picked up by `$(cat key.txt)`. Compare the last four
characters against the key's displayed suffix in the dashboard.

## Rotation

Create the new key, deploy it, confirm traffic has moved using the per-key request chart, then
revoke the old key. Revocation takes effect within 30 seconds globally.

Keys can be given an expiry at creation, from 24 hours to 365 days. Expiring keys emit a
`key.expiring` webhook 7 days and 24 hours before expiry.

## IP allowlisting

Available on Growth and Enterprise. Up to 50 CIDR ranges per key. An empty allowlist means all IPs
are permitted. Changes propagate within 60 seconds. IPv6 ranges are supported; ranges wider than
`/24` on IPv4 are rejected.
