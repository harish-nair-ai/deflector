---
doc_id: data-export-and-retention
title: Data Export and Retention
category: account
owner: platform-team
last_reviewed: 2026-06-05
---

# Data Export and Retention

## Exports

Create an export with `POST /v1/exports`, specifying a resource type and an optional date range.
Exports are asynchronous. The response returns an export ID; poll `GET /v1/exports/{id}` or listen
for the `export.completed` webhook.

| Property | Value |
|---|---|
| Formats | CSV, JSONL, Parquet |
| Maximum rows per export | 50 million |
| Maximum date range per export | 24 months |
| Typical completion | Under 10 minutes for fewer than 1 million rows |
| Download URL validity | 24 hours from completion |
| Concurrent exports | 1 on Starter, 3 on Growth, 10 on Enterprise |
| Endpoint limit | 10 export creations per hour |

Download URLs are pre-signed and single-account scoped. If a URL expires, re-request it with
`GET /v1/exports/{id}/url` — the export itself does not need to be recreated, provided it is within
the 30-day artifact window.

Exports of more than 10 million rows are automatically split into multiple part files, listed under
`parts` in the export object.

## Retention while the account is active

| Data type | Retention |
|---|---|
| Records | Indefinite while the account is active |
| API request logs | 30 days on Starter, 90 days on Growth, 400 days on Enterprise |
| Webhook delivery logs | 30 days, all plans |
| Export artifacts | 30 days from completion |
| Audit log | 400 days on Enterprise, 90 days otherwise |

## Retention after downgrade or closure

- **Downgrade to Developer.** Records are preserved, but log retention shortens to the Developer
  window (7 days) immediately. Logs already older than 7 days are removed at the next daily sweep.
- **Cancellation.** Data is retained for 90 days after the paid period ends. Reactivating within
  that window restores everything.
- **After 90 days.** Data is queued for deletion and removed within a further 30 days.

## Deletion requests

Submit a deletion request from Settings → Privacy → Delete data, or by emailing the privacy team.
Requests require confirmation from an account owner.

- Deletion completes within **30 days** of confirmation.
- Backups are purged on their own rolling cycle, completing within **90 days** total.
- A deletion certificate is issued on completion if requested at submission time.
- Deletion is irreversible. There is no undo, and support cannot restore deleted data.

Individual end-user deletion (for a data subject request) can be scoped to a single record subject
via `DELETE /v1/records/{id}?erase=true`, which performs a hard erase rather than a soft delete and
is likewise irreversible.

## Data residency

Accounts are provisioned in one of three regions, chosen at signup and fixed thereafter:

| Region | Location |
|---|---|
| `us-east` | Virginia, USA |
| `eu-central` | Frankfurt, Germany |
| `ap-south` | Mumbai, India |

Records stay in the account's region. Migrating an account between regions is a manual,
Enterprise-only project handled by the platform team and typically takes 2–4 weeks.
