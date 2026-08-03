---
doc_id: billing-plans-and-invoicing
title: Plans, Seats and Invoicing
category: billing
owner: billing-team
last_reviewed: 2026-07-02
---

# Plans, Seats and Invoicing

## Plan pricing

| Plan | Base price | Included requests / month | Overage per 1,000 requests | Seats included |
|---|---|---|---|---|
| Developer | $0 | 50,000 | Not available — hard stop | 1 |
| Starter | $99 / month | 1,000,000 | $0.90 | 5 |
| Growth | $499 / month | 10,000,000 | $0.55 | 25 |
| Enterprise | Custom | Custom | Custom | Unlimited |

Annual billing is offered at a 15% discount on the base price. Overage is always billed monthly in
arrears regardless of billing interval.

Developer accounts do not accrue overage. When the 50,000 request allowance is exhausted, the API
returns `403` with `code: "quota_exhausted"` until the next monthly reset or an upgrade.

## Billing cycle

The billing period starts on the day of month you subscribed. Invoices are generated at 00:00 UTC on
the renewal date and charged to the default payment method within one hour. Overage for the period
just ended appears on the same invoice as the upcoming period's base price.

Invoices are available under Settings → Billing → Invoices as PDF and CSV. Account owners and users
with the `billing_admin` role can access them; `developer` and `viewer` roles cannot.

## Seats

A seat is any user with an active login on the account. Seats are counted at invoice generation
time, not continuously. Adding a seat mid-cycle triggers a prorated charge on the next invoice;
removing a seat mid-cycle produces a prorated credit, applied to the next invoice rather than
refunded to the payment method.

Additional seats beyond the plan allowance cost $19 per seat per month on Starter and $15 on Growth.

## Payment methods and failures

We accept credit and debit cards, ACH (US only), and wire transfer for Enterprise annual contracts.

If a charge fails, we retry on a fixed schedule: day 1, day 3, day 5, and day 7 after the failure.
Each attempt sends an email to all `billing_admin` users. After the fourth failed attempt the
account moves to `past_due`, which:

- keeps the API serving read endpoints,
- rejects write endpoints with `402` and `code: "payment_required"`,
- and suspends scheduled exports.

Accounts in `past_due` for 30 days are downgraded to Developer, and data beyond the Developer
retention window becomes subject to the retention policy.

## Tax and currency

All prices are in USD and exclusive of tax. VAT, GST and equivalent taxes are added at invoice time
based on the billing address on file. Customers with a valid tax registration number can enter it
under Settings → Billing → Tax details to apply reverse charge where applicable.

## Changing plans

Upgrades take effect immediately and are prorated for the remainder of the cycle. Downgrades take
effect at the end of the current cycle; the account keeps the higher plan's limits until then. A
downgrade that would leave more seats active than the target plan allows is blocked until seats are
removed.
