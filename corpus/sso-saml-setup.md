---
doc_id: sso-saml-setup
title: SSO and SAML Setup
category: integration_errors
owner: identity-team
last_reviewed: 2026-04-14
---

# SSO and SAML Setup

SAML 2.0 single sign-on is available on **Growth and Enterprise** plans. SCIM provisioning is
Enterprise only.

## Setup order

1. In Meridian, go to Settings → Security → SSO and copy the **ACS URL** and **Entity ID**.
2. Create the application in your identity provider using those values.
3. Copy the IdP metadata URL back into Meridian, or upload the metadata XML.
4. Map attributes (below).
5. Test with **Test connection** before enforcing. Testing does not affect live logins.
6. Enable **Enforce SSO** once a successful test is recorded.

Enforcing SSO disables password login for all members except users holding the `owner` role, who
keep a password fallback. This is deliberate: it prevents a misconfigured IdP from locking every
administrator out of the account.

## Required attribute mapping

| Meridian attribute | Expected SAML claim | Required |
|---|---|---|
| `email` | `NameID` (format `emailAddress`) or `http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress` | Yes |
| `first_name` | `givenname` | No |
| `last_name` | `surname` | No |
| `groups` | `groups` (multi-valued) | Only if using group-to-role mapping |

Email is matched case-insensitively against existing members. A successful assertion for an unknown
email creates a new member with the default role if **Just-in-time provisioning** is on; otherwise
it fails with `sso_user_not_provisioned`.

## Common SSO errors

| Error | Cause | Fix |
|---|---|---|
| `sso_signature_invalid` | IdP signing certificate rotated | Re-import metadata; enable auto-refresh from the metadata URL |
| `sso_audience_mismatch` | Entity ID in the IdP does not match ours exactly | Compare character by character; a trailing slash breaks it |
| `sso_assertion_expired` | Clock skew over 3 minutes | Check NTP on the IdP |
| `sso_user_not_provisioned` | JIT off and user does not exist | Invite the user, or enable JIT |
| `sso_no_email_claim` | NameID format is `persistent`, not `emailAddress` | Change the NameID format in the IdP |
| `sso_replay_detected` | Assertion ID reused | Usually a browser back-button retry; retry the login cleanly |

## Certificate rotation

Meridian caches IdP metadata for 24 hours. If auto-refresh is enabled we pick up a rotated
certificate within that window. If it is disabled, a rotation breaks every login immediately with
`sso_signature_invalid` and requires a manual metadata re-import.

We strongly recommend enabling auto-refresh from a metadata URL rather than uploading static XML.

## SCIM provisioning

Enterprise accounts can provision and deprovision users via SCIM 2.0. The base URL and bearer token
are generated under Settings → Security → SCIM. Supported operations are user create, update,
deactivate, and group membership sync. Deactivation via SCIM revokes active sessions within 5
minutes.

Deleting a user in SCIM deactivates the Meridian member rather than deleting it, preserving audit
history.
