"""Generates the messy source documents the ingestion pipeline is tested against.

This is a fixture builder, not part of the service. Real support knowledge bases arrive as PDF
manuals and help-centre HTML exports, so the corpus deliberately contains both rather than the clean
markdown that makes RAG demos look easy.

Two properties are engineered on purpose:

  * the error-code table is long enough to span a page break in the PDF, so the ingester's
    header-propagation logic is actually exercised rather than merely claimed;
  * the guide embeds a rendered architecture figure, so the figure-captioning path has real image
    bytes to work with.

Run: python tools/build_source_docs.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "corpus_raw"
RAW.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------------------
# The architecture figure
# ---------------------------------------------------------------------------------------

def build_figure() -> Path:
    """Draw a request-lifecycle diagram containing facts that appear nowhere in the prose.

    This matters for the evaluation: if the only place the 30-second gateway timeout is stated is
    inside this image, then answering a question about it proves the figure was genuinely read, not
    that the answer leaked in from surrounding text.
    """
    W, H = 1500, 620
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    def font(size: int):
        for path in (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    f_title, f_box, f_small = font(34), font(23), font(18)

    d.text((40, 26), "Meridian request lifecycle", fill="black", font=f_title)

    boxes = [
        (40, 130, "Client SDK", "retries 3x\non 5xx / 429", "#e8f0fe"),
        (330, 130, "Edge gateway", "TLS 1.2+\ntimeout 30s", "#fce8e6"),
        (620, 130, "Rate limiter", "sliding window\nper API key", "#fef7e0"),
        (910, 130, "Auth service", "scope check\nkey hash lookup", "#e6f4ea"),
        (1200, 130, "Core API", "writes are\nidempotent 24h", "#f3e8fd"),
    ]

    for x, y, title, sub, colour in boxes:
        d.rounded_rectangle([x, y, x + 250, y + 170], radius=14, fill=colour, outline="#5f6368", width=3)
        d.text((x + 20, y + 22), title, fill="black", font=f_box)
        for i, line in enumerate(sub.split("\n")):
            d.text((x + 20, y + 68 + i * 28), line, fill="#3c4043", font=f_small)

    for x, _, *_ in boxes[:-1]:
        d.line([x + 250, 215, x + 290, 215], fill="#5f6368", width=4)
        d.polygon([(x + 290, 215), (x + 276, 207), (x + 276, 223)], fill="#5f6368")

    # The failure path, with numbers stated only here.
    d.rounded_rectangle([330, 400, 900, 560], radius=14, fill="#fff3e0", outline="#e8710a", width=3)
    d.text((355, 420), "Rejected before reaching Core API", fill="#b06000", font=f_box)
    d.text((355, 464), "429 rate_limit_exceeded  ->  honour Retry-After", fill="#3c4043", font=f_small)
    d.text((355, 494), "401 auth_invalid  ->  key revoked or truncated", fill="#3c4043", font=f_small)
    d.text((355, 524), "Gateway rejects bodies over 1 MB before auth runs", fill="#3c4043", font=f_small)

    d.line([455, 300, 455, 400], fill="#e8710a", width=4)
    d.polygon([(455, 400), (447, 386), (463, 386)], fill="#e8710a")
    d.line([745, 300, 745, 400], fill="#e8710a", width=4)
    d.polygon([(745, 400), (737, 386), (753, 386)], fill="#e8710a")

    d.text((955, 470), "Figure 1 — request path and rejection points", fill="#5f6368", font=f_small)

    path = RAW / "figure-request-lifecycle.png"
    img.save(path, "PNG")
    return path


# ---------------------------------------------------------------------------------------
# The PDF operations guide
# ---------------------------------------------------------------------------------------

def build_pdf(figure_path: Path) -> Path:
    path = RAW / "meridian-operations-guide.pdf"
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14, spaceAfter=8)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=17, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6)

    def table(data, widths):
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eaed")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9aa0a6")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return t

    story = []
    story.append(Paragraph("Meridian Platform Operations Guide", h1))
    story.append(Paragraph("Document version 4.2 &middot; reviewed 2026-06-28 &middot; platform team", body))
    story.append(
        Paragraph(
            "This guide covers operational limits, delivery guarantees and failure handling for the "
            "Meridian API. It supersedes the operations appendix of the integration handbook.",
            body,
        )
    )

    story.append(Paragraph("1. Support response targets", h2))
    story.append(
        Paragraph(
            "Response targets are measured from ticket creation to first human response, during the "
            "support hours applicable to the plan.",
            body,
        )
    )
    story.append(
        table(
            [
                ["Plan", "First response", "Support hours", "Named contact"],
                ["Developer", "No target", "Community forum only", "No"],
                ["Starter", "1 business day", "09:00-17:00 local", "No"],
                ["Growth", "4 business hours", "09:00-21:00 local", "No"],
                ["Enterprise (standard)", "1 hour for urgent", "24x5", "Yes"],
                ["Enterprise (premier)", "15 minutes for urgent", "24x7", "Yes, plus on-call"],
            ],
            [95, 105, 130, 110],
        )
    )
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("2. Uptime commitments and service credits", h2))
    story.append(
        Paragraph(
            "Service credits are issued against the monthly invoice for the affected month. Credits "
            "must be requested within 30 days of the incident and are the sole remedy for downtime.",
            body,
        )
    )
    story.append(
        table(
            [
                ["Monthly uptime", "Service credit", "Applies to"],
                ["99.95% or above", "None", "All paid plans"],
                ["99.00% to 99.94%", "10% of monthly fee", "Growth and Enterprise"],
                ["95.00% to 98.99%", "25% of monthly fee", "Growth and Enterprise"],
                ["Below 95.00%", "50% of monthly fee", "Growth and Enterprise"],
            ],
            [110, 130, 160],
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        Paragraph(
            "The uptime commitment is 99.95% for Growth and 99.99% for Enterprise. Scheduled "
            "maintenance announced at least 72 hours in advance is excluded from the calculation.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("3. Request lifecycle", h2))
    story.append(
        Paragraph(
            "The diagram below shows the path a request takes and the points at which it can be "
            "rejected before reaching the core API.",
            body,
        )
    )
    story.append(RLImage(str(figure_path), width=170 * mm, height=70 * mm))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("4. Incident severity definitions", h2))
    story.append(
        table(
            [
                ["Severity", "Definition", "Target mitigation"],
                ["Sev-1", "Complete outage or data loss affecting many customers", "60 minutes"],
                ["Sev-2", "Major feature unavailable, no workaround", "4 hours"],
                ["Sev-3", "Degraded performance, workaround exists", "2 business days"],
                ["Sev-4", "Cosmetic or documentation defect", "Next release"],
            ],
            [60, 250, 100],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("5. Error code reference", h2))
    story.append(
        Paragraph(
            "The table below spans a page boundary. Each row is independent; the header repeats on "
            "continuation pages.",
            body,
        )
    )

    rows = [["Code", "HTTP", "Retryable", "Operator action"]]
    catalogue = [
        ("rate_limit_exceeded", "429", "Yes, after Retry-After", "None; client must back off"),
        ("endpoint_limit_exceeded", "429", "Yes, next hour", "Raise sub-limit if justified"),
        ("temporary_block", "429", "After 15 minutes", "Check for tight retry loop"),
        ("auth_invalid", "401", "No", "Confirm key not revoked or truncated"),
        ("auth_env_mismatch", "401", "No", "Test key used against production"),
        ("scope_insufficient", "403", "No", "Issue key with required scope"),
        ("ip_not_allowed", "403", "No", "Add source IP to allowlist"),
        ("quota_exhausted", "403", "Next billing cycle", "Upgrade plan"),
        ("payment_required", "402", "After payment", "Route to billing team"),
        ("resource_not_found", "404", "No", "Check environment and account"),
        ("conflict_version", "409", "Yes, after re-read", "None; optimistic concurrency"),
        ("idempotency_key_reused", "409", "No", "Client bug; key reused with new body"),
        ("payload_too_large", "413", "No", "Advise batch endpoint"),
        ("unprocessable_entity", "422", "No", "Semantically impossible request"),
        ("internal_error", "500", "Yes, with backoff", "Check incident channel"),
        ("service_unavailable", "503", "Yes, honour Retry-After", "Deploy or capacity event"),
        ("upstream_timeout", "504", "Yes, idempotent only", "Check dependency dashboards"),
        ("malformed_json", "400", "No", "Client serialisation bug"),
        ("missing_content_type", "400", "No", "Advise application/json"),
        ("cursor_expired", "400", "No, restart iteration", "Cursor older than 24 hours"),
        ("method_not_allowed", "405", "No", "Wrong verb for endpoint"),
        ("unsupported_media_type", "415", "No", "Non-JSON content type"),
        ("validation_failed", "400", "No", "Read the field property"),
        ("sso_signature_invalid", "401", "No", "IdP certificate rotated"),
        ("sso_audience_mismatch", "401", "No", "Entity ID mismatch in IdP"),
        ("sso_assertion_expired", "401", "Retry login", "Clock skew above 3 minutes"),
        ("sso_user_not_provisioned", "403", "No", "Enable JIT or invite user"),
        ("webhook_signature_mismatch", "n/a", "No", "Raw body re-serialised by client"),
        ("export_limit_exceeded", "429", "Next hour", "10 export creations per hour"),
        ("region_mismatch", "403", "No", "Account pinned to another region"),
        ("tls_version_unsupported", "n/a", "No", "TLS 1.0/1.1 disabled 2025-09-01"),
        ("auth_missing", "401", "No", "No Authorization header sent"),
        ("auth_scheme_unsupported", "401", "No", "Query-string auth removed 2025-11-01"),
        ("scim_token_invalid", "401", "No", "Regenerate SCIM bearer token"),
        ("scim_user_conflict", "409", "No", "Duplicate externalId in IdP"),
        ("export_range_too_wide", "422", "No", "Maximum 24 months per export"),
        ("export_row_limit", "422", "No", "Maximum 50 million rows"),
        ("export_url_expired", "410", "Re-request URL", "Download URL valid 24 hours"),
        ("webhook_endpoint_disabled", "n/a", "No", "20 consecutive non-2xx deliveries"),
        ("webhook_undeliverable", "n/a", "Replay only", "8 attempts exhausted over 40 hours"),
        ("batch_size_exceeded", "422", "No", "Maximum 500 records per batch call"),
        ("concurrent_export_limit", "429", "When one finishes", "Plan concurrency ceiling reached"),
        ("connection_ceiling", "n/a", "Yes", "Connection reset, not an HTTP error"),
        ("version_unsupported", "400", "No", "Pinned version past 24-month window"),
        ("field_immutable", "422", "No", "Field cannot change after creation"),
        ("account_suspended", "403", "No", "Chargeback or compliance hold"),
    ]
    rows.extend([list(r) for r in catalogue])
    story.append(table(rows, [120, 42, 110, 175]))

    SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Meridian Platform Operations Guide",
    ).build(story)
    return path


# ---------------------------------------------------------------------------------------
# The help-centre HTML export
# ---------------------------------------------------------------------------------------

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Managing seats and usage &middot; Meridian Help Centre</title>
  <meta name="article-id" content="hc-402">
  <meta name="category" content="billing">
  <meta name="last-reviewed" content="2026-07-09">
</head>
<body>
  <nav class="breadcrumbs"><a href="/">Help</a> &rsaquo; <a href="/billing">Billing</a> &rsaquo; Seats</nav>
  <header><h1>Managing seats and usage</h1></header>

  <article>
    <p>Seat charges are the most common source of billing questions. This article explains how seats
    are counted, when they are charged, and how to read the usage dashboard.</p>

    <h2>How seats are counted</h2>
    <p>A seat is any user with an active login. Seats are counted at the moment the invoice is
    generated, not continuously through the month. Deactivating a user before the invoice date
    therefore avoids the charge entirely.</p>

    <table class="pricing">
      <caption>Seat allowances and overage pricing</caption>
      <thead>
        <tr><th>Plan</th><th>Seats included</th><th>Extra seat / month</th><th>Maximum seats</th></tr>
      </thead>
      <tbody>
        <tr><td>Developer</td><td>1</td><td>Not available</td><td>1</td></tr>
        <tr><td>Starter</td><td>5</td><td>$19</td><td>25</td></tr>
        <tr><td>Growth</td><td>25</td><td>$15</td><td>250</td></tr>
        <tr><td>Enterprise</td><td>Unlimited</td><td>Included</td><td>Unlimited</td></tr>
      </tbody>
    </table>

    <h2>Roles and what they can see</h2>
    <p>Billing visibility is role-gated. A developer who cannot find the invoices page is almost
    always missing the <code>billing_admin</code> role rather than hitting a bug.</p>

    <table class="roles">
      <caption>Role permissions for billing surfaces</caption>
      <thead>
        <tr><th>Role</th><th>View invoices</th><th>Change payment method</th><th>Add seats</th><th>Cancel plan</th></tr>
      </thead>
      <tbody>
        <tr><td>owner</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
        <tr><td>billing_admin</td><td>Yes</td><td>Yes</td><td>Yes</td><td>No</td></tr>
        <tr><td>developer</td><td>No</td><td>No</td><td>No</td><td>No</td></tr>
        <tr><td>viewer</td><td>No</td><td>No</td><td>No</td><td>No</td></tr>
      </tbody>
    </table>

    <h2>Reading the usage dashboard</h2>
    <p>The usage chart shows requests per day against your monthly allowance. The shaded band is
    projected usage for the remainder of the cycle, based on the trailing seven-day average.</p>
    <figure>
      <img src="images/usage-dashboard.png" alt="Usage dashboard showing a request volume chart with a projection band and an overage warning banner">
      <figcaption>Figure 2 &mdash; the usage dashboard, with the projected-overage warning visible.</figcaption>
    </figure>
    <p>The amber banner appears once projected usage exceeds 90% of the allowance. It is a
    projection, not a charge, and disappears if traffic falls back.</p>

    <h2>Common questions</h2>
    <dl>
      <dt>Does removing a seat refund me?</dt>
      <dd>No. Removing a seat mid-cycle produces a prorated <em>credit</em> applied to the next
      invoice. It is not returned to the payment method.</dd>
      <dt>Are deactivated users still charged?</dt>
      <dd>No, provided they were deactivated before the invoice generation date.</dd>
      <dt>Can I exceed the maximum seat count?</dt>
      <dd>Not self-serve. Growth accounts needing more than 250 seats must move to Enterprise.</dd>
    </dl>
  </article>
</body>
</html>
"""


def build_html() -> Path:
    path = RAW / "help-centre-seats-and-usage.html"
    path.write_text(HTML, encoding="utf-8")

    # The figure the article references, so the ingester has a real image to caption.
    img = Image.new("RGB", (1200, 560), "white")
    d = ImageDraw.Draw(img)

    def font(size: int):
        for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
        return ImageFont.load_default()

    f_t, f_s = font(28), font(18)
    d.text((36, 26), "Usage — current billing cycle", fill="black", font=f_t)
    d.rounded_rectangle([36, 80, 1164, 140], radius=8, fill="#fef7e0", outline="#e8710a", width=3)
    d.text((58, 98), "Projected usage is 94% of your 10,000,000 request allowance", fill="#b06000", font=f_s)

    base = 480
    heights = [70, 95, 88, 130, 150, 142, 175, 168, 190, 205, 198, 232, 246, 240]
    for i, h in enumerate(heights):
        x = 60 + i * 58
        d.rectangle([x, base - h, x + 40, base], fill="#1a73e8")
    for i in range(6):
        x = 60 + (len(heights) + i) * 58
        h = 250 + i * 9
        d.rectangle([x, base - h, x + 40, base], fill="#c6dafc")

    d.line([36, base, 1164, base], fill="#5f6368", width=3)
    d.text((60, base + 14), "day 1", fill="#5f6368", font=f_s)
    d.text((1040, base + 14), "day 20 (projected)", fill="#5f6368", font=f_s)
    d.text((36, base + 48), "Solid = actual   ·   Pale = projected", fill="#5f6368", font=f_s)

    images_dir = RAW / "images"
    images_dir.mkdir(exist_ok=True)
    img.save(images_dir / "usage-dashboard.png", "PNG")
    return path


if __name__ == "__main__":
    fig = build_figure()
    pdf = build_pdf(fig)
    html = build_html()
    for p in (fig, pdf, html):
        print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size:,} bytes)")
