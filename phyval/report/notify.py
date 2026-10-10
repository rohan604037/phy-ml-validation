"""
Report publication by email.

Always writes a ready-to-send .eml file (opens in Outlook, Thunderbird or Apple Mail).
If SMTP settings are provided through environment variables, it also sends it:

    PHYVAL_SMTP_HOST      e.g. smtp.gmail.com
    PHYVAL_SMTP_PORT      e.g. 587 (STARTTLS)
    PHYVAL_SMTP_USER      login user name
    PHYVAL_SMTP_PASSWORD  password or app password (never hard-code it)
    PHYVAL_SMTP_FROM      sender address (defaults to the user name)

SharePoint upload is not implemented: it needs a Microsoft 365 tenant and Graph API
credentials. The HTML report is a single self-contained file, so it can be uploaded
to SharePoint or any document library as is.
"""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


def make_email(subject: str, summary_html: str, attachment: Path, to: str | None) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("PHYVAL_SMTP_FROM") or os.environ.get("PHYVAL_SMTP_USER") or "phyval@localhost"
    msg["To"] = to or "validation-team@example.com"
    msg.set_content("This message contains an HTML summary. The full report is attached.")
    msg.add_alternative(summary_html, subtype="html")
    msg.add_attachment(Path(attachment).read_bytes(), maintype="text", subtype="html", filename=Path(attachment).name)
    return msg


def publish(subject: str, summary_html: str, report_path: Path, out_dir: Path, to: str | None = None, log=print) -> Path:
    msg = make_email(subject, summary_html, report_path, to)
    eml = Path(out_dir) / (Path(report_path).stem + ".eml")
    eml.write_bytes(bytes(msg))
    log(f"  email draft written: {eml}")
    host = os.environ.get("PHYVAL_SMTP_HOST")
    if to and host:
        port = int(os.environ.get("PHYVAL_SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.starttls()
            if os.environ.get("PHYVAL_SMTP_USER"):
                s.login(os.environ["PHYVAL_SMTP_USER"], os.environ.get("PHYVAL_SMTP_PASSWORD", ""))
            s.send_message(msg)
        log(f"  email sent to {to}")
    elif to:
        log("  (PHYVAL_SMTP_HOST not set - email saved as .eml but not sent)")
    return eml
