"""A saved lead to sales by e-mail: the visitor's name and e-mail address,
everything the card reader read, the product with its photo, and the
picture of the zone (card and product) as it was read.

Configured in /etc/default/phytec-detect (no address, no mail):

    PHYTEC_LEAD_MAIL_TO=sales@example.com      (several: comma-separated)
    PHYTEC_LEAD_MAIL_CC=me@example.com         (optional, comma-separated)
    PHYTEC_LEAD_SMTP=relay.example.com[:25]    (a relay without login)
    PHYTEC_LEAD_MAIL_FROM=demo@example.com     (default: the first recipient)

card_desk.CardDesk sends each new lead in a thread of its own -- once per
person and product, as it saves them once (leads.csv); a mail that
cannot go out is only logged (the lead is in leads.csv anyway). By hand:

    python3 -m phytec_detect.lead_mail --last     # the newest lead in leads.csv
"""

import argparse
import csv
import glob
import mimetypes
import os
import smtplib
import socket
import sys
from email.message import EmailMessage

LEADS_DIR = "/var/lib/phytec-detect/leads"
TIMEOUT_S = 20


def _addresses(name):
    return [a.strip() for a in os.environ.get(name, "").split(",") if a.strip()]


def config():
    """(recipients, cc, smtp host, port, sender) from the environment, None if off."""
    to = _addresses("PHYTEC_LEAD_MAIL_TO")
    relay = os.environ.get("PHYTEC_LEAD_SMTP", "").strip()
    if not to or not relay:
        return None
    host, _, port = relay.partition(":")
    sender = os.environ.get("PHYTEC_LEAD_MAIL_FROM", "").strip() or to[0]
    return to, _addresses("PHYTEC_LEAD_MAIL_CC"), host, int(port or 25), sender


def message(lead, zone_jpg=None, product_images=(), to=(), sender="", cc=()):
    """lead: dict with Zeit, Vorname, Nachname, E-Mail, Artikelnummer, Produkt,
    Text der Karte (a leads.csv row); product_images: (article, path) pairs."""
    name = f"{lead.get('Vorname', '')} {lead.get('Nachname', '')}".strip() or "unbekannt"
    msg = EmailMessage()
    msg["Subject"] = f"Messe-Lead: {name} – {lead.get('Produkt', '')}"
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    if lead.get("E-Mail"):
        msg["Reply-To"] = lead["E-Mail"]
    card = [t.strip() for t in lead.get("Text der Karte", "").split("|") if t.strip()]
    body = [
        "Ein Besucher hat am PHYTEC-Messestand Visitenkarte und Produkt abgelegt.",
        "",
        f"Zeit:          {lead.get('Zeit', '')}",
        f"Vorname:       {lead.get('Vorname', '')}",
        f"Nachname:      {lead.get('Nachname', '')}",
        f"E-Mail:        {lead.get('E-Mail') or '(nicht gelesen -- siehe Kartenbild)'}",
        f"Produkt:       {lead.get('Produkt', '')}",
        f"Artikelnummer: {lead.get('Artikelnummer', '')}",
        "",
        "Alles, was auf der Karte gelesen wurde (OCR, ungeprüft):",
        *[f"  {t}" for t in card],
        "",
        "Anhang: Bild der Ablage (Karte und Produkt), Produktbild.",
        "Die Angaben sind automatisch gelesen -- bitte mit dem Kartenbild abgleichen.",
    ]
    msg.set_content("\n".join(body))
    files = ([("ablage", zone_jpg)] if zone_jpg else []) + list(product_images)
    for label, path in files:
        if not path or not os.path.exists(path):
            continue
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        main, sub = ctype.split("/", 1)
        with open(path, "rb") as f:
            msg.add_attachment(f.read(), maintype=main, subtype=sub,
                               filename=f"{label}{os.path.splitext(path)[1]}")
    return msg


def send(lead, zone_jpg=None, product_images=(), log=print):
    """Mail one lead; False (logged) if off or it failed."""
    cfg = config()
    if cfg is None:
        return False
    to, cc, host, port, sender = cfg
    try:
        msg = message(lead, zone_jpg, product_images, to, sender, cc)
        with smtplib.SMTP(host, port, timeout=TIMEOUT_S) as smtp:
            smtp.ehlo()
            if smtp.has_extn("starttls"):
                smtp.starttls()
                smtp.ehlo()
            smtp.send_message(msg)
        log(f"lead mailed to {', '.join(to + cc)}: {lead.get('Vorname', '')} "
            f"{lead.get('Nachname', '')} {lead.get('Artikelnummer', '')}")
        return True
    except (OSError, smtplib.SMTPException, socket.timeout) as error:
        log(f"lead not mailed ({host}:{port}): {type(error).__name__}: {error}")
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--last", action="store_true", help="send the newest lead of leads.csv")
    parser.add_argument("--leads-dir", default=LEADS_DIR)
    parser.add_argument("--models", default="/opt/phytec-detect/models",
                        help="where to look for the product photos (*/images/<article>.jpg)")
    args = parser.parse_args(argv)
    if not args.last:
        parser.error("nothing to do (--last)")
    if config() is None:
        sys.exit("PHYTEC_LEAD_MAIL_TO / PHYTEC_LEAD_SMTP not set")
    with open(os.path.join(args.leads_dir, "leads.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    if not rows:
        sys.exit("no lead in leads.csv")
    lead = rows[-1]
    art = lead.get("Artikelnummer", "")
    pictures = sorted(glob.glob(os.path.join(args.leads_dir, f"*_{art}*.jpg")))
    photo = sorted(glob.glob(os.path.join(args.models, "*", "images", f"{art}.jpg")))
    ok = send(lead, pictures[-1] if pictures else None,
              [(art, photo[-1])] if photo else [])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
