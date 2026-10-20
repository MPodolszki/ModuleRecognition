#!/usr/bin/env python3
"""dataset/products -> products.json for the board's info panel.

Per class (article number, the model's label):
  name     Bezeichnung from products.csv (falls back to the article number)
  summary  short lines for the screen: summary.txt if present, otherwise
           the populated options from info.tsv
  options  the full info.tsv rows (column name -> value), for later use
  link     first line of link.txt if present (shop page; the panel shows it
           as a QR code)
  chart    chart.png/.jpg if present, copied to <model>/charts/<article>.<ext>
           (the panel shows it next to the QR code)

produktinfo.txt in each product folder (filled in by hand, "Feld: Wert",
see FIELDS; written for every product by --vorlagen) comes first, an old
produktinfos.csv row is used where a folder has none: its short description, highlights, key figures (cores, size, temperature,
available until), category, "new" flag, product page and kit links (QR
codes) and chart file override the per-folder files.

    uv run tools/product_info.py models/products_v1_n
    uv run tools/product_info.py --vorlagen      # produktinfo.txt where missing
"""

import argparse
import csv
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
INFO_FILE = "produktinfo.txt"

# (number in the layout picture, field, where it shows up in the panel);
# number None = not shown. tools/render_panel.py draws the same numbers.
FIELDS = [
    (1, "Bezeichnung", "Titel (groß) und letzter Teil im Kopfbalken"),
    (2, "Kategorie", "graue Zeile über dem Titel und Kopfbalken, z. B. System on Module, "
                     "Development Kit"),
    (3, "Neu", "orangefarbenes NEU neben der Kategorie: ja / nein"),
    (4, "Artikelnummer", "unter dem Titel -- kommt aus dem Ordnernamen, nicht eintragen"),
    (5, "Kurzbeschreibung", "1-2 Sätze unter der Artikelnummer, höchstens 3 Zeilen "
                            "(ca. 270 Zeichen)"),
    (6, "Kerne", "Kennzahl 1, kurz (ca. 16 Zeichen), z. B. 4 × A53 + M7"),
    (7, "Größe in mm", "Kennzahl 2, z. B. 40 × 37 × 6"),
    (8, "Temperatur", "Kennzahl 3, z. B. -40 °C / +85 °C"),
    (9, "Verfügbar bis", "Kennzahl 4, z. B. 2035"),
    (10, "Vergleich", "Highlights 1-4 (je ca. 55 Zeichen) für den Vergleich: bei allen "
                      "Produkten dasselbe Thema auf demselben Platz, z. B. 1 CPU, 2 KI, "
                      "3 Speicher, 4 Schnittstellen"),
    (11, "Alleinstellungsmerkmal", "Highlights 5-8 mit Häkchen -- was nur dieses Produkt "
                                   "kann, je ca. 55 Zeichen"),
    (12, "Diagramm-Datei", "Bild \"Im Vergleich\" unten links, Datei in diesem Ordner "
                           "(z. B. chart.png)"),
    (13, "Link Produktseite", "QR-Code \"Zum Produkt\" -- Produktseite auf phytec.de"),
    (14, "Link Kit", "QR-Code ganz rechts -- Entwicklungskit"),
    (14, "Kit-Bezeichnung", "Aufschrift des Kit-Knopfs über dem QR-Code (leer: \"Zum Kit\")"),
    (None, "Aussprache (Sprachsuche)", "nicht angezeigt -- wie man den Namen spricht"),
    (None, "Bemerkung", "nicht angezeigt -- nur für euch"),
]


SLOTS = {"Vergleich": 4, "Alleinstellungsmerkmal": 4}   # numbered fields, 4 places each
KEYS = {field.lower(): field for _, field, _ in FIELDS
        if field != "Artikelnummer" and field not in SLOTS}
KEYS.update({"neu (ja/nein)": "Neu", "artikelnummer": None})
# what fits on the panel (characters, roughly); render_panel.py warns above it
# (render_panel.py also measures what really fits)
LIMITS = {"Bezeichnung": 30, "Kategorie": 30, "Kurzbeschreibung": 270, "Kit-Bezeichnung": 24}


def slot_keys():
    """'Vergleich 1'..'Vergleich 4', 'Alleinstellungsmerkmal 1'..'4'."""
    return [f"{name} {i}" for name, n in SLOTS.items() for i in range(1, n + 1)]


def _slot(key):
    """'vergleich 2' -> ('Vergleich', 2); 'vergleich' -> ('Vergleich', None);
    the old 'highlight 5' -> ('Highlight', 5); anything else None."""
    name, _, num = key.partition(" ")
    if num and not num.strip().isdigit():
        return None
    for field in list(SLOTS) + ["Highlight"]:
        if name == field.lower():
            return field, int(num) if num.strip() else None
    return None


def read_info_file(path, problems=None):
    """produktinfo.txt -> the same dict as a produktinfos.csv row. Forgiving,
    since people edit it by hand: field names in any case, "Vergleich 3:"
    as well as "Vergleich:" (next free place), the old "Highlight:" (1-4 go
    to Vergleich, 5-8 to Alleinstellungsmerkmal), and a line that is no
    "Feld:" (a value wrapped by the editor) continues the field above.
    Anything odd goes to `problems` (a list) if given."""
    row, last, old_highlights = {}, None, []
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            last = None if not text else last
            continue
        key, sep, value = text.partition(":")
        key = " ".join(key.lower().split())
        slot = _slot(key) if sep else None
        field = None if slot else (KEYS.get(key, "?") if sep else "?")
        if field == "?":
            if last is None:
                if problems is not None:
                    problems.append(f"Zeile {n} unbekannt: {text[:40]}")
                continue
            row[last] = f"{row.get(last, '')} {text}".strip()
            continue
        value = value.strip()
        if slot:
            name, num = slot
            if name == "Highlight":
                num = num or len(old_highlights) + 1
                old_highlights.append(num)
                name, num = ("Vergleich", num) if num <= 4 else ("Alleinstellungsmerkmal", num - 4)
            if num is None:  # next free place
                num = next((i for i in range(1, SLOTS[name] + 1)
                            if not row.get(f"{name} {i}")), SLOTS[name] + 1)
            if not 1 <= num <= SLOTS[name]:
                if problems is not None and value:
                    problems.append(f"{name} {num}: nur {SLOTS[name]} Plätze")
                last = None
                continue
            last = f"{name} {num}"
            row[last] = value
            continue
        last = field
        if field is None:     # Artikelnummer: the folder name counts
            continue
        last = "Neu (ja/nein)" if field == "Neu" else field
        row[last] = value
    if problems is not None:
        for field, limit in LIMITS.items():
            if len(row.get(field, "")) > limit:
                problems.append(f"{field} zu lang ({len(row[field])} > {limit}): "
                                f"{row[field][:30]}…")
    return row


def from_sheet(row):
    """A produktinfos.csv row (Highlight 1-8) in the slot form."""
    row = dict(row)
    for i in range(1, 9):
        h = row.pop(f"Highlight {i}", "")
        if h:
            row[f"Vergleich {i}" if i <= 4 else f"Alleinstellungsmerkmal {i - 4}"] = h
    return row


def write_info_file(path, art, row):
    """A produktinfo.txt with every field, its place in the panel as a comment,
    pre-filled from `row` (read_info_file's form, or {})."""
    out = ["# Produktinfo für das Info-Fenster der Messe-Demo -- " + art,
           "#",
           "# Eine Angabe pro Zeile:  Feld: Wert   (Zeilen mit # sind Kommentare)",
           "# Leere Felder bleiben im Fenster einfach weg.",
           "# Wo welches Feld erscheint ([1]..[14]): dataset/products/Info-Fenster_Aufbau.pdf",
           "# Vorschau ohne Hardware:  uv run tools/render_panel.py",
           "# Aufs Board:  uv run tools/product_info.py models/<modell> und tools/deploy.sh",
           ""]
    for nr, field, where in FIELDS:
        tag = f"[{nr}] " if nr else ""
        if field == "Artikelnummer":
            out += [f"# {tag}Artikelnummer {art} -- {where.split(' -- ')[1]}", ""]
            continue
        if field != "Kit-Bezeichnung":
            out.append(f"# {tag}{where}")
        if field in SLOTS:
            out += [f"{field} {i}: {row.get(f'{field} {i}', '')}".rstrip()
                    for i in range(1, SLOTS[field] + 1)]
        elif field == "Neu":
            out.append(f"Neu: {row.get('Neu (ja/nein)', '')}".rstrip())
        else:
            out.append(f"{field}: {row.get(field, '')}".rstrip())
        if field not in ("Link Kit", "Kerne", "Größe in mm", "Temperatur"):
            out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def read_tsv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if not rows:
        return []
    head = [h.strip() for h in rows[0]]
    return [{h: (r[i].strip() if i < len(r) else "") for i, h in enumerate(head)} for r in rows[1:]]


def summarize(options, limit=12):
    """Option names worth showing: drop placeholders ("-") and options whose
    number is 0 unless they describe the base part (no number semantics are
    known for every product family, so this stays deliberately simple)."""
    lines = []
    for row in options:
        name = row.get("Option", "")
        desc = row.get("Optionsbeschreibung", "")
        number = row.get("Optionsnummer", "")
        if not name or name == "-":
            continue
        if number == "0" and lines:  # first row = base CPU option, keep it
            continue
        head = row.get("Optionskopf Beschreibung", "")
        text = name if not head or len(head) > 30 else f"{name} {head}"
        if desc and len(text) < 14 and len(desc) < 40 and desc.lower() not in text.lower() \
                and text.lower() not in desc.lower():
            text = f"{text} ({desc})"
        lines.append(" ".join(text.split()))
    return lines[:limit]


def product_list(products_dir):
    """Article numbers in products.csv, in order."""
    with open(products_dir / "products.csv", newline="", encoding="utf-8") as f:
        return [row[0].strip() for row in csv.reader(f, delimiter=";")
                if row and row[0].strip() and row[0].strip() != "Artikelnummer"
                and not row[0].startswith("#")]


def build(products_dir, labels, model_dir=None):
    names = {}
    csv_path = products_dir / "products.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter=";"):
                if row and row[0] and row[0] != "Artikelnummer":
                    names[row[0].strip()] = row[1].strip() if len(row) > 1 else ""
    sheet = {}
    sheet_path = products_dir / "produktinfos.csv"
    if not sheet_path.exists():
        sheet_path = products_dir / "_verworfen" / "produktinfos.csv"
    if sheet_path.exists():
        with open(sheet_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f, delimiter=";"):
                art = (row.get("Artikelnummer") or "").strip()
                if art:
                    sheet[art] = {k.strip(): (v or "").strip() for k, v in row.items() if k}
    out = {}
    for art in labels:
        folder = products_dir / art
        options = read_tsv(folder / "info.tsv") if (folder / "info.tsv").exists() else []
        summary_file = folder / "summary.txt"
        summary = [l.strip() for l in summary_file.read_text(encoding="utf-8").splitlines()
                   if l.strip()] if summary_file.exists() else summarize(options)
        link_file = folder / "link.txt"
        link = next((l.strip() for l in link_file.read_text(encoding="utf-8").splitlines()
                     if l.strip()), "") if link_file.exists() else ""
        out[art] = {"name": names.get(art) or art, "summary": summary, "options": options}
        info_file = folder / INFO_FILE
        row = read_info_file(info_file) if info_file.exists() else from_sheet(sheet.get(art, {}))
        # QR codes: product page and development kit (old sheets: "Link Mehr Infos")
        links = []
        if row.get("Link Produktseite"):
            links.append(["Zum Produkt", row["Link Produktseite"]])
        kit = row.get("Link Kit") or row.get("Link Mehr Infos") or link
        if kit:
            links.append([row.get("Kit-Bezeichnung") or "Zum Kit", kit])
        if links:
            out[art]["links"] = links
            out[art]["link"] = links[-1][1]
        if row.get("Kurzbeschreibung"):
            out[art]["description"] = row["Kurzbeschreibung"]
        if row.get("Bezeichnung"):
            out[art]["name"] = row["Bezeichnung"]
        compare = [row.get(f"Vergleich {i}", "") for i in range(1, 5)]
        usp = [row[f"Alleinstellungsmerkmal {i}"] for i in range(1, 5)
               if row.get(f"Alleinstellungsmerkmal {i}")]
        if any(compare) or usp:
            # highlights: the list on the single panel; compare: 4 places, kept
            # in position for the side-by-side view; usp: what only it has
            out[art]["highlights"] = [h for h in compare if h] + usp
            out[art]["compare"] = compare
            out[art]["usp"] = usp
        facts = [(label, row.get(key, "")) for key, label in
                 (("Kerne", "Kerne"), ("Größe in mm", "Größe in mm"),
                  ("Temperatur", "Temperatur"), ("Verfügbar bis", "Verfügbar bis"))]
        if any(v for _, v in facts):
            out[art]["facts"] = [[k, v] for k, v in facts if v]
        if row.get("Kategorie"):
            out[art]["category"] = row["Kategorie"]
        if row.get("Neu (ja/nein)", "").lower() in ("ja", "j", "yes", "x", "1"):
            out[art]["new"] = True
        if row.get("Aussprache (Sprachsuche)"):
            out[art]["spoken"] = row["Aussprache (Sprachsuche)"]
        chart_name = row.get("Diagramm-Datei")
        chart = (folder / chart_name) if chart_name and (folder / chart_name).exists() else \
            next((folder / f"chart{e}" for e in (".png", ".jpg", ".jpeg")
                  if (folder / f"chart{e}").exists()), None)
        if chart is not None and model_dir is not None:
            (model_dir / "charts").mkdir(exist_ok=True)
            dest = model_dir / "charts" / f"{art}{chart.suffix.lower()}"
            shutil.copyfile(chart, dest)
            out[art]["chart"] = f"charts/{dest.name}"  # relative to the model dir
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", nargs="?", help="model directory with labels.txt")
    parser.add_argument("--products", default=str(ROOT / "dataset" / "products"))
    parser.add_argument("--vorlagen", action="store_true",
                        help=f"write {INFO_FILE} into every product folder of products.csv "
                             "that has none (pre-filled from produktinfos.csv)")
    args = parser.parse_args()
    if args.vorlagen:
        products = pathlib.Path(args.products)
        sheet = {}
        if (products / "produktinfos.csv").exists():
            with open(products / "produktinfos.csv", newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    if (row.get("Artikelnummer") or "").strip():
                        sheet[row["Artikelnummer"].strip()] = {
                            k.strip(): (v or "").strip() for k, v in row.items() if k}
        for art in product_list(products):
            path = products / art / INFO_FILE
            if path.parent.is_dir() and not path.exists():
                write_info_file(path, art, from_sheet(sheet.get(art, {})))
                print(f"wrote {path.relative_to(ROOT)}")
        return
    if not args.model_dir:
        parser.error("model_dir is needed (or --vorlagen)")
    model_dir = pathlib.Path(args.model_dir)
    labels = [l.strip() for l in (model_dir / "labels.txt").read_text().splitlines() if l.strip()]
    info = build(pathlib.Path(args.products), labels, model_dir)
    (model_dir / "products.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    for art, d in info.items():
        print(f"{art}: {d['name']}\n  " + "\n  ".join(d["summary"]))


if __name__ == "__main__":
    main()
