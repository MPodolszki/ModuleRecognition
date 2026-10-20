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

produktinfos.csv (one row per product, filled in by hand) comes first:
its highlights, key figures (cores, size, temperature, available until),
category, "new" flag, link and chart file override the per-folder files.

    uv run tools/product_info.py models/products_v1_n
"""

import argparse
import csv
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]


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
        row = sheet.get(art, {})
        link = row.get("Link Mehr Infos") or link
        if link:
            out[art]["link"] = link
        if row.get("Bezeichnung"):
            out[art]["name"] = row["Bezeichnung"]
        highlights = [row[k] for k in sorted(row) if k.startswith("Highlight") and row[k]]
        if highlights:
            out[art]["highlights"] = highlights
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
    parser.add_argument("model_dir", help="model directory with labels.txt")
    parser.add_argument("--products", default=str(ROOT / "dataset" / "products"))
    args = parser.parse_args()
    model_dir = pathlib.Path(args.model_dir)
    labels = [l.strip() for l in (model_dir / "labels.txt").read_text().splitlines() if l.strip()]
    info = build(pathlib.Path(args.products), labels, model_dir)
    (model_dir / "products.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    for art, d in info.items():
        print(f"{art}: {d['name']}\n  " + "\n  ".join(d["summary"]))


if __name__ == "__main__":
    main()
