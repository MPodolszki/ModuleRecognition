#!/usr/bin/env python3
"""Info panel preview without the board: a PDF (and PNGs) of what the demo
shows when a product is lifted.

    uv run tools/render_panel.py                      # all products
    uv run tools/render_panel.py PCM-070-0F3243I.A3   # just these

Writes dataset/products/Info-Fenster_Aufbau.pdf:
  page 1     the layout: the panel with every field framed and numbered
             [1]..[13], the same numbers as in each produktinfo.txt
  page 2     the panel on the monitor, over a demo screenshot
  then       one page per product: its produktinfo.txt on the left, the
             panel it gives on the right
and the first two pages as PNG next to it. The panel is drawn by the
board's own code (board/phytec_detect/info_panel.py) at the monitor's size,
so what is in the PDF is what the demo shows.
"""

import argparse
import pathlib
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "board"))
sys.path.insert(0, str(ROOT / "tools"))

from phytec_detect import info_panel  # noqa: E402
from product_info import FIELDS, INFO_FILE, build, product_list  # noqa: E402

SCREEN_W, SCREEN_H = 1920, 1080
PANEL_W = SCREEN_W // 2          # InfoPanel fraction 0.5
FONTS = pathlib.Path(info_panel.FONTS)
MARK = (230, 90, 20)             # RGB, frames and numbers
SCREENSHOT = ROOT / "runs" / "sim" / "demo_v6.png"

# field name as marked by info_panel -> number in FIELDS
NUMBER = {field: nr for nr, field, _ in FIELDS if nr}
NUMBER["Highlight 1-8"] = NUMBER["Highlight"]


def font(size, weight="Regular"):
    return ImageFont.truetype(str(FONTS / f"Roboto-{weight}.ttf"), size)


def render(art, info):
    """(panel as PIL RGB, marks) for one product."""
    page = info_panel.Page(PANEL_W, SCREEN_H)
    info_panel._render_single(page, art, info)
    bgr = page.image()[..., :3]
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), page.marks


def badge(draw, x, y, nr, r=15):
    draw.ellipse((x - r, y - r, x + r, y + r), fill=MARK, outline=(255, 255, 255), width=2)
    text = str(nr)
    f = font(17 if nr < 10 else 14, "Bold")
    draw.text((x - draw.textlength(text, font=f) / 2, y - 11 if nr < 10 else y - 9), text, font=f, fill=(255, 255, 255))


def layout_page(panel, marks):
    """Page 1: the panel with numbered frames, the legend beside it."""
    page = Image.new("RGB", (SCREEN_W, SCREEN_H), (255, 255, 255))
    framed = panel.copy()
    draw = ImageDraw.Draw(framed)
    shown = set()
    for field, (x0, y0, x1, y1) in marks:
        nr = NUMBER.get(field)
        if nr is None:
            continue
        draw.rectangle((x0 - 6, y0 - 4, x1 + 6, y1 + 4), outline=MARK, width=3)
        if nr not in shown:
            badge(draw, x0 - 16, y0 - 12, nr)
            shown.add(nr)
    page.paste(framed, (SCREEN_W - PANEL_W, 0))
    draw = ImageDraw.Draw(page)
    x, y = 60, 50
    draw.text((x, y), "Info-Fenster: Aufbau", font=font(44, "Bold"), fill=(20, 20, 20))
    y += 64
    for line in ("Rechte Bildschirmhälfte, fährt ein, wenn ein Produkt angehoben wird,",
                 f"und nach 30 s wieder weg. Eingetragen in {INFO_FILE} im Produktordner:",
                 "dataset/products/<Artikelnummer>/" + INFO_FILE):
        draw.text((x, y), line, font=font(21), fill=(90, 90, 90))
        y += 30
    y += 14
    for nr, field, where in FIELDS:
        if nr is None:
            continue
        if field == "Kit-Bezeichnung":
            continue  # with Link Kit
        if field == "Link Kit":
            where += "; Kit-Bezeichnung = Aufschrift des Knopfs darüber"
        badge(draw, x + 17, y + 16, nr)
        label = {"Highlight": "Highlight  (bis zu 8)",
                 "Link Kit": "Link Kit  +  Kit-Bezeichnung"}.get(field, field)
        draw.text((x + 50, y + 2), label, font=font(21, "Medium"), fill=(20, 20, 20))
        lines = wrap(draw, where, PANEL_W - 160, font(17))
        for i, line in enumerate(lines):
            draw.text((x + 50, y + 29 + 21 * i), line, font=font(17), fill=(100, 100, 100))
        y += 29 + 21 * len(lines) + 12
    draw.text((x, SCREEN_H - 44), "Nicht angezeigt: Aussprache (Sprachsuche), Bemerkung. "
              "Leere Felder fallen weg.", font=font(19), fill=(100, 100, 100))
    return page


def monitor_page(panel):
    """Page 2: the panel slid in over a demo screenshot."""
    shot = cv2.imread(str(SCREENSHOT)) if SCREENSHOT.exists() else None
    if shot is None:
        shot = np.full((SCREEN_H, SCREEN_W, 3), 60, np.uint8)
    shot = cv2.resize(shot, (SCREEN_W, SCREEN_H))
    page = Image.fromarray(cv2.cvtColor(shot, cv2.COLOR_BGR2RGB))
    page.paste(panel, (SCREEN_W - PANEL_W, 0))
    draw = ImageDraw.Draw(page)
    draw.line((SCREEN_W - PANEL_W, 0, SCREEN_W - PANEL_W, SCREEN_H), fill=info_panel.PETROL,
              width=4)
    draw.rectangle((0, SCREEN_H - 56, 640, SCREEN_H), fill=(0, 0, 0))
    draw.text((24, SCREEN_H - 44), "So auf dem Monitor (1920 × 1080)", font=font(24, "Medium"),
              fill=(255, 255, 255))
    return page


def product_page(art, panel, info_file):
    """One product: its produktinfo.txt left, the panel right."""
    page = Image.new("RGB", (SCREEN_W, SCREEN_H), (246, 246, 246))
    page.paste(panel, (SCREEN_W - PANEL_W, 0))
    draw = ImageDraw.Draw(page)
    x, y = 40, 36
    draw.text((x, y), art, font=font(30, "Bold"), fill=(20, 20, 20))
    y += 44
    rel = info_file.relative_to(ROOT) if info_file.exists() else "(keine " + INFO_FILE + ")"
    draw.text((x, y), str(rel), font=font(17), fill=(110, 110, 110))
    y += 36
    if not info_file.exists():
        return page
    f_key, f_val, f_cmt = font(17, "Medium"), font(17), font(15)
    for line in info_file.read_text(encoding="utf-8-sig").splitlines()[8:]:
        if y > SCREEN_H - 30:
            break
        if not line.strip():
            y += 8
            continue
        if line.startswith("#"):
            draw.text((x, y), line.lstrip("# "), font=f_cmt, fill=(150, 150, 150))
            y += 20
            continue
        key, _, value = line.partition(":")
        draw.text((x, y), key + ":", font=f_key, fill=info_panel.PETROL)
        kx = x + draw.textlength(key + ":  ", font=f_key)
        for i, part in enumerate(wrap(draw, value.strip(), PANEL_W - 80 - (kx - x), f_val) or [""]):
            draw.text((kx, y), part, font=f_val, fill=(20, 20, 20))
            y += 23
    return page


def wrap(draw, text, width, f):
    words, lines, line = text.split(), [], ""
    for word in words:
        test = f"{line} {word}".strip()
        if draw.textlength(test, font=f) > width and line:
            lines.append(line)
            line = word
        else:
            line = test
    return lines + ([line] if line else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("articles", nargs="*", help="default: all of products.csv")
    parser.add_argument("--products", default=str(ROOT / "dataset" / "products"))
    parser.add_argument("--example", default="PCM-070-0F3243I.A3",
                        help="product for the layout and monitor pages")
    parser.add_argument("--out", default=None, help="PDF (default: <products>/Info-Fenster_Aufbau.pdf)")
    args = parser.parse_args()
    products = pathlib.Path(args.products)
    arts = args.articles or product_list(products)
    out = pathlib.Path(args.out) if args.out else products / "Info-Fenster_Aufbau.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        infos = build(products, sorted(set(arts) | {args.example}), pathlib.Path(tmp))
        for info in infos.values():  # charts: relative to the (temporary) model dir
            if info.get("chart"):
                info["chart"] = str(pathlib.Path(tmp) / info["chart"])
        example, marks = render(args.example, infos[args.example])
        pages = [layout_page(example, marks), monitor_page(example)]
        for art in arts:
            panel, _ = render(art, infos[art])
            pages.append(product_page(art, panel, products / art / INFO_FILE))
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=150)
    pages[0].save(out.with_name(out.stem + "_Seite1.png"))
    pages[1].save(out.with_name(out.stem + "_Monitor.png"))
    print(f"{out} ({len(pages)} Seiten)")


if __name__ == "__main__":
    main()
