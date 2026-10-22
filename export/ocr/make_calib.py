#!/usr/bin/env python3
"""Calibration pictures for the INT8 OCR models (export/ocr/to_tflite.py).

    uv run export/ocr/make_calib.py OUT_DIR [card.png ...]

det_calib.npy  N x 320 x 512 x 3 uint8 BGR: cards as phytec_detect.ocr
               feeds them (scaled into 320 x 512, white margin)
rec_calib.npy  N x 48 x 320 x 3 uint8 BGR: single text lines, 48 high,
               grey (128 = 0 after normalisation) to the right

The given card pictures (straightened crops from the table camera) are
used turned, scaled and lit differently; the rest are drawn cards: light
and dark, names with umlauts, titles, addresses, phone numbers and mail
addresses -- in the size the camera sees them (a card ~280 px wide),
scaled up, so they are as soft as the real ones.
"""

import glob
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DET_H, DET_W, REC_H, REC_W = 320, 512, 48, 320
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FONT_FILES = sorted(glob.glob(os.path.join(ROOT, "board/phytec_detect/fonts/*.ttf"))
                    + glob.glob("/usr/share/fonts/**/DejaVuSans*.ttf", recursive=True)
                    + glob.glob("/usr/share/fonts/**/LiberationS*.ttf", recursive=True)
                    + glob.glob("/usr/share/fonts/**/NotoSans-*.ttf", recursive=True))[:24]

FIRST = ["Martin", "Khalid", "Anna", "Jürgen", "Sören", "Björn", "Ömer", "Claudia", "Hans-Peter",
         "Marie", "François", "José", "Łukasz", "Katharina", "Tim", "Özlem", "Günther", "Sophie"]
LAST = ["Podolszki", "Neumann", "Müller", "Schäfer", "Weiß", "Böhm", "Nguyen", "García",
        "Kowalski", "Jäger", "van der Berg", "Öztürk", "Schmidt", "Großmann", "Lefèvre"]
TITLE = ["Field Application Engineer", "Leitung FAE & Support", "Sales Manager", "Geschäftsführer",
         "Software Developer", "Key Account Manager", "Head of Hardware", "Dr.-Ing.", "Vertrieb"]
COMPANY = ["PHYTEC Messtechnik GmbH", "Embedded Systems AG", "Müller & Söhne KG",
           "Visionworks Ltd.", "Automatisierung GmbH", "NXP Semiconductors", "Hailo Technologies"]
STREET = ["Barcelona-Allee 1", "Hauptstraße 12a", "Am Mühlbach 7", "Rue de la Paix 3", "Ring 44"]
CITY = ["55129 Mainz, Germany", "80331 München", "10115 Berlin", "75001 Paris", "8001 Zürich"]


def _font(size):
    return ImageFont.truetype(random.choice(FONT_FILES), size)


def _person_lines():
    first, last = random.choice(FIRST), random.choice(LAST)
    mail = f"{first[0].lower()}.{last.split()[-1].lower()}@example.com"
    return [random.choice(COMPANY), f"{first} {last}", random.choice(TITLE),
            random.choice(STREET), random.choice(CITY),
            f"Tel. +49 {random.randint(100, 9999)} {random.randint(1000, 99999)}-{random.randint(0, 99)}",
            mail, "www." + random.choice(["phytec.de", "example.com", "hailo.ai"])]


def _camera_soft(img, width=280):
    """As the table camera sees it: ~280 px wide, then scaled back up."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
    noisy = np.clip(small.astype(np.int16) + np.random.randint(-6, 7, small.shape), 0, 255)
    return cv2.resize(noisy.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)


def drawn_card():
    dark = random.random() < 0.2
    bg = random.randint(20, 70) if dark else random.randint(200, 255)
    fg = random.randint(200, 255) if dark else random.randint(0, 70)
    card = Image.new("L", (850, 550), bg)
    d = ImageDraw.Draw(card)
    lines = _person_lines()
    y = random.randint(30, 80)
    x = random.randint(30, 120)
    for i, text in enumerate(lines):
        size = random.randint(34, 52) if i == 1 else random.randint(20, 32)
        d.text((x + (random.randint(0, 300) if i > 2 and random.random() < 0.3 else 0), y),
               text, fill=fg, font=_font(size))
        y += size + random.randint(8, 22)
        if y > 500:
            break
    img = cv2.cvtColor(np.array(card), cv2.COLOR_GRAY2BGR)
    return _camera_soft(img)


def to_det(card):
    h, w = card.shape[:2]
    s = min(DET_W / w, DET_H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = np.full((DET_H, DET_W, 3), 255, np.uint8)
    canvas[:nh, :nw] = cv2.resize(card, (nw, nh), interpolation=cv2.INTER_CUBIC)
    return canvas


def to_rec(line):
    out = np.full((REC_H, REC_W, 3), 128, np.uint8)
    lw = min(REC_W, max(1, int(np.ceil(REC_H * line.shape[1] / line.shape[0]))))
    out[:, :lw] = cv2.resize(line, (lw, REC_H), interpolation=cv2.INTER_CUBIC)
    return out


def drawn_line():
    text = random.choice(_person_lines())
    size = random.randint(18, 40)
    font = _font(size)
    w = int(font.getlength(text)) + 16
    dark = random.random() < 0.2
    bg, fg = (random.randint(20, 70), 230) if dark else (random.randint(200, 255), random.randint(0, 70))
    img = Image.new("L", (w, size + 14), bg)
    ImageDraw.Draw(img).text((8, 4), text, fill=fg, font=font)
    line = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)
    return _camera_soft(line, width=max(8, int(line.shape[1] * random.uniform(0.25, 0.5))))


def varied(card):
    angle = random.uniform(-4, 4)
    h, w = card.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, random.uniform(0.85, 1.0))
    out = cv2.warpAffine(card, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return np.clip(out.astype(np.float32) * random.uniform(0.8, 1.15) + random.uniform(-15, 15),
                   0, 255).astype(np.uint8)


def main():
    out_dir, cards = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    real = [cv2.imread(c) for c in cards]
    det = [to_det(varied(c)) for c in real for _ in range(24)]
    det += [to_det(drawn_card()) for _ in range(200)]
    rec = [to_rec(drawn_line()) for _ in range(400)]
    # lines of the real cards: from their det boxes (phytec_detect.ocr, ONNX)
    sys.path.insert(0, os.path.join(ROOT, "board"))
    from phytec_detect.ocr import CardReader
    model_dir = os.environ.get("OCR_MODELS")
    if model_dir:
        reader = CardReader(model_dir, cpu=True)
        for c in real:
            for _ in range(8):
                v = varied(c)
                for rect in reader.lines(v):
                    rec.append(to_rec(reader._line_image(v, rect)))
    random.shuffle(det)
    random.shuffle(rec)
    np.save(os.path.join(out_dir, "det_calib.npy"), np.stack(det))
    np.save(os.path.join(out_dir, "rec_calib.npy"), np.stack(rec))
    cv2.imwrite(os.path.join(out_dir, "det_sample.jpg"), np.vstack(det[:4]))
    cv2.imwrite(os.path.join(out_dir, "rec_sample.jpg"), np.vstack(rec[:12]))
    print(f"{len(det)} cards, {len(rec)} lines -> {out_dir}  (fonts: {len(FONT_FILES)})")


if __name__ == "__main__":
    main()
