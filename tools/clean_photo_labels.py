#!/usr/bin/env python3
"""Fix the automatic boxes of phytec-photo pictures in place (the .json next
to each photo), before build_products.py turns them into labels.

Two known faults of the boxes phytec-photo writes:

  * NXP-Ara240 (the only product with a cable): on the i.MX95 the box often
    took in the cable loop.  The box is re-fitted to the heatsink / metal
    cover (dark or bright blob, the thin cable opened away); a picture where
    no blob of the card's size is found is skipped.
  * "lifted" pictures (the product held in a hand): the box is often the
    hand, the product hidden in it.  The box is replaced by what a trained
    model finds of that product (--model, conf >= --conf); none: skipped.

The original box is kept as "rect_orig", "skip": true makes build_products
leave the picture out, "label_fix" says what was done.  Running it again
starts from rect_orig, so it can be re-run with other settings.

    uv run tools/clean_photo_labels.py --model runs/detect/products_v8_n/weights/best.pt
"""

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from synth_dataset import find_module  # noqa: E402

CABLE = "NXP-Ara240"
# the Ara240's card seen by the demo cameras: long side 185-235 px, short 55-85
CARD_LONG, CARD_SHORT = (180, 240), (55, 88)


def card_rect(gray, near):
    """minAreaRect of the card body nearest to `near` (cx, cy), or None."""
    H, W = gray.shape
    small = cv2.resize(gray, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, 61), (W, H))
    diff = bg.astype(int) - gray
    mask = ((diff > 40) | (diff < -30)).astype(np.uint8) * 255   # heatsink or metal cover
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # the cable is ~8 px wide: an opening wider than that removes it
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        rect = cv2.minAreaRect(c)
        long_, short = max(rect[1]), min(rect[1])
        if not (CARD_LONG[0] < long_ < CARD_LONG[1] and CARD_SHORT[0] < short < CARD_SHORT[1]
                and area > 0.75 * long_ * short):
            continue
        d = np.hypot(rect[0][0] - near[0], rect[0][1] - near[1])
        if best is None or d < best[0]:
            best = (d, rect)
    return None if best is None else best[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--products", default=str(ROOT / "dataset/products"))
    parser.add_argument("--model", required=True, help="weights for the lifted pictures")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    counts = {}
    for meta_path in sorted(pathlib.Path(args.products).glob("*/*/*.json")):
        photo = meta_path.with_suffix(".jpg")
        if not photo.exists():
            continue
        meta = json.loads(meta_path.read_text())
        art = meta_path.parent.parent.name
        lifted = meta.get("kind") == "lifted"
        if (art != CABLE and not lifted) or meta.get("box_user"):
            continue  # a box drawn in label_review.py stays
        img = cv2.imread(str(photo))
        orig = meta.get("rect_orig", meta.get("rect")) or find_module(img)
        orig = [list(map(float, orig[0])), list(map(float, orig[1])), float(orig[2])]
        rect, fix = None, ""
        if lifted:
            res = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            mine = [(s, b) for s, c, b in zip(res.boxes.conf.tolist(), res.boxes.cls.tolist(),
                                              res.boxes.xyxy.tolist()) if res.names[int(c)] == art]
            if mine:
                s, (x0, y0, x1, y1) = max(mine)
                rect = [[(x0 + x1) / 2, (y0 + y1) / 2], [x1 - x0, y1 - y0], 0.0]
                fix = f"lifted: model box {s:.2f}"
            else:
                fix = "lifted: product not found by the model"
        else:
            r = card_rect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), orig[0])
            if r is not None:
                rect = [list(map(float, r[0])), list(map(float, r[1])), float(r[2])]
                fix = "cable: box fitted to the card"
            else:
                fix = "cable: card not found"
        meta["rect_orig"] = orig
        meta["label_fix"] = fix
        if rect is None:
            meta["skip"] = True
        else:
            meta["rect"] = rect
            meta.pop("skip", None)
        key = (art if art == CABLE else "lifted") + (" skipped" if rect is None else " fixed")
        counts[key] = counts.get(key, 0) + 1
        if not args.dry_run:
            meta_path.write_text(json.dumps(meta, indent=1))
    for k, v in sorted(counts.items()):
        print(f"{k:24s} {v}")


if __name__ == "__main__":
    main()
