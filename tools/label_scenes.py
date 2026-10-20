#!/usr/bin/env python3
"""Boxes for the phytec-scenes pictures: the float model finds the products,
their stripe on the table names them.

    uv run tools/label_scenes.py runs/detect/products_v4_n/weights/best.pt

Reads dataset/products/_scenes/<scene>_<n>.jpg/.json (fetched from the
board), writes <scene>_<n>.boxes.json next to them: [[article, x0, y0, x1,
y1], ...] in picture pixels. A picture is used only if the model finds
exactly one product in each stripe -- otherwise (product on the stripe
line, one missed, a phone taken for a product) it is skipped and listed.
The class the model gives is ignored on purpose: telling these look-alikes
apart is what the scenes are for.
"""

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights")
    parser.add_argument("--scenes", default=str(ROOT / "dataset" / "products" / "_scenes"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    used = skipped = 0
    for meta_path in sorted(pathlib.Path(args.scenes).glob("*.json")):
        if meta_path.name.endswith(".boxes.json"):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        articles = meta["articles"]
        jpg = meta_path.with_suffix(".jpg")
        r = model.predict(str(jpg), conf=args.conf, imgsz=args.imgsz, agnostic_nms=True,
                          verbose=False)[0]
        w = r.orig_shape[1]
        boxes = [[float(v) for v in b] for b in r.boxes.xyxy.tolist()]
        stripes = [[] for _ in articles]
        for b in boxes:
            cx = (b[0] + b[2]) / 2
            stripes[min(len(articles) - 1, int(cx / w * len(articles)))].append(b)
        out = meta_path.with_name(meta_path.stem + ".boxes.json")
        if all(len(s) == 1 for s in stripes):
            out.write_text(json.dumps([[a, *s[0]] for a, s in zip(articles, stripes)]))
            used += 1
        else:
            out.unlink(missing_ok=True)
            skipped += 1
            print(f"skip {jpg.name}: products per stripe {[len(s) for s in stripes]}")
    print(f"{used} scene pictures labelled, {skipped} skipped")


if __name__ == "__main__":
    main()
