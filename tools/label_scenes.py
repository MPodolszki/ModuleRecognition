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

Laid out the wrong way round? For two-product scenes the sizes are checked
against each product's reference shot (phytec-photo): if most pictures of a
scene have the bigger product where the smaller one belongs, the whole
scene was laid out reversed and its two names are swapped.
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
    products = pathlib.Path(args.scenes).parent

    def ref_size(article):
        for meta in (products / article).glob("vm017_isp/*.json"):
            m = json.loads(meta.read_text(encoding="utf-8"))
            if m.get("kind") == "reference" and m.get("rect"):
                w, h = m["rect"][1]
                return (w * h) ** 0.5
        return None
    used = skipped = 0
    written = {}  # scene -> [(path, [[article, box...], ...])]
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
            labelled = [[a, *s[0]] for a, s in zip(articles, stripes)]
            out.write_text(json.dumps(labelled))
            written.setdefault("+".join(articles), []).append((out, labelled))
            used += 1
        else:
            out.unlink(missing_ok=True)
            skipped += 1
            print(f"skip {jpg.name}: products per stripe {[len(s) for s in stripes]}")
    for scene, items in written.items():
        arts = scene.split("+")
        if len(arts) != 2:
            continue
        ref = [ref_size(a) for a in arts]
        if None in ref or abs(ref[0] - ref[1]) < 0.1 * max(ref):
            continue  # too alike in size to tell
        def size(b):
            return ((b[3] - b[1]) * (b[4] - b[2])) ** 0.5
        wrong = sum((size(lab[0]) > size(lab[1])) != (ref[0] > ref[1]) for _, lab in items)
        if wrong > len(items) / 2:
            print(f"{scene}: laid out reversed in {wrong}/{len(items)} pictures -- names swapped")
            for path, lab in items:
                path.write_text(json.dumps([[lab[1][0], *lab[0][1:]], [lab[0][0], *lab[1][1:]]]))
    print(f"{used} scene pictures labelled, {skipped} skipped")


if __name__ == "__main__":
    main()
