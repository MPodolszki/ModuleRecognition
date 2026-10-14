"""Speed and accuracy of every backend on the same labelled test set.

    phytec-benchmark -d /usr/share/phytec-detect/testset
    phytec-benchmark -d testset -b hailo -b npu -j results.json

The test set is a YOLO-format directory (images/ and labels/ next to each
other, labels as `class cx cy w h`, normalised). Every backend runs every
picture once (after a few warm-up runs, which also hide the NPU's one-time
graph compilation) and is scored with ultralytics' mAP definition at
conf 0.001 -- the same numbers `yolo val` prints for the FP32 model on the
PC, so the INT8 loss on each accelerator is directly visible.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from .backends import BACKENDS, NAMES
from .detector import Detector
from .metrics import MeanAP

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_testset(root):
    img_dir = os.path.join(root, "images") if os.path.isdir(os.path.join(root, "images")) else root
    lbl_dir = img_dir.replace(os.sep + "images", os.sep + "labels") \
        if os.sep + "images" in img_dir else os.path.join(root, "labels")
    items = []
    for dirpath, _, files in os.walk(img_dir):
        for name in sorted(files):
            if not name.lower().endswith(EXTS):
                continue
            img = os.path.join(dirpath, name)
            rel = os.path.relpath(img, img_dir)
            lbl = os.path.join(lbl_dir, os.path.splitext(rel)[0] + ".txt")
            rows = np.loadtxt(lbl, ndmin=2) if os.path.exists(lbl) and os.path.getsize(lbl) \
                else np.zeros((0, 5))
            items.append((img, rows[:, :5]))
    return items


def to_xyxy(rows, shape):
    h, w = shape[:2]
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def bench(name, model_dir, items, warmup, hef=None):
    det = Detector(name, model_dir, conf=0.001, max_det=300, hef=hef)
    metric = MeanAP(det.nc)
    first = cv2.imread(items[0][0])
    t0 = time.perf_counter()
    for _ in range(warmup):
        det(first)
    warm_s = time.perf_counter() - t0
    for stage in det.times.values():
        stage.clear()

    pre, infer, post = [], [], []
    for path, rows in items:
        img = cv2.imread(path)
        dets = det(img)
        pre.append(det.times["pre"][-1])
        infer.append(det.times["infer"][-1])
        post.append(det.times["post"][-1])
        metric.add(dets, to_xyxy(rows, img.shape), rows[:, 0])
    det.close()

    ms = lambda v: 1000 * float(np.mean(v))
    total = np.add(np.add(pre, infer), post)
    return {"backend": name, "model": det.paths, "load_s": det.load_s,
            "warmup_s": warm_s, "pre_ms": ms(pre), "infer_ms": ms(infer),
            "infer_p95_ms": 1000 * float(np.percentile(infer, 95)),
            "post_ms": ms(post), "fps_infer": 1000 / ms(infer),
            "fps_total": 1 / float(np.mean(total)), **metric.summary()}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phytec-benchmark", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--data", default="/usr/share/phytec-detect/testset")
    parser.add_argument("-m", "--model-dir", default="/usr/share/phytec-detect/model")
    parser.add_argument("-b", "--backend", action="append", choices=BACKENDS,
                        help="repeatable (default: hailo, npu, cpu, fp32)")
    parser.add_argument("--hef", help="explicit HEF for the hailo backend")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("-n", "--limit", type=int, help="only the first N pictures")
    parser.add_argument("-j", "--json", help="also write the results here")
    args = parser.parse_args(argv)

    items = load_testset(args.data)
    if args.limit:
        items = items[:args.limit]
    if not items:
        print(f"no pictures under {args.data}", file=sys.stderr)
        return 2
    n_gt = sum(len(rows) for _, rows in items)
    print(f"test set: {args.data}: {len(items)} pictures, {n_gt} labelled objects\n")

    results = []
    for name in args.backend or list(BACKENDS):
        print(f"{NAMES[name]} ...", flush=True)
        try:
            results.append(bench(name, args.model_dir, items, args.warmup, args.hef))
        except Exception as error:
            print(f"  skipped: {type(error).__name__}: {error}", flush=True)

    head = (f"{'backend':22s} {'infer ms':>9s} {'p95':>6s} {'pre+post':>9s} "
            f"{'fps*':>6s} {'mAP50':>6s} {'50-95':>6s} {'P':>5s} {'R':>5s}")
    print("\n" + head + "\n" + "-" * len(head))
    for r in results:
        print(f"{NAMES[r['backend']]:22s} {r['infer_ms']:9.1f} {r['infer_p95_ms']:6.1f} "
              f"{r['pre_ms'] + r['post_ms']:9.1f} {r['fps_total']:6.1f} "
              f"{r['mAP50']:6.3f} {r['mAP50-95']:6.3f} {r['P@0.25']:5.2f} {r['R@0.25']:5.2f}")
    print("* fps = one picture after the other incl. pre/post on the CPU (no pipelining)")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"data": args.data, "pictures": len(items), "objects": n_gt,
                       "results": results}, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
