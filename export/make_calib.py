#!/usr/bin/env python3
"""Pick calibration pictures and letterbox them exactly like the board does.

Writes one uint8 array (N, imgsz, imgsz, 3), RGB, shared by both
quantisers -- the Hailo DFC and onnx2tf -- so the two INT8 models are
calibrated on identical data and their accuracy can be compared fairly.

Pictures are spread evenly over the source directories (sorted, then every
k-th), so a burst of near-identical camera frames does not dominate.

    python export/make_calib.py dataset/images/train -o models/phyflex_n/calib_640.npy
"""

import argparse
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "board"))
from phytec_detect.postprocess import letterbox  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help="image directories")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-n", "--count", type=int, default=512,
                        help="number of pictures (Hailo recommends >= 1024 for "
                             "optimization_level 2+, 64 is the minimum)")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    files = sorted(p for src in args.sources for p in pathlib.Path(src).rglob("*")
                   if p.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"no images in {args.sources}")
    if len(files) > args.count:
        files = [files[i] for i in np.linspace(0, len(files) - 1, args.count).astype(int)]

    calib = np.empty((len(files), args.imgsz, args.imgsz, 3), dtype=np.uint8)
    for i, path in enumerate(files):
        calib[i], _ = letterbox(cv2.imread(str(path)), args.imgsz)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, calib)
    print(f"{len(files)} pictures -> {out} {calib.shape} {calib.dtype}")


if __name__ == "__main__":
    main()
