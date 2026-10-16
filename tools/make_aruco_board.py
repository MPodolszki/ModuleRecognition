#!/usr/bin/env python3
"""Printable ArUco target for phytec-calibrate (DICT_5X5_100).

    uv run tools/make_aruco_board.py -o calib_target          # A4 + A3 PDFs

Print at 100 % (no "fit to page"), lay it flat on the table where the
modules will lie, so both cameras see it. Larger is better: the mapping is
only valid inside the area the markers cover.
"""

import argparse

import cv2
import numpy as np
from PIL import Image

PAPER_MM = {"A4": (297, 210), "A3": (420, 297)}  # landscape
DPI = 300


def board(paper, marker_mm, gap_mm, margin_mm=12):
    w_mm, h_mm = PAPER_MM[paper]
    px = lambda mm: int(round(mm / 25.4 * DPI))
    img = np.full((px(h_mm), px(w_mm)), 255, np.uint8)
    cols = int((w_mm - 2 * margin_mm + gap_mm) // (marker_mm + gap_mm))
    rows = int((h_mm - 2 * margin_mm + gap_mm) // (marker_mm + gap_mm))
    x_off = (w_mm - cols * marker_mm - (cols - 1) * gap_mm) / 2
    y_off = (h_mm - rows * marker_mm - (rows - 1) * gap_mm) / 2
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    ids = 0
    for r in range(rows):
        for c in range(cols):
            m = cv2.aruco.generateImageMarker(dictionary, ids, px(marker_mm))
            y, x = px(y_off + r * (marker_mm + gap_mm)), px(x_off + c * (marker_mm + gap_mm))
            img[y:y + m.shape[0], x:x + m.shape[1]] = m
            ids += 1
    cv2.putText(img, f"phytec-calibrate  DICT_5X5_100  {marker_mm} mm  print at 100 %",
                (px(margin_mm), px(h_mm - 4)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    return img, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", default="calib_target")
    parser.add_argument("--marker-mm", type=float, default=40)
    parser.add_argument("--gap-mm", type=float, default=12)
    args = parser.parse_args()
    for paper in PAPER_MM:
        img, n = board(paper, args.marker_mm, args.gap_mm)
        path = f"{args.out}_{paper}.pdf"
        Image.fromarray(img).save(path, resolution=DPI)
        print(f"{path}: {n} markers of {args.marker_mm:g} mm")


if __name__ == "__main__":
    main()
