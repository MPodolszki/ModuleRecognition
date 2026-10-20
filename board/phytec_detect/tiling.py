"""Full-resolution detection by tiles.

The model sees 640x640. On the 2592x1944 VM-017 picture, 80-100 cm above
the table, a module is only a few dozen pixels wide -- shrinking the whole
frame to 640 would leave it a dozen. Instead the frame is cut into
overlapping 640x640 tiles (optionally after a `scale`, e.g. 0.5 for a
quarter of the tiles), each tile is detected on its own, and the boxes are
mapped back and merged.

If the lens shows an image circle (fisheye), tiles that lie (almost)
outside it are skipped; the circle is looked for once in the first frame,
and with a normal lens none is found and every tile is used.
"""

import cv2
import numpy as np


def find_image_circle(gray, thres=12):
    """(cx, cy, r) of the lit fisheye circle, or None if the frame is full."""
    small = cv2.resize(gray, (gray.shape[1] // 8, gray.shape[0] // 8),
                       interpolation=cv2.INTER_AREA)
    mask = (cv2.GaussianBlur(small, (5, 5), 0) > thres).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    h, w = small.shape
    if 2 * r > 1.15 * max(h, w):  # bigger than the frame: no visible circle
        return None
    return cx * 8, cy * 8, r * 8


def _starts(length, tile, overlap):
    if length <= tile:
        return [0]
    n = int(np.ceil((length - tile) / (tile * (1 - overlap)))) + 1
    return [int(round(v)) for v in np.linspace(0, length - tile, n)]


class Tiler:
    def __init__(self, shape, tile=640, overlap=0.2, scale=1.0, circle=None,
                 min_inside=0.08):
        """Tiles for frames of `shape` (h, w), in frame coordinates.

        circle (cx, cy, r) in frame pixels drops tiles with less than
        `min_inside` of their area inside it.
        """
        h, w = shape[:2]
        self.scale = scale
        sh, sw = int(round(h * scale)), int(round(w * scale))
        self.size = (sw, sh)  # scaled frame the tiles live in
        self.tiles = []  # (x0, y0, x1, y1) in scaled-frame pixels
        for y0 in _starts(sh, tile, overlap):
            for x0 in _starts(sw, tile, overlap):
                x1, y1 = min(x0 + tile, sw), min(y0 + tile, sh)
                if circle is not None and \
                        self._inside(circle, scale, x0, y0, x1, y1) < min_inside:
                    continue
                self.tiles.append((x0, y0, x1, y1))

    @staticmethod
    def _inside(circle, scale, x0, y0, x1, y1):
        cx, cy, r = (v * scale for v in circle)
        ys, xs = np.mgrid[y0:y1:16, x0:x1:16]
        return float(((xs - cx) ** 2 + (ys - cy) ** 2 <= r * r).mean())

    def crops(self, frame):
        """Yield (crop, (x0, y0)) for each tile."""
        if self.scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)
        for x0, y0, x1, y1 in self.tiles:
            yield frame[y0:y1, x0:x1], (x0, y0)


def edge_cut(dets, tile, frame_size, margin=3):
    """Boolean per detection (tile-local coords already shifted to the scaled
    frame): does the box touch a tile edge that is *inside* the frame? Such a
    box is likely a product cut in half by the tile."""
    x0, y0, x1, y1 = tile
    fw, fh = frame_size
    cut = np.zeros(len(dets), bool)
    if x0 > 0:
        cut |= dets[:, 0] <= x0 + margin
    if y0 > 0:
        cut |= dets[:, 1] <= y0 + margin
    if x1 < fw:
        cut |= dets[:, 2] >= x1 - margin
    if y1 < fh:
        cut |= dets[:, 3] >= y1 - margin
    return cut


def _overlaps(box, others):
    xx0 = np.maximum(box[0], others[:, 0])
    yy0 = np.maximum(box[1], others[:, 1])
    xx1 = np.minimum(box[2], others[:, 2])
    yy1 = np.minimum(box[3], others[:, 3])
    inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / (a + b - inter + 1e-9), inter / (np.minimum(a, b) + 1e-9)


def merge(dets, cut=None, iou_thres=0.45, contain_thres=0.6):
    """NMS over boxes from overlapping tiles, across classes: one product
    gets one box, even if two tiles (or two classes) both report it.

    `cut` marks boxes touching an inner tile edge (edge_cut): such a partial
    box is dropped whenever it overlaps a box some tile saw whole. Then
    plain IoU NMS, plus a containment test -- a partial box overlaps the
    full one far more than its IoU says."""
    if len(dets) < 2:
        return dets
    if cut is not None and cut.any() and (~cut).any():
        whole = dets[~cut]
        keep_cut = [i for i in np.flatnonzero(cut)
                    if _overlaps(dets[i], whole)[1].max() < 0.3]
        dets = np.concatenate([whole, dets[keep_cut]])
    dets = dets[np.argsort(-dets[:, 4])]
    # all pairs at once, then the same greedy pass (highest score first)
    b = dets[:, :4]
    xx0 = np.maximum(b[:, None, 0], b[None, :, 0])
    yy0 = np.maximum(b[:, None, 1], b[None, :, 1])
    xx1 = np.minimum(b[:, None, 2], b[None, :, 2])
    yy1 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / (area[:, None] + area[None, :] - inter + 1e-9)
    contained = inter / (np.minimum(area[:, None], area[None, :]) + 1e-9)
    drops = (iou > iou_thres) | (contained > contain_thres)
    keep, alive = [], np.ones(len(dets), bool)
    for i in range(len(dets)):
        if alive[i]:
            keep.append(i)
            alive[i + 1:] &= ~drops[i, i + 1:]
    return dets[keep]
