"""Mapping from the detection camera (mono) to the live camera (colour)
for points on the table.

Both cameras look down at the same flat table, so one 2-D -> 2-D mapping
is enough. With ordinary (non-fisheye) lenses that is a homography, which
also holds outside the area the calibration target covered. Should the
detection lens bend straight lines noticeably, a cubic polynomial in both
coordinates (10 coefficients per axis) absorbs the distortion too -- but
is only trustworthy inside the covered area. fit() takes the homography
unless the polynomial is clearly better.

The point pairs are ArUco marker corners both cameras see
(phytec-calibrate).
"""

import json

import cv2
import numpy as np

DEGREE = 3


def _terms(xy, norm):
    """Polynomial terms of normalised points: 1, x, y, x^2, xy, y^2, ..."""
    cx, cy, s = norm
    x = (xy[:, 0] - cx) / s
    y = (xy[:, 1] - cy) / s
    return np.stack([x ** (d - j) * y ** j for d in range(DEGREE + 1) for j in range(d + 1)],
                    axis=1)


class PlaneMap:
    def __init__(self, kind, params, src_shape, dst_shape, hull, rms, norm=None,
                 view_crop=None):
        self.kind = kind                                  # "homography" | "poly3"
        self.params = np.asarray(params, np.float64)      # 3x3, or (terms, 2)
        self.norm = tuple(norm) if norm is not None else None
        self.src_shape, self.dst_shape = tuple(src_shape), tuple(dst_shape)
        self.hull = np.asarray(hull, np.float32)          # covered area, src pixels
        self.rms = rms
        # Both sides are in *sensor* pixels (independent of crop/binning);
        # view_crop is the live camera's sensor window (x, y, w, h) showing
        # what the detection camera sees.
        self.view_crop = tuple(view_crop) if view_crop else None

    @classmethod
    def fit(cls, src, dst, src_shape, dst_shape, kind="auto"):
        """Returns (PlaneMap, number of rejected point pairs)."""
        src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
        if len(src) < 8:
            raise ValueError(f"need >= 8 point pairs, got {len(src)}")
        hull = cv2.convexHull(src.astype(np.float32)).reshape(-1, 2)

        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        good = inl.ravel().astype(bool)
        h_rms = _rms(cv2.perspectiveTransform(src[good, None], H)[:, 0], dst[good])
        homography = cls("homography", H, src_shape[:2], dst_shape[:2], hull, h_rms)
        if kind == "homography" or (kind == "auto" and (h_rms < 3.0 or len(src) < 40)):
            return homography, int((~good).sum())

        norm = (src_shape[1] / 2, src_shape[0] / 2, max(src_shape[:2]) / 2)
        A = _terms(src, norm)
        coef = np.linalg.lstsq(A, dst, rcond=None)[0]
        # One round of outlier rejection (a misdetected corner would bend
        # the whole polynomial).
        err = np.linalg.norm(A @ coef - dst, axis=1)
        pgood = err < max(3 * np.median(err), 2.0)
        coef = np.linalg.lstsq(A[pgood], dst[pgood], rcond=None)[0]
        p_rms = _rms(A[pgood] @ coef, dst[pgood])
        if kind == "poly3" or p_rms < 0.7 * h_rms:
            return cls("poly3", coef, src_shape[:2], dst_shape[:2], hull, p_rms, norm), \
                int((~pgood).sum())
        return homography, int((~good).sum())

    def __call__(self, xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        if self.kind == "homography":
            return cv2.perspectiveTransform(xy[:, None], self.params)[:, 0]
        return _terms(xy, self.norm) @ self.params

    def boxes(self, dets):
        """Map detection boxes: the box outline (corners and edge midpoints,
        which a residual lens distortion may bend) goes through the mapping;
        the result is the bounding box of the mapped outline."""
        if not len(dets):
            return dets
        out = dets.copy()
        t = np.linspace(0, 1, 5)
        x0, y0, x1, y1 = (dets[:, k:k + 1].astype(np.float64) for k in range(4))
        xs, ys = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t  # (N, 5) each
        n = len(dets)
        outline = np.concatenate([np.stack([xs, np.broadcast_to(y0, xs.shape)], 2),
                                  np.stack([xs, np.broadcast_to(y1, xs.shape)], 2),
                                  np.stack([np.broadcast_to(x0, ys.shape), ys], 2),
                                  np.stack([np.broadcast_to(x1, ys.shape), ys], 2)], 1)
        m = self(outline.reshape(-1, 2)).reshape(n, 20, 2)  # all boxes in one call
        out[:, 0], out[:, 1] = m[..., 0].min(1), m[..., 1].min(1)
        out[:, 2], out[:, 3] = m[..., 0].max(1), m[..., 1].max(1)
        return out

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"kind": self.kind, "params": self.params.tolist(), "norm": self.norm,
                       "coords": "sensor", "src_shape": self.src_shape,
                       "dst_shape": self.dst_shape, "hull": self.hull.tolist(),
                       "rms_px": self.rms, "view_crop": self.view_crop}, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["kind"], d["params"], d["src_shape"], d["dst_shape"], d["hull"],
                   d["rms_px"], d.get("norm"), d.get("view_crop"))


def frame_to_sensor(xy, cam):
    """Frame pixels of a Camera -> sensor pixels (undo crop and binning)."""
    x0, y0 = cam.crop[:2] if cam.crop else (0, 0)
    return np.asarray(xy, np.float64) / cam.scale + [x0, y0]


def sensor_to_frame(xy, cam):
    x0, y0 = cam.crop[:2] if cam.crop else (0, 0)
    return (np.asarray(xy, np.float64) - [x0, y0]) * cam.scale


def view_crop_for(plane, det_window, sensor_size, aspect=16 / 9, margin=0.02):
    """Live-camera sensor window (x, y, w, h) covering the detection
    camera's window (x, y, w, h, in its sensor pixels), widened to `aspect`."""
    x, y, w, h = det_window
    t = np.linspace(0, 1, 9)
    outline = np.concatenate([np.stack([x + w * t, np.full(9, y)], 1),
                              np.stack([x + w * t, np.full(9, y + h)], 1),
                              np.stack([np.full(9, x), y + h * t], 1),
                              np.stack([np.full(9, x + w), y + h * t], 1)])
    m = plane(outline)
    x0, y0 = m.min(0)
    x1, y1 = m.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cw, ch = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    if cw / ch < aspect:
        cw = ch * aspect
    else:
        ch = cw / aspect
    sw, sh = sensor_size
    cw, ch = min(cw, sw), min(ch, sh)
    cx = min(max(cx, cw / 2), sw - cw / 2)
    cy = min(max(cy, ch / 2), sh - ch / 2)
    return [int(cx - cw / 2) & ~1, int(cy - ch / 2) & ~1, int(cw) & ~1, int(ch) & ~1]


def _rms(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
