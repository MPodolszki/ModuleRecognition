#!/usr/bin/env python3
"""Pictures as the i.MX8MP ISP delivers them at 960x544 (phytec-detect
--isp-width 960), made from the ISP's 1920x1080 pictures the photo program
takes -- so a model can be trained for the faster ISP-scaled input.

The ISP's scaled output is not just the 1920x1080 picture halved: it frames
a slightly different window, bent differently near the edges (another lens
correction), and it is softer. Measured on the Hailo board (same table,
both outputs, SIFT pairs): a 3rd-order polynomial maps between the two
(median residual 0.44 px of the 1920 picture), and a Gaussian blur of
sigma ~0.9 before resampling matches the ISP's softness (mean difference
1.9 grey levels, Laplacian variance 145 vs 140).

    cd board && python3 -m phytec_detect.isp960 fit ../runs/ispcmp   # full_*.raw / s960_*.raw
    from phytec_detect.isp960 import to_isp960, boxes_to_isp960       # -> isp960.json here
"""

import functools
import json
import pathlib
import sys

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PARAMS = HERE / "isp960.json"
FULL, SMALL = (1920, 1080), (960, 544)


def _terms(p, cx, cy, s):
    x, y = (p[:, 0] - cx) / s, (p[:, 1] - cy) / s
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y,
                     x ** 3, x * x * y, x * y * y, y ** 3], 1)


@functools.lru_cache(maxsize=1)
def _params():
    p = json.loads(PARAMS.read_text())
    gx, gy = np.meshgrid(np.arange(SMALL[0], dtype=np.float64),
                         np.arange(SMALL[1], dtype=np.float64))
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    src = (_terms(grid, 480, 272, 480) @ np.array(p["small_to_full"])).reshape(
        SMALL[1], SMALL[0], 2).astype(np.float32)
    return p, src


def to_isp960(img):
    """1920x1080 picture (gray or BGR) -> 960x544 as the ISP scales it;
    None for any other size (not an ISP picture of that mode)."""
    if img.shape[1] != FULL[0] or img.shape[0] != FULL[1]:
        return None
    p, src = _params()
    blurred = cv2.GaussianBlur(img, (0, 0), p["sigma"])
    return cv2.remap(blurred, src[..., 0], src[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def points_to_isp960(xy):
    """1920x1080 picture points -> 960x544: the measured small -> full
    polynomial solved for the small point (Newton, from the plain halving)."""
    p, _ = _params()
    coef = np.array(p["small_to_full"])
    target = np.asarray(xy, np.float64).reshape(-1, 2)
    q = target / 2.0
    for _ in range(12):
        f = _terms(q, 480, 272, 480) @ coef - target
        h = 0.5
        jx = (_terms(q + [h, 0], 480, 272, 480) @ coef - _terms(q - [h, 0], 480, 272, 480) @ coef) / (2 * h)
        jy = (_terms(q + [0, h], 480, 272, 480) @ coef - _terms(q - [0, h], 480, 272, 480) @ coef) / (2 * h)
        det = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
        dq = np.stack([(jy[:, 1] * f[:, 0] - jy[:, 0] * f[:, 1]) / det,
                       (-jx[:, 1] * f[:, 0] + jx[:, 0] * f[:, 1]) / det], 1)
        q = q - dq
        if np.abs(dq).max() < 1e-4:
            break
    return q


def boxes_to_isp960(boxes):
    """[(cls, x0, y0, x1, y1)] in 1920x1080 pixels -> the same in the
    960x544 picture: the bounding box of the 960 pixels that show a point of
    the box (forward map only -- exact up to the edges of the 960 view,
    where the way back has no solution). Boxes not in view are dropped."""
    _, src = _params()
    grid = src[::2, ::2]  # every 2nd 960 pixel is plenty for a box
    out = []
    for cid, x0, y0, x1, y1 in boxes:
        inside = (grid[..., 0] >= x0) & (grid[..., 0] <= x1) & \
            (grid[..., 1] >= y0) & (grid[..., 1] <= y1)
        ys, xs = np.nonzero(inside)
        if len(xs) < 4:
            continue
        a0, a1 = xs.min() * 2.0, min(xs.max() * 2.0 + 2, SMALL[0] - 1)
        b0, b1 = ys.min() * 2.0, min(ys.max() * 2.0 + 2, SMALL[1] - 1)
        if a1 - a0 > 4 and b1 - b0 > 4:
            out.append((cid, float(a0), float(b0), float(a1), float(b1)))
    return out


def fit(folder):
    """Fit both directions from pairs of raw captures of the same still
    scene: full_<k>_<n>.raw (GRAY8 1920x1080) and s960_<k>_<n>.raw (NV12
    960x544, Y plane first)."""
    folder = pathlib.Path(folder)
    sift = cv2.SIFT_create(12000)
    small_pts, full_pts, pairs = [], [], []
    for f in sorted(folder.glob("full_*.raw")):
        s = folder / f.name.replace("full_", "s960_")
        if not s.exists():
            continue
        a = np.fromfile(f, np.uint8).reshape(FULL[1], FULL[0])
        b = np.fromfile(s, np.uint8)[:SMALL[0] * SMALL[1]].reshape(SMALL[1], SMALL[0])
        pairs.append((a, b))
        ka, da = sift.detectAndCompute(cv2.resize(a, (960, 540), interpolation=cv2.INTER_AREA), None)
        kb, db = sift.detectAndCompute(b, None)
        for m1, m2 in cv2.BFMatcher().knnMatch(db, da, k=2):
            if m1.distance < 0.75 * m2.distance:
                small_pts.append(kb[m1.queryIdx].pt)
                full_pts.append(np.array(ka[m1.trainIdx].pt) * 2.0)
    small_pts, full_pts = np.float64(small_pts), np.float64(full_pts)

    def robust(src, dst, cx, cy, s):
        good = np.ones(len(src), bool)
        for _ in range(6):
            coef = np.linalg.lstsq(_terms(src[good], cx, cy, s), dst[good], rcond=None)[0]
            err = np.linalg.norm(_terms(src, cx, cy, s) @ coef - dst, axis=1)
            good = err < max(3 * np.median(err[good]), 1.0)
        return coef, err[good], good
    s2f, err, good = robust(small_pts, full_pts, 480, 272, 480)
    params = {"small_to_full": s2f.tolist(), "sigma": 0.9,
              "pairs": int(good.sum()), "residual_full_px_median": float(np.median(err))}
    PARAMS.write_text(json.dumps(params, indent=1))
    _params.cache_clear()
    best = None
    for sigma in (0.6, 0.75, 0.9, 1.05, 1.2):
        params["sigma"] = sigma
        PARAMS.write_text(json.dumps(params, indent=1))
        _params.cache_clear()
        e = np.mean([np.abs(to_isp960(a).astype(np.float32) - b)[10:-10, 10:-10].mean() for a, b in pairs])
        if best is None or e < best[1]:
            best = (sigma, e)
    params["sigma"], params["mean_abs_diff"] = best[0], float(best[1])
    PARAMS.write_text(json.dumps(params, indent=1))
    _params.cache_clear()
    print(f"{params['pairs']} point pairs, residual {params['residual_full_px_median']:.2f} px, "
          f"sigma {best[0]}, mean |diff| {best[1]:.2f} grey levels -> {PARAMS}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "fit":
        fit(sys.argv[2])
    else:
        sys.exit(__doc__)
