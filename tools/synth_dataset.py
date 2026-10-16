#!/usr/bin/env python3
"""Synthetic bootstrap data: paste cut-out modules onto backgrounds.

Meant to carry the model until enough real phyCAM pictures are labelled
(phytec-capture + Label Studio). Each source photo shows one module on a
plain, light background; the module is found automatically (darker or more
saturated than the large-scale background, most rectangle-like blob not
touching the border) and pasted with random scale, rotation, perspective,
colour, blur and noise -- 0..3 times per background, plus pure-background
negatives -- labelled with the box of the pasted quad.

v2 (after the first board test detected only close-up modules):
  * module size log-uniform from 3 % of the picture, so distant modules exist
  * module photos chosen by resolution, so big pastes are not upscaled blur
  * real phyCAM frames as backgrounds (--phycam), white-balanced like the
    board's Camera does, sometimes left raw-green
  * colour-cast augmentation of the whole picture
  * the module photos themselves (and zoomed crops) as real training pictures

    uv run tools/synth_dataset.py --module dataset/raw/phone/*.jpg \\
        --phycam dataset/raw/phycam_bg \\
        --backgrounds datasets_ext/coco128/images/train2017 -o dataset_synth_v2 -n 4000
"""

import argparse
import math
import pathlib
import random
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "board"))
from phytec_detect.camera import white_balance  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_module(img):
    """minAreaRect ((cx, cy), (w, h), angle) of the module in a photo."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    background = cv2.resize(cv2.medianBlur(small, 61), (W, H), interpolation=cv2.INTER_LINEAR)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    # Saturation relative to the surroundings: under tinted light the whole
    # background is saturated (a blue-lit photo was boxed as one big blob).
    sat_small = cv2.resize(sat, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    sat_bg = cv2.resize(cv2.medianBlur(sat_small, 61), (W, H), interpolation=cv2.INTER_LINEAR)
    mask = (((background.astype(int) - gray) > 35) |
            ((sat.astype(int) - sat_bg) > 60)).astype(np.uint8) * 255
    k = max(5, int(min(H, W) * 0.01)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.001 * H * W or area > 0.9 * H * W:
            continue
        x, y, w, h = cv2.boundingRect(c)
        rect = cv2.minAreaRect(c)
        fill = area / (rect[1][0] * rect[1][1] + 1e-9)
        aspect = max(rect[1]) / (min(rect[1]) + 1e-9)
        border = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
        score = fill - 0.5 * border - 0.3 * (aspect > 1.6)
        if best is None or score > best[0]:
            best = (score, rect)
    if best is None:
        raise ValueError("no module found")
    return best[1]


def cut_module(img, rect):
    """Upright crop of the module, slightly inset to drop background pixels."""
    (cx, cy), (w, h), angle = rect
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    upright = cv2.warpAffine(img, rot, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    inset = 0.015
    x0, x1 = int(cx - w / 2 * (1 - inset)), int(cx + w / 2 * (1 - inset))
    y0, y1 = int(cy - h / 2 * (1 - inset)), int(cy + h / 2 * (1 - inset))
    return upright[max(y0, 0):y1, max(x0, 0):x1].copy()


def rect_box(rect, shape):
    pts = cv2.boxPoints(rect)
    h, w = shape[:2]
    x0, y0 = np.clip(pts.min(0), 0, [w, h])
    x1, y1 = np.clip(pts.max(0), 0, [w, h])
    return np.array([x0, y0, x1, y1])


def random_background(phycam, others, size):
    roll = random.random()
    if phycam and roll < 0.5:
        bg = cv2.imread(str(random.choice(phycam)))
        if random.random() < 0.7:
            bg = white_balance(bg)
    elif others and roll < 0.85:
        bg = cv2.imread(str(random.choice(others)))
    else:  # plain desk / mat / light-box colours with a little texture
        color = np.array(random.choice([(200, 210, 220), (60, 60, 60), (230, 230, 230),
                                        (40, 90, 40), (150, 120, 90), (30, 30, 120),
                                        (205, 205, 200)]))
        color = np.clip(color + np.random.randint(-30, 30, 3), 0, 255)
        bg = np.full((size, size, 3), color, np.uint8)
        noise = cv2.GaussianBlur(np.random.randn(size, size).astype(np.float32) * 25,
                                 (0, 0), random.uniform(1, 8))
        bg = np.clip(bg + noise[..., None], 0, 255).astype(np.uint8)
    scale = size / max(bg.shape[:2])
    return cv2.resize(bg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def color_jitter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-6, 6)) % 180
    hsv[..., 1] *= random.uniform(0.6, 1.4)
    hsv[..., 2] *= random.uniform(0.5, 1.4)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    gamma = random.uniform(0.7, 1.4)
    return np.clip(255 * (out / 255.0) ** gamma, 0, 255).astype(np.uint8)


def color_cast(img):
    """Whole-picture cast, mostly the raw sensor's green, sometimes dark."""
    gains = np.array([random.uniform(0.6, 1.0), random.uniform(1.0, 1.5),
                      random.uniform(0.6, 1.0)]) if random.random() < 0.7 else \
        np.random.uniform(0.75, 1.25, 3)
    gains *= random.uniform(0.5, 1.1)
    return np.clip(img * gains, 0, 255).astype(np.uint8)


def pick_module(modules, target):
    """The lowest-resolution photo that still does not need upscaling > 1.5x."""
    ok = [m for m in modules if max(m.shape[:2]) * 1.5 >= target]
    return random.choice(ok) if ok else max(modules, key=lambda m: max(m.shape[:2]))


def paste(bg, modules, occupied, min_frac, max_frac):
    """Warp a module into `bg` at a random pose. Returns the box or None."""
    bh, bw = bg.shape[:2]
    target = math.exp(random.uniform(math.log(min_frac), math.log(max_frac))) * min(bh, bw)
    module = pick_module(modules, target)
    mh, mw = module.shape[:2]
    s = target / max(mh, mw)
    angle = random.choice([0, 90, 180, 270]) + random.uniform(-25, 25) \
        if random.random() < 0.6 else random.uniform(0, 360)
    corners = np.array([[0, 0], [mw, 0], [mw, mh], [0, mh]], np.float32) - [mw / 2, mh / 2]
    # Perspective: shrink one side a little, as from a tilted camera.
    tilt = random.uniform(0, 0.25)
    side = random.randrange(4)
    corners[side] *= 1 - tilt
    corners[(side + 1) % 4] *= 1 - tilt
    corners += np.random.uniform(-0.03, 0.03, corners.shape) * [mw, mh]
    rad = np.deg2rad(angle)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    quad = (corners * s) @ rot.T
    x_min, y_min = quad.min(0)
    x_max, y_max = quad.max(0)
    if x_max - x_min >= bw or y_max - y_min >= bh:
        return None
    for _ in range(20):
        cx = random.uniform(-x_min, bw - x_max)
        cy = random.uniform(-y_min, bh - y_max)
        box = np.array([cx + x_min, cy + y_min, cx + x_max, cy + y_max])
        if all(_overlap(box, o) < 0.15 for o in occupied):
            break
    else:
        return None
    dst = (quad + [cx, cy]).astype(np.float32)
    src = np.array([[0, 0], [mw, 0], [mw, mh], [0, mh]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(color_jitter(module), matrix, (bw, bh),
                                 flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((mh, mw), 255, np.uint8), matrix, (bw, bh))
    mask = cv2.GaussianBlur(mask, (3, 3), 0)[..., None] / 255.0
    # A soft drop shadow so the board does not look cut out.
    if random.random() < 0.6:
        off = np.random.randint(2, 12, 2)
        shadow = np.roll(mask[..., 0], tuple(off), axis=(0, 1))
        shadow = cv2.GaussianBlur(shadow, (0, 0), 6)[..., None] * random.uniform(0.2, 0.5)
        bg[:] = (bg * (1 - shadow)).astype(np.uint8)
    bg[:] = (warped * mask + bg * (1 - mask)).astype(np.uint8)
    return box


def _overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def degrade(img):
    if random.random() < 0.25:
        img = color_cast(img)
    if random.random() < 0.4:
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 2.0))
    if random.random() < 0.2:  # motion blur
        k = random.randrange(5, 15)
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2] = 1 / k
        rot = cv2.getRotationMatrix2D((k / 2, k / 2), random.uniform(0, 180), 1)
        img = cv2.filter2D(img, -1, cv2.warpAffine(kernel, rot, (k, k)))
    if random.random() < 0.5:
        img = np.clip(img + np.random.randn(*img.shape) * random.uniform(2, 10), 0, 255
                      ).astype(np.uint8)
    quality = random.randint(50, 95)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], 1)


def write(out, split, name, img, boxes):
    h, w = img.shape[:2]
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img)
    with open(out / "labels" / split / f"{name}.txt", "w") as f:
        for x0, y0, x1, y1 in boxes:
            x0, x1 = np.clip([x0, x1], 0, w)
            y0, y1 = np.clip([y0, y1], 0, h)
            f.write(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                    f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")


def real_variants(img, box, n, size):
    """The photo itself plus n random crops around the module (zoom levels)."""
    H, W = img.shape[:2]
    scale = size / max(H, W)
    yield cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), box * scale
    bw, bh = box[2] - box[0], box[3] - box[1]
    for _ in range(n):
        side = min(min(H, W), max(bw, bh) * random.uniform(1.6, 8))
        x0 = random.uniform(max(0, box[2] - side), min(box[0], W - side))
        y0 = random.uniform(max(0, box[3] - side), min(box[1], H - side))
        crop = img[int(y0):int(y0 + side), int(x0):int(x0 + side)]
        s = size / crop.shape[1]
        crop = cv2.resize(crop, (size, int(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
        yield degrade(crop), (box - [int(x0), int(y0), int(x0), int(y0)]) * s


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", nargs="+", required=True, help="photos of the module")
    parser.add_argument("--phycam", nargs="*", default=[], help="phyCAM frames (raw, no module)")
    parser.add_argument("--backgrounds", nargs="*", default=[], help="any other pictures")
    parser.add_argument("--class-name", default="phyflex_imx8mp")
    parser.add_argument("-o", "--out", default="dataset_synth")
    parser.add_argument("-n", "--count", type=int, default=4000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--negatives", type=float, default=0.15)
    parser.add_argument("--min-size", type=float, default=0.03,
                        help="smallest module, fraction of the picture's short side")
    parser.add_argument("--max-size", type=float, default=0.6)
    parser.add_argument("--real-crops", type=int, default=12,
                        help="zoomed crops per module photo added as real pictures")
    parser.add_argument("--size", type=int, default=960)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    photos = []
    for path in args.module:
        img = cv2.imread(str(path))
        rect = find_module(img)
        photos.append((path, img, rect))
    modules = [cut_module(img, rect) for _, img, rect in photos]
    print("modules:", [m.shape[:2] for m in modules])

    def files(dirs):
        return sorted(p for d in dirs for p in pathlib.Path(d).rglob("*")
                      if p.suffix.lower() in EXTS)
    phycam, others = files(args.phycam), files(args.backgrounds)
    out = pathlib.Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_val = int(args.count * args.val_fraction)
    for i in range(args.count):
        split = "val" if i < n_val else "train"
        bg = random_background(phycam, others, args.size)
        boxes = []
        if random.random() >= args.negatives:
            for _ in range(random.choice([1, 1, 1, 2, 2, 3])):
                box = paste(bg, modules, boxes, args.min_size, args.max_size)
                if box is not None:
                    boxes.append(box)
        write(out, split, f"synth_{i:05d}", degrade(bg), boxes)

    n_real = 0
    for path, img, rect in photos:
        for j, (pic, box) in enumerate(real_variants(img, rect_box(rect, img.shape),
                                                     args.real_crops, args.size)):
            write(out, "train", f"real_{pathlib.Path(path).stem}_{j:02d}", pic, [box])
            n_real += 1

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: {args.class_name}\n")
    print(f"{args.count} synthetic ({n_val} val) + {n_real} real-photo pictures -> {out}")


if __name__ == "__main__":
    main()
