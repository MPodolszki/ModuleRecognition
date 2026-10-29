#!/usr/bin/env python3
"""dataset/products/<Artikelnummer>/ -> YOLO dataset, labelled automatically.

Layout (see dataset/products/README.md): one folder per product named by its
article number, photos showing only that product on a light background,
products.csv (Artikelnummer;Bezeichnung), _background/ for pictures without
any product. HEIC (iPhone) is read via pillow-heif, EXIF rotation applied.

Output (default dataset_products/):
  images|labels/{train,val}/real_*   the photos, box from segmentation, class
                                     from the folder; every 5th photo of a
                                     product goes to val
  images|labels/{train,val}/synth_*  scenes with several products pasted at
                                     random size/pose on table-like, office
                                     and _background pictures
  cutouts/<Artikelnummer>/*.png      the segmented products (used for synth)
  check_<Artikelnummer>.jpg          every photo with its automatic box --
                                     look at these before training
  data.yaml, classes.csv

The detection camera is monochrome, so everything is written as gray
(--color keeps colour).

    uv run tools/build_products.py --phycam dataset/raw/phycam_bg \\
        --backgrounds datasets_ext/coco128/images/train2017 -n 6000
"""

import argparse
import csv
import math
import os
import pathlib
import random
import shutil
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from synth_dataset import (_overlap, color_jitter, cut_module, degrade,  # noqa: E402
                           find_module, random_background, rect_box)
sys.path.insert(0, str(ROOT / "board"))
from phytec_detect.isp960 import boxes_to_isp960, to_isp960  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif"}
REAL_SIZE = 1600      # long side of the real photos in the dataset
SYNTH_SIZE = 1280     # long side of synthetic scenes
# pictures from the demo setup's detection camera (phytec-photo): VM-017,
# and the VM-117 (same AR0521 sensor, other lens and view) from 2026-10-23
SETUP_FOLDERS = ("vm017", "vm017_isp", "vm117", "vm117_isp", "imx95_isp")


def read_image(path):
    """BGR image from any supported file, HEIC included, EXIF-rotated."""
    if path.suffix.lower() in (".heic", ".heif"):
        import pillow_heif
        from PIL import Image, ImageOps
        pillow_heif.register_heif_opener()
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def images_in(folder):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)


def load_classes(products):
    """Article numbers in products.csv order, then any other folders."""
    names, seen = {}, []
    csv_path = products / "products.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter=";"):
                if len(row) >= 1 and row[0] and row[0] != "Artikelnummer":
                    names[row[0].strip()] = row[1].strip() if len(row) > 1 else ""
    folders = [p.name for p in sorted(products.iterdir())
               if p.is_dir() and not p.name.startswith("_") and images_in(p)]
    for art in list(names) + folders:
        if art in folders and art not in seen:
            seen.append(art)
    return seen, names


def to_out(img, gray):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) if gray else img


def write(out, split, name, img, boxes):
    """boxes: list of (class_id, x0, y0, x1, y1) in img pixels."""
    h, w = img.shape[:2]
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(out / "labels" / split / f"{name}.txt", "w") as f:
        for c, x0, y0, x1, y1 in boxes:
            x0, x1 = np.clip([x0, x1], 0, w)
            y0, y1 = np.clip([y0, y1], 0, h)
            f.write(f"{c} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                    f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")


def table_background(size):
    """Bright table-like surfaces: the demo's real scene."""
    base = random.uniform(150, 245)
    bg = np.full((size, int(size * 0.75), 3), base, np.float32)
    bg += np.random.uniform(-12, 12, 3)  # slight tint
    yy, xx = np.mgrid[0:bg.shape[0], 0:bg.shape[1]].astype(np.float32)
    # uneven lighting: a broad gradient and a soft hot spot
    bg *= (1 + random.uniform(-0.25, 0.25) * xx / bg.shape[1]
           + random.uniform(-0.25, 0.25) * yy / bg.shape[0])[..., None]
    cx, cy = random.uniform(0, bg.shape[1]), random.uniform(0, bg.shape[0])
    bg *= (1 + random.uniform(0, 0.2) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                                               / (2 * (size / 3) ** 2)))[..., None]
    noise = cv2.GaussianBlur(np.random.randn(*bg.shape[:2]).astype(np.float32),
                             (0, 0), random.uniform(0.8, 6)) * random.uniform(3, 15)
    bg += noise[..., None]
    if random.random() < 0.3:  # a few lines / edges like table seams or sheets of paper
        for _ in range(random.randint(1, 3)):
            p1 = (random.randint(0, bg.shape[1]), random.randint(0, bg.shape[0]))
            p2 = (random.randint(0, bg.shape[1]), random.randint(0, bg.shape[0]))
            cv2.line(bg, p1, p2, (base * random.uniform(0.6, 0.95),) * 3,
                     random.randint(1, 4))
    return np.clip(bg, 0, 255).astype(np.uint8)


def paste_one(bg, cutout, occupied, min_frac, max_frac):
    """Paste `cutout` at a random pose; returns the box or None."""
    bh, bw = bg.shape[:2]
    target = math.exp(random.uniform(math.log(min_frac), math.log(max_frac))) * min(bh, bw)
    mh, mw = cutout.shape[:2]
    s = target / max(mh, mw)
    angle = random.uniform(0, 360)
    corners = np.array([[0, 0], [mw, 0], [mw, mh], [0, mh]], np.float32) - [mw / 2, mh / 2]
    tilt = random.uniform(0, 0.12)  # top-down camera: little perspective
    side = random.randrange(4)
    corners[side] *= 1 - tilt
    corners[(side + 1) % 4] *= 1 - tilt
    rad = np.deg2rad(angle)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    quad = (corners * s) @ rot.T
    x_min, y_min = quad.min(0)
    x_max, y_max = quad.max(0)
    if x_max - x_min >= bw or y_max - y_min >= bh:
        return None
    for _ in range(30):
        cx = random.uniform(-x_min, bw - x_max)
        cy = random.uniform(-y_min, bh - y_max)
        box = np.array([cx + x_min, cy + y_min, cx + x_max, cy + y_max])
        if all(_overlap(box, o) < 0.05 for o in occupied):
            break
    else:
        return None
    src = np.array([[0, 0], [mw, 0], [mw, mh], [0, mh]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, (quad + [cx, cy]).astype(np.float32))
    warped = cv2.warpPerspective(color_jitter(cutout), matrix, (bw, bh),
                                 flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((mh, mw), 255, np.uint8), matrix, (bw, bh))
    mask = cv2.GaussianBlur(mask, (3, 3), 0)[..., None] / 255.0
    if random.random() < 0.7:  # soft shadow
        off = np.random.randint(2, 10, 2)
        shadow = np.roll(mask[..., 0], tuple(off), axis=(0, 1))
        shadow = cv2.GaussianBlur(shadow, (0, 0), 5)[..., None] * random.uniform(0.15, 0.45)
        bg[:] = (bg * (1 - shadow)).astype(np.uint8)
    bg[:] = (warped * mask + bg * (1 - mask)).astype(np.uint8)
    return box


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--products", default=str(ROOT / "dataset" / "products"))
    parser.add_argument("-o", "--out", default=str(ROOT / "dataset_products"))
    parser.add_argument("--phycam", nargs="*", default=[], help="camera frames without products")
    parser.add_argument("--backgrounds", nargs="*", default=[], help="any other pictures")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4,
                        help="processes for the synthetic scenes")
    parser.add_argument("--neg-repeat", type=int, default=3,
                        help="training copies of each hard negative (_background/*negatives*/)")
    parser.add_argument("-n", "--synth", type=int, default=6000, help="synthetic scenes")
    parser.add_argument("--min-size", type=float, default=0.04,
                        help="smallest product, fraction of the scene's short side")
    parser.add_argument("--max-size", type=float, default=0.45)
    parser.add_argument("--color", action="store_true", help="keep colour (default: gray)")
    parser.add_argument("--setup-repeat", type=int, default=5,
                        help="training copies of each demo-setup picture (vm017*/, vm117*/ folders)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--isp960", action="store_true",
                        help="also every ISP picture (1920x1080: photos, scenes, negatives) as "
                             "the ISP delivers it at 960x544 (tools/isp960.py) -- so the model "
                             "works with phytec-detect --isp-width 960 as well")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    # No OpenCV worker threads in this process: the synthetic scenes are
    # made in forked processes, and a fork taken while OpenCV's thread pool
    # holds a lock leaves the children hanging (0 % CPU, nothing written).
    cv2.setNumThreads(0)
    gray = not args.color
    products, out = pathlib.Path(args.products), pathlib.Path(args.out)
    classes, display = load_classes(products)
    if not classes:
        sys.exit(f"no product folders with pictures under {products}")
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    # --- real photos + cutouts --------------------------------------------
    cutouts = {c: [] for c in range(len(classes))}
    n_real = {"train": 0, "val": 0}
    for cid, art in enumerate(classes):
        (out / "cutouts" / art).mkdir(parents=True)
        tiles = []
        for k, path in enumerate(images_in(products / art)):
            # the folder in the name: each board numbers its pictures from
            # 001 (vm017_isp/X_001 and imx95_isp/X_001 must not overwrite)
            stem = path.stem if path.parent == products / art else f"{path.parent.name}_{path.stem}"
            img = read_image(path)
            meta_path = path.with_suffix(".json")
            rect, lifted = None, False
            if meta_path.exists():  # phytec-photo found the product already
                import json
                m = json.loads(meta_path.read_text())
                # tools/label_review.py (by hand) over clean_photo_labels.py
                if m["skip_user"] if m.get("skip_user") is not None else m.get("skip"):
                    continue
                lifted = m.get("kind") == "lifted"
                if m.get("rect"):
                    (cx, cy), (rw, rh), ang = m["rect"]
                    rect = ((cx, cy), (rw, rh), ang)
            if rect is None:
                try:
                    rect = find_module(img)
                except ValueError:
                    print(f"  {path.name}: no product found -- skipped")
                    continue
            if not lifted:  # a hand around it: no cutout for the synthetic scenes
                cut = cut_module(img, rect)
                cv2.imwrite(str(out / "cutouts" / art / f"{stem}.png"), cut)
                cutouts[cid].append(cut)
            scale = REAL_SIZE / max(img.shape[:2])
            small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            x0, y0, x1, y1 = rect_box(rect, img.shape) * scale
            split = "val" if k % 5 == 4 else "train"
            # Pictures from the demo setup itself (phytec-photo, in vm017/)
            # are what the model will see -- weight them up against the
            # thousands of synthetic scenes.
            copies = args.setup_repeat if (split == "train" and path.parent.name in SETUP_FOLDERS) else 1
            for c in range(copies):
                write(out, split, f"real_{art}_{stem}" + (f"_r{c}" if c else ""),
                      to_out(small, gray), [(cid, x0, y0, x1, y1)])
                n_real[split] += 1
            small960 = to_isp960(img) if args.isp960 else None
            if small960 is not None:
                boxes960 = boxes_to_isp960([(cid, *rect_box(rect, img.shape))])
                for c in range(copies if boxes960 else 0):
                    write(out, split, f"real960_{art}_{stem}" + (f"_r{c}" if c else ""),
                          to_out(small960, gray), boxes960)
                    n_real[split] += 1
            vis = small.copy()
            cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 4)
            # zoom on the box, so a sloppy box is visible
            pad = int(0.6 * max(x1 - x0, y1 - y0))
            zx0, zy0 = max(int(x0) - pad, 0), max(int(y0) - pad, 0)
            zoom = vis[zy0:int(y1) + pad, zx0:int(x1) + pad]
            tiles.append(cv2.resize(zoom, (220, 220)))
        if tiles:
            rows = [np.hstack(tiles[i:i + 8] + [np.zeros_like(tiles[0])] * (8 - len(tiles[i:i + 8])))
                    for i in range(0, len(tiles), 8)]
            cv2.imwrite(str(out / f"check_{art}.jpg"), np.vstack(rows))
        print(f"{art}: {len(cutouts[cid])} photos labelled")

    # scenes with several products (phytec-scenes + tools/label_scenes.py):
    # real look-alikes side by side, weighted like the setup pictures
    scene_dir = products / "_scenes"
    n_scene = 0
    for k, box_path in enumerate(sorted(scene_dir.glob("*.boxes.json")) if scene_dir.exists() else []):
        import json
        jpg = box_path.with_name(box_path.name[:-len(".boxes.json")] + ".jpg")
        labelled = [b for b in json.loads(box_path.read_text()) if b[0] in classes]
        if not jpg.exists() or not labelled:
            continue
        img = read_image(jpg)
        scale = REAL_SIZE / max(img.shape[:2])
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        boxes = [(classes.index(a), x0 * scale, y0 * scale, x1 * scale, y1 * scale)
                 for a, x0, y0, x1, y1 in labelled]
        split = "val" if k % 5 == 4 else "train"
        for c in range(args.setup_repeat if split == "train" else 1):
            write(out, split, f"scene_{jpg.stem}" + (f"_r{c}" if c else ""), to_out(small, gray), boxes)
            n_scene += 1
        small960 = to_isp960(img) if args.isp960 else None
        if small960 is not None:
            boxes960 = boxes_to_isp960([(classes.index(a), x0, y0, x1, y1)
                                        for a, x0, y0, x1, y1 in labelled])
            for c in range(args.setup_repeat if split == "train" else 1):
                write(out, split, f"scene960_{jpg.stem}" + (f"_r{c}" if c else ""),
                      to_out(small960, gray), boxes960)
                n_scene += 1
    if n_scene:
        print(f"scenes: {n_scene} pictures (with copies)")

    # real ISP-960 pictures from phytec-improve: labelled by the reference
    # model, taken where the ISP-960 model was weak -- weighted like the
    # setup pictures (weak places more: every weak shot is kept)
    isp_dir = products / "_isp960"
    n_isp = 0
    for k, meta_path in enumerate(sorted(isp_dir.rglob("*.json")) if isp_dir.exists() else []):
        import json
        meta = json.loads(meta_path.read_text())
        jpg = meta_path.with_suffix(".jpg")
        boxes = [(classes.index(a), x0, y0, x1, y1) for a, x0, y0, x1, y1 in meta.get("boxes", [])
                 if a in classes]
        if not jpg.exists() or not boxes:
            continue
        img = read_image(jpg)
        split = "val" if k % 5 == 4 else "train"
        for c in range(args.setup_repeat if split == "train" else 1):
            write(out, split, f"isp960_{meta_path.stem}" + (f"_r{c}" if c else ""),
                  to_out(img, gray), boxes)
            n_isp += 1
    if n_isp:
        print(f"real ISP-960 pictures (phytec-improve): {n_isp} (with copies)")

    usable = [c for c in cutouts if cutouts[c]]
    if not usable:
        sys.exit("no product could be segmented")

    # --- synthetic scenes ----------------------------------------------------
    def files(dirs):
        return sorted(p for d in dirs for p in pathlib.Path(d).rglob("*")
                      if p.suffix.lower() in EXTS)
    neg = files([products / "_background"])
    phycam, others = files(args.phycam), files(args.backgrounds)
    n_val = args.synth // 10

    def synth_range(lo, hi):
        """Scenes lo..hi-1, in a worker process (forked: cutouts are shared)."""
        random.seed(args.seed * 100003 + lo)
        np.random.seed((args.seed * 100003 + lo) % 2**32)
        cv2.setNumThreads(1)
        for i in range(lo, hi):
            roll = random.random()
            if roll < 0.55:
                bg = table_background(SYNTH_SIZE)
            elif neg and roll < 0.75:
                bg = read_image(random.choice(neg))
                s = SYNTH_SIZE / max(bg.shape[:2])
                bg = cv2.resize(bg, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            else:
                bg = random_background(phycam, others, SYNTH_SIZE)
            boxes = []
            if random.random() > 0.1:
                for _ in range(random.choice([1, 1, 2, 2, 3, 4, 5])):
                    cid = random.choice(usable)
                    box = paste_one(bg, random.choice(cutouts[cid]), [b[1:] for b in boxes],
                                    args.min_size, args.max_size)
                    if box is not None:
                        boxes.append((cid, *box))
            split = "val" if i < n_val else "train"
            write(out, split, f"synth_{i:05d}", to_out(degrade(bg), gray), boxes)

    # all cores: the scenes are independent (fork keeps the cutouts shared)
    import multiprocessing as mp
    workers = max(1, min(args.jobs, args.synth // 200 or 1))
    step = -(-args.synth // workers)
    procs = [mp.get_context("fork").Process(target=synth_range,
                                            args=(k, min(k + step, args.synth)))
             for k in range(0, args.synth, step)]
    for p_ in procs:
        p_.start()
    for p_ in procs:
        p_.join()
        if p_.exitcode:
            sys.exit(f"a synthetic-scene worker failed ({p_.exitcode})")

    # pictures of _background as plain negatives; the hard ones (phones,
    # keys, hands, ... on the demo table, a "negatives" folder) count more
    k = 0
    for path in neg:
        img = read_image(path)
        s = REAL_SIZE / max(img.shape[:2])
        small = cv2.resize(img, None, fx=s, fy=s)
        small960 = to_isp960(img) if args.isp960 else None
        for _ in range(args.neg_repeat if "negatives" in str(path.parent) else 1):
            write(out, "train", f"neg_{k:04d}", to_out(degrade(small), gray), [])
            k += 1
            if small960 is not None:
                write(out, "train", f"neg960_{k:04d}", to_out(small960, gray), [])

    names = "".join(f"  {i}: '{a}'\n" for i, a in enumerate(classes))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\n"
                                   f"val: images/val\nnames:\n{names}")
    with open(out / "classes.csv", "w", encoding="utf-8") as f:
        f.write("id;Artikelnummer;Bezeichnung\n")
        for i, a in enumerate(classes):
            f.write(f"{i};{a};{display.get(a, '')}\n")
    print(f"{len(classes)} classes, real {n_real}, synthetic {args.synth} ({n_val} val), "
          f"negatives {len(neg)} -> {out}")


if __name__ == "__main__":
    main()
