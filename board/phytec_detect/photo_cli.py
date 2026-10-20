"""Guided product photography with the detection camera.

    phytec-photo --article PCM-070-0F3243I.A3 --name "phyCORE-i.MX8M Plus"

The screen shows the live detection picture and, for every shot of the
plan, a target frame: where the product should lie and how it is turned,
plus a line of text ("7/31  Vorderseite, oben links, ca. 45 Grad"). The
product is found as the dark object on the light table (no model needed,
the product is new); once it lies on the target and holds still, the
camera fires by itself. Lifted shots ("ca. 25 cm anheben") fire after a
countdown.

Saved per shot, in --out/<article>/:
  <article>_NNN.jpg    full-resolution GRAY8 picture
  <article>_NNN.txt    YOLO label, class 0, box of the found product
  <article>_NNN.json   shot plan entry, product rectangle, settings
and one empty-table picture in --out/_background/.

Fetch them with tools/fetch_captures.sh into dataset/products/.
"""

import argparse
import json
import math
import os
import random
import signal
import sys
import threading
import time

import cv2
import numpy as np

from .camera import Camera, Screen, die, screen_size, to_bgrx

VIEW_W = 1296
POSITIONS = [  # (name, x, y) as fractions of the frame
    ("Mitte", 0.5, 0.5), ("oben links", 0.2, 0.25), ("oben Mitte", 0.5, 0.22),
    ("oben rechts", 0.8, 0.25), ("rechts", 0.83, 0.5), ("unten rechts", 0.8, 0.77),
    ("unten Mitte", 0.5, 0.8), ("unten links", 0.2, 0.77), ("links", 0.17, 0.5),
]


def make_plan(back=True, lifted=4, seed=0):
    rnd = random.Random(seed)
    plan = []
    sides = ["Vorderseite"] + (["Rueckseite"] if back else [])
    for side in sides:
        for k, (where, x, y) in enumerate(POSITIONS):
            angles = [0, rnd.choice([30, 45, 60])] if side == "Vorderseite" else \
                [rnd.choice([0, 20, 45, 70])]
            for a in angles:
                plan.append({"kind": "table", "side": side, "where": where, "x": x, "y": y,
                             "angle": a})
    for k in range(lifted):
        where, x, y = POSITIONS[[0, 2, 4, 8][k % 4]]
        plan.append({"kind": "lifted", "side": "Vorderseite" if k % 2 == 0 else "Rueckseite",
                     "where": where, "x": x, "y": y, "angle": 0})
    return plan


_TABLE = {"img": None, "n": 0}


def _table(small, every=10):
    """The table without the products: a grey closing (max then min) with a
    kernel larger than any product removes the dark objects and keeps the
    table's own brightness gradient. (A median of that size still saw mostly
    product where one lay and lost the contrast.)

    It costs ~80 ms on the Cortex-A53, so it is refreshed only every `every`
    frames and on a background thread (OpenCV releases the GIL, so it runs
    on another core) -- the table does not move, the product does, and the
    preview keeps its frame rate. (A rectangular kernel or half the
    resolution were fast too, but found objects on the empty table.)"""
    t = _TABLE

    def compute(img):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
        return cv2.GaussianBlur(cv2.morphologyEx(img, cv2.MORPH_CLOSE, k), (0, 0), 5)

    if t["img"] is None or t["img"].shape != small.shape:
        t["img"] = compute(small)  # first frame: wait for it
    elif t["n"] % every == 0 and not (t.get("thread") and t["thread"].is_alive()):
        snap = small.copy()

        def work():
            t["img"] = compute(snap)
        t["thread"] = threading.Thread(target=work, daemon=True)
        t["thread"].start()
    t["n"] += 1
    return t["img"]


def find_product(gray, min_frac=0.0015, max_frac=0.25):
    """minAreaRect of the darkest compact object on the light table, or None."""
    s = 8
    # Every 8th pixel instead of an INTER_AREA resize of the 5 MP frame: the
    # preview has to keep up with someone placing the product.
    # The 3x3 blur takes the aliasing noise of plain striding out again --
    # without it, noise blobs counted as objects on the empty table.
    small = cv2.GaussianBlur(np.ascontiguousarray(gray[::s, ::s]), (3, 3), 0)
    table = _table(small)
    mask = ((table.astype(np.int16) - small) > 35).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_frac * small.size < area < max_frac * small.size):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + bw >= small.shape[1] - 1 or y + bh >= small.shape[0] - 1:
            continue  # touches the border: table edge, monitor, arm
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        fill = area / (rw * rh + 1e-9)
        if fill < 0.6 or max(rw, rh) > 3 * min(rw, rh) or min(rw, rh) * s < 60:
            continue  # not a compact product: table seams, cables, shadows
        if best is None or area > best[0]:
            best = (area, ((cx * s, cy * s), (rw * s, rh * s), ang))
    return best[1] if best else None


def angle_diff(a, b):
    """Difference of two rectangle angles, symmetric under 90 degrees."""
    d = abs((a - b) % 90)
    return min(d, 90 - d)


def draw_rect(img, rect, scale, color, thickness=2, dashed=False):
    (cx, cy), (w, h), a = rect
    pts = cv2.boxPoints(((cx * scale, cy * scale), (w * scale, h * scale), a)).astype(int)
    if not dashed:
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
        return
    for i in range(4):
        p, q = pts[i], pts[(i + 1) % 4]
        for t in np.arange(0, 1, 0.1):
            a0, a1 = p + (q - p) * t, p + (q - p) * min(t + 0.05, 1)
            cv2.line(img, tuple(a0.astype(int)), tuple(a1.astype(int)), color, thickness)


def text(img, s, y, scale=1.0, color=(255, 255, 255, 255), thick=2):
    cv2.putText(img, s, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0, 255), thick + 4,
                cv2.LINE_AA)
    cv2.putText(img, s, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def save(out_dir, name, gray, rect, meta):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, name)
    cv2.imwrite(base + ".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = gray.shape[:2]
    with open(base + ".txt", "w") as f:
        if rect is not None:
            pts = cv2.boxPoints(rect)
            x0, y0 = np.clip(pts.min(0), 0, [w, h])
            x1, y1 = np.clip(pts.max(0), 0, [w, h])
            f.write(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                    f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")
    with open(base + ".json", "w") as f:
        json.dump(dict(meta, rect=[list(rect[0]), list(rect[1]), rect[2]] if rect else None),
                  f, indent=1)


LIST_W = 440  # product list on the right (--list-csv)


def read_products(path):
    """products.csv (Artikelnummer;Bezeichnung) -> [(article, name)]."""
    rows = []
    try:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(";")]
                if parts[0] and parts[0].lower() != "artikelnummer":
                    rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
    except OSError:
        pass
    return rows


class ProductList:
    """The product list beside the camera picture, re-read every few seconds
    (products.csv may change while the queue runs): done, current, to do."""

    def __init__(self, csv, capture, current, every=5.0):
        self.csv, self.capture, self.current, self.every = csv, capture, current, every
        self.rows, self.done, self._t = [], set(), -1e9
        self._out = None  # screen buffer, list drawn into it on refresh only

    def _draw_list(self):
        out, sw, sh = self._out, self._out.shape[1], self._out.shape[0]
        out[:, sw - LIST_W:] = 0
        x, font = sw - LIST_W + 20, cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, "Produkte", (x, 50), font, 1.1, (255, 255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"{len(self.done)} von {len(self.rows)} fotografiert", (x, 85), font,
                    0.65, (200, 200, 200, 255), 1, cv2.LINE_AA)
        line_h = max(24, min(40, (sh - 130) // max(1, len(self.rows))))
        y = 130
        for article, name in self.rows:
            if article == self.current:
                mark, color = ">", (0, 230, 255, 255)
            elif article in self.done:
                mark, color = "+", (0, 200, 0, 255)
            else:
                mark, color = "-", (200, 200, 200, 255)
            text = f"{mark} {name or article}"
            fs = 0.7
            while cv2.getTextSize(text, font, fs, 2)[0][0] > LIST_W - 30 and fs > 0.4:
                fs -= 0.05
            cv2.putText(out, text, (x, y), font, fs, color, 2, cv2.LINE_AA)
            y += line_h
            if y > sh - 10:
                break

    def compose(self, view, size):
        """Camera picture scaled into the left part of a size (w, h) screen,
        the list in the right part."""
        sw, sh = size
        if self._out is None:
            self._out = np.zeros((sh, sw, 4), np.uint8)
        if time.monotonic() - self._t >= self.every:
            self._t = time.monotonic()
            self.rows = read_products(self.csv)
            self.done = {a for a, _ in self.rows
                         if os.path.exists(os.path.join(self.capture, a, "DONE"))}
            self._draw_list()
        vw = sw - LIST_W
        vh = view.shape[0] * vw // view.shape[1]
        y0 = (sh - vh) // 2
        cv2.resize(view, (vw, vh), dst=self._out[y0:y0 + vh, :vw],
                   interpolation=cv2.INTER_LINEAR)
        return self._out


def _aspect(t):
    if not t:
        return None
    w, _, h = t.partition(":")
    return float(w) / float(h or 1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phytec-photo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--article", required=True, help="article number = folder name")
    parser.add_argument("--name", default="", help="product name (shown, stored)")
    parser.add_argument("--out", default="/var/lib/phytec-detect/capture")
    parser.add_argument("--no-back", action="store_true", help="product has no usable back side")
    parser.add_argument("--lifted", type=int, default=4, help="lifted shots (default 4)")
    parser.add_argument("--aspect", default="16:9")
    parser.add_argument("--crop-y", type=float, default=1.0)
    parser.add_argument("--hold", type=float, default=1.0, help="seconds still before a shot")
    parser.add_argument("--source", default="csi1")
    parser.add_argument("--intro", type=float, default=0.0,
                        help="seconds to announce the product before the first shot "
                             "(phytec-photo-queue)")
    parser.add_argument("--wait-start", action="store_true",
                        help="announce the product until someone lays it in the middle "
                             "(phytec-photo-queue: no need to sit in front of it)")
    parser.add_argument("--previous", default="",
                        help="product finished before this one (shown while waiting)")
    parser.add_argument("--list-csv", default=None,
                        help="show this products.csv on the right, with what is done "
                             "(phytec-photo-queue)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the whole procedure without saving anything, over and "
                             "over (trade fair: phytec-trial)")
    parser.add_argument("--isp", action="store_true",
                        help="through the ISP in its current mode (as phytec-detect --isp); "
                             "--aspect/--crop-y do not apply")
    parser.add_argument("--start", type=int, default=None,
                        help="continue at this shot number of the plan (2 = first table "
                             "shot), skipping the background and reference shots; the "
                             "product size comes from the saved reference shot")
    args = parser.parse_args(argv)

    plan = make_plan(back=not args.no_back, lifted=args.lifted,
                     seed=sum(map(ord, args.article)))
    out_dir = os.path.join(args.out, args.article)
    done = 0 if args.dry_run else len([f for f in os.listdir(out_dir) if f.endswith(".jpg")]) \
        if os.path.isdir(out_dir) else 0

    try:
        cam = Camera(args.source, isp=True) if args.isp else \
            Camera(args.source, aspect=_aspect(args.aspect), crop_y=args.crop_y)
    except RuntimeError as error:
        die(str(error))
    stop = threading.Event()
    # restored on return: phytec-photo-queue calls main() again and again,
    # and must still end on SIGTERM while it waits for new products
    old_handlers = {n: signal.getsignal(n) for n in (signal.SIGINT, signal.SIGTERM,
                                                     signal.SIGUSR1)}
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    snap = threading.Event()  # kill -USR1: save the screen to /tmp/phytec-photo.png
    signal.signal(signal.SIGUSR1, lambda *_: snap.set())

    screen = None
    step = -2              # -2: empty table, -1: reference in the middle, 0..: plan
    size = None            # product (w, h) from the reference shot
    if args.start is not None:
        step = max(0, args.start - 2)   # shot 1 is the reference, shot 2 plan step 0
        for name in sorted(os.listdir(out_dir)):
            if name.endswith(".json"):
                with open(os.path.join(out_dir, name)) as f:
                    meta = json.load(f)
                if meta.get("kind") == "reference" and meta.get("rect"):
                    size = tuple(sorted(meta["rect"][1], reverse=True))
                    break
        if size is None:
            die("no reference shot in " + out_dir + " -- start without --start")
    stable_since, last_c = None, None
    flip_until, banner, flip_text = None, None, ""
    frame_times, n_frames = [], 0
    flash_until, countdown_at = 0.0, None
    shot = done
    intro_until = time.monotonic() + args.intro
    intro_text = None
    plist = ProductList(args.list_csv, args.out, args.article) if args.list_csv else None
    finished = False
    try:
        while not stop.is_set() and step < len(plan):
            ok, gray = cam.read()
            if not ok:
                die("could not read from camera")
            now = time.monotonic()
            h, w = gray.shape[:2]
            step_px = max(1, w // VIEW_W)   # 2 for the 2592-wide sensor
            scale = 1.0 / step_px
            view = to_bgrx(np.ascontiguousarray(gray[::step_px, ::step_px]))
            frame_times = (frame_times + [now])[-20:]
            n_frames += 1
            if n_frames % 60 == 0 and len(frame_times) > 1:
                print(f"preview {(len(frame_times) - 1) / (frame_times[-1] - frame_times[0]):.1f} fps",
                      flush=True)
            rect = find_product(gray)
            title = f"{args.article}  {args.name}"

            if step == -2:
                lines = ["Tisch leer raeumen - Hintergrundbild"]
                fire = rect is None and now >= intro_until
                hold = 2.0
                banner = intro_text or f"NAECHSTES PRODUKT: {args.name or args.article}" \
                    if now < intro_until else None
            elif step == -1:
                lines = ["Produkt (Vorderseite) in die Mitte legen"]
                if args.wait_start:
                    banner = f"BEREIT: {args.name or args.article} - zum Start in die Mitte legen"
                    if args.previous:
                        lines.append(f"Fertig: {args.previous}")
                target = ((w / 2, h / 2), None, 0)
                fire = rect is not None and \
                    math.hypot(rect[0][0] - w / 2, rect[0][1] - h / 2) < 0.12 * w
                hold = args.hold
            else:
                p = plan[step]
                tx, ty = p["x"] * w, p["y"] * h
                if size is not None:  # a big board must fit, clear of the table edge
                    # (the product finder misses products right at the edge)
                    a = math.radians(p["angle"])
                    hx = (abs(math.cos(a)) * size[0] + abs(math.sin(a)) * size[1]) / 2 + 0.09 * w
                    hy = (abs(math.sin(a)) * size[0] + abs(math.cos(a)) * size[1]) / 2 + 0.09 * h
                    top = 160 * h / 1080  # the instruction bar covers the picture's top
                    tx = min(max(tx, hx), w - hx) if 2 * hx < w else w / 2
                    ty = min(max(ty, hy + top), h - hy) if 2 * hy + top < h else (h + top) / 2
                target = ((tx, ty), size, p["angle"])
                prev = plan[step - 1] if step > 0 else {"side": "Vorderseite", "kind": "table"}
                if p["side"] != prev["side"] and p["kind"] == "table" and flip_until is None:
                    flip_until = now + 5.0   # the side changes: say so, loudly
                    flip_text = f"MODUL UMDREHEN: {p['side']} nach oben"
                if p["kind"] == "lifted" and prev["kind"] != "lifted" and flip_until is None:
                    flip_until = now + 6.0   # from here on: in the hand, in the air
                    flip_text = "MODUL ANHEBEN: 25-30 cm hoch, FLACH halten"
                banner = flip_text if flip_until is not None and now < flip_until else None
                if p["kind"] == "lifted":
                    lines = [f"{step + 1}/{len(plan)}  {p['side']} nach oben, {p['where']}: "
                             f"ca. 25-30 cm ANHEBEN",
                             "FLACH halten (parallel zum Tisch), Hand nur am Rand"]
                    if banner:
                        countdown_at = None
                    elif countdown_at is None:
                        countdown_at = now + 5.0
                    # Held too low? Only judged when the product is found on its
                    # own (a hand around it often merges with it).
                    too_low = rect is not None and size is not None and \
                        max(rect[1]) < 1.15 * size[0]
                    if too_low and countdown_at is not None:
                        countdown_at = now + 5.0
                        lines.append("HOEHER HALTEN - Modul wirkt noch nicht groesser")
                    fire = countdown_at is not None and now >= countdown_at
                    hold = 0.0
                    if countdown_at is not None and not banner:
                        big = f"{max(0.0, countdown_at - now):.0f}"
                        vh, vw = view.shape[:2]
                        cv2.putText(view, big, (vw // 2 - 40, vh // 2 + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0, 255), 18, cv2.LINE_AA)
                        cv2.putText(view, big, (vw // 2 - 40, vh // 2 + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 230, 255, 255), 8,
                                    cv2.LINE_AA)
                else:
                    lines = [f"{step + 1}/{len(plan)}  {p['side']}, {p['where']}, "
                             f"ca. {p['angle']} Grad gedreht"]
                    fire = rect is not None and \
                        math.hypot(rect[0][0] - tx, rect[0][1] - ty) < 0.07 * w and \
                        angle_diff(rect[2], p["angle"]) < 15
                    hold = args.hold
                if p["kind"] == "table":
                    draw_rect(view, ((tx, ty), size, p["angle"]), scale, (0, 230, 255, 255), 3,
                              dashed=True)

            # still?
            if rect is not None:
                c = np.array(rect[0])
                moving = last_c is None or np.linalg.norm(c - last_c) > 0.004 * w
                last_c = c
            else:
                moving = step != -2
                last_c = None
            if fire and not moving:
                stable_since = stable_since or now
            else:
                stable_since = None
            if rect is not None:
                draw_rect(view, rect, scale, (0, 255, 0, 255) if fire else (0, 0, 255, 255), 3)

            if step >= 0 and flip_until is not None and now < flip_until:
                stable_since = None  # no shot while the flip banner is up
            if (stable_since and now - stable_since >= hold) or (fire and hold == 0.0):
                meta = {"article": args.article, "name": args.name, "aspect": args.aspect,
                        "crop_y": args.crop_y, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                if args.isp:
                    meta.update(path="isp", aspect=None, crop_y=None, sensor_window=cam.crop)
                if step == -2:
                    if not args.dry_run:
                        save(os.path.join(args.out, "_background"),
                             f"bg_{args.article}_{int(time.time())}", gray, None, meta)
                else:
                    if step == -1:
                        size = tuple(sorted(rect[1], reverse=True))
                    meta.update(plan[step] if step >= 0 else {"kind": "reference"})
                    shot += 1
                    if not args.dry_run:
                        save(out_dir, f"{args.article}_{shot:03d}", gray, rect, meta)
                    print(f"shot {shot}: {meta.get('kind')} {meta.get('side', '')} "
                          f"{meta.get('where', '')}", flush=True)
                step += 1
                stable_since, countdown_at, flash_until = None, None, now + 0.3
                flip_until = None
                if args.dry_run and step >= len(plan):  # show it again, from the start
                    step, size, shot = -2, None, 0
                    intro_until = now + 8.0
                    intro_text = "FERTIG - so einfach lernt die Demo ein neues Produkt"

            # overlay
            cv2.rectangle(view, (0, 0), (view.shape[1], 120 + 34 * (len(lines) - 1)),
                          (0, 0, 0, 255), -1)
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) \
                if len(frame_times) > 1 else 0.0
            text(view, title, 40, 0.9, (0, 230, 255, 255))
            cv2.putText(view, f"{fps:4.1f} fps", (view.shape[1] - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160, 255), 2, cv2.LINE_AA)
            for i, line in enumerate(lines):
                text(view, line, 82 + 34 * i, 0.85)
            if banner:
                vh, vw = view.shape[:2]
                # waiting for the start: green and above the middle, where the
                # product goes; otherwise red across the middle
                by, bcol = (vh // 4, (0, 140, 0, 255)) if step == -1 else (vh // 2, (0, 0, 200, 255))
                cv2.rectangle(view, (0, by - 70), (vw, by + 70), bcol, -1)
                fs = 1.6
                while cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, fs, 4)[0][0] > vw - 80:
                    fs -= 0.1
                tw = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, fs, 4)[0][0]
                cv2.putText(view, banner, ((vw - tw) // 2, by + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            fs, (255, 255, 255, 255), 4, cv2.LINE_AA)
            if now < flash_until:
                view[:] = 255
            if plist is not None:
                view = plist.compose(view, screen_size())
            if snap.is_set():
                snap.clear()
                cv2.imwrite("/tmp/phytec-photo.png", view[..., :3])
            if screen is None:
                screen = Screen(view.shape[1], view.shape[0])
            screen.show(view)
        if step >= len(plan):
            print(f"done: {shot - done} new pictures in {out_dir}", flush=True)
            with open(os.path.join(out_dir, "DONE"), "w") as f:  # phytec-photo-queue
                f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
            finished = True
    finally:
        cam.release()
        if screen is not None:
            screen.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return 0 if finished else 1


if __name__ == "__main__":
    sys.exit(main())
