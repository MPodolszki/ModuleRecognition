"""Scenes with several products, for the look-alikes the model mixes up.

    phytec-scenes                                  # dataset/products/szenen.csv
    phytec-scenes --scenes /var/lib/phytec-detect/szenen.csv --per-scene 20

szenen.csv lists the scenes, one per line: the article numbers that lie on
the table together, left to right (separated by ';'), e.g. AM64x next to
AM65x. For each scene the monitor names the products and their places --
the table is split into as many vertical stripes as there are products,
"LINKS: AM64x  RECHTS: AM65x" -- and a picture is taken every --interval
seconds while someone keeps moving, turning and flipping them, each inside
its own stripe (phones, keys, hands may join in). tools/label_scenes.py on
the PC finds the boxes with the float model and names them by their stripe
-- not by the model's class, which is exactly what is unreliable for these
look-alikes.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time

import cv2

from .camera import Camera, Screen, die, screen_size, to_bgrx
from .photo_cli import draw_rect, find_products, read_products

OUT = "/var/lib/phytec-detect/capture/_scenes"


def _xspan(rect):
    """Leftmost and rightmost x of a turned rectangle."""
    xs = cv2.boxPoints(rect)[:, 0]
    return float(xs.min()), float(xs.max())


def read_scenes(path):
    scenes = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(";") if p.strip()]
            if parts and not parts[0].startswith("#") and parts[0].lower() != "artikelnummern":
                scenes.append(parts)
    return scenes


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phytec-scenes", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", default="/var/lib/phytec-detect/szenen.csv")
    parser.add_argument("--products", default="/var/lib/phytec-detect/products.csv")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--per-scene", type=int, default=20)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--intro", type=float, default=12.0,
                        help="seconds to lay the products out before the first picture")
    args = parser.parse_args(argv)

    names = dict(read_products(args.products))
    scenes = read_scenes(args.scenes)
    if not scenes:
        die(f"no scenes in {args.scenes}")
    os.makedirs(args.out, exist_ok=True)
    try:
        cam = Camera("csi1", isp=True)
    except RuntimeError as error:
        die(str(error))
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    sw, sh = screen_size()
    screen = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        for k, scene in enumerate(scenes):
            tag = "+".join(scene)
            done = len([f for f in os.listdir(args.out) if f.startswith(tag + "_")
                        and f.endswith(".jpg")])
            if done >= args.per_scene:
                continue
            label = "  +  ".join(names.get(a, a) for a in scene)
            places = (["LINKS", "RECHTS"] if len(scene) == 2 else
                      ["LINKS", "MITTE", "RECHTS"] if len(scene) == 3 else
                      [f"STREIFEN {i + 1}" for i in range(len(scene))])
            start, last, n, ready_since = time.monotonic(), 0.0, done, None
            print(f"scene {k + 1}/{len(scenes)}: {label}", flush=True)
            while not stop.is_set() and n < args.per_scene:
                ok, gray = cam.read()
                if not ok:
                    die("could not read from camera")
                now = time.monotonic()
                intro = now - start < args.intro
                # Pictures only while each stripe holds exactly one product and
                # none lies across a stripe line -- otherwise the labels are wrong.
                h, w = gray.shape[:2]
                found = find_products(gray)
                status = []
                for i in range(len(scene)):
                    x0, x1 = w * i / len(scene), w * (i + 1) / len(scene)
                    inside = [r for r in found if x0 <= r[0][0] < x1]
                    across = any(_xspan(r)[0] < x0 - 5 or _xspan(r)[1] > x1 + 5 for r in inside)
                    status.append("ok" if len(inside) == 1 and not across else
                                  "fehlt" if not inside else
                                  "auf der Linie" if across else f"{len(inside)} Teile")
                ready = all(st == "ok" for st in status)
                ready_since = (ready_since or now) if ready else None
                steady = ready_since is not None and now - ready_since >= 0.8
                if not intro and steady and now - last >= args.interval:
                    last = now
                    n += 1
                    base = os.path.join(args.out, f"{tag}_{n:03d}")
                    cv2.imwrite(base + ".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    with open(base + ".json", "w", encoding="utf-8") as f:
                        json.dump({"articles": scene, "names": [names.get(a, a) for a in scene],
                                   "path": "isp", "sensor_window": cam.crop,
                                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
                    print(f"  {n}/{args.per_scene}", flush=True)
                view = to_bgrx(cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA))
                for i in range(1, len(scene)):  # the stripes, each product stays in its own
                    x = sw * i // len(scene)
                    for y in range(150, sh, 40):
                        cv2.line(view, (x, y), (x, y + 22), (0, 230, 255, 255), 4)
                scale = sw / w
                for r in found:
                    draw_rect(view, r, scale, (0, 220, 0, 255) if ready else (0, 0, 255, 255), 3)
                for i, a in enumerate(scene):
                    x0 = sw * i // len(scene)
                    ok_ = status[i] == "ok"
                    text = f"{places[i]}: {names.get(a, a)}" + ("" if ok_ else f"  ({status[i]})")
                    cv2.putText(view, text, (x0 + 20, sh - 30), font, 1.1, (0, 0, 0, 255), 6,
                                cv2.LINE_AA)
                    cv2.putText(view, text, (x0 + 20, sh - 30), font, 1.1,
                                (0, 220, 0, 255) if ok_ else (0, 0, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(view, (0, 0), (sw, 150), (0, 0, 0, 255), -1)
                cv2.putText(view, f"Szene {k + 1}/{len(scenes)}:  {label}", (24, 56), font, 1.3,
                            (0, 230, 255, 255), 3, cv2.LINE_AA)
                hint = (f"Hinlegen - Start in {args.intro - (now - start):.0f} s" if intro else
                        f"Bild {n}/{args.per_scene} - umlegen, drehen, umdrehen: jedes im eigenen Streifen"
                        if ready else
                        f"Bild {n}/{args.per_scene} - warte: je ein Produkt pro Streifen (rot = falsch)")
                cv2.putText(view, hint, (24, 112), font, 1.0, (255, 255, 255, 255), 2, cv2.LINE_AA)
                if not intro and now - last < 0.25:
                    view[150:] = 255  # flash: picture taken
                if screen is None:
                    screen = Screen(sw, sh)
                screen.show(view)
            if stop.is_set():
                break
        else:
            print("all scenes done", flush=True)
    finally:
        cam.release()
        if screen is not None:
            screen.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
