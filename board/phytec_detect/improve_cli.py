"""Make the ISP-960 model (v7) as good as the reference (v6): it asks for
products and places, and takes real ISP-960 pictures where v7 is weak.

    phytec-improve --articles PCL-071-5432DE11I.A0,phyBOARD-Nash

For every product (weakest first, --articles) and every marked place on
the table (corners, the drop zone, middle, edges):

 1. "Lege <Produkt> auf die Markierung": the detection camera runs at
    1920x1080 and the reference model (--teacher, v6, the accurate CPU path)
    looks for it. Only when the reference recognises exactly the asked
    product, sure (>= --teacher-conf) and lying still on the mark for a
    second, does anything happen -- so the right module lies there and the
    labels are right.
 2. The camera switches to the ISP's 960x544 output (about a second; the
    product lies still) and the model to improve (--student, v7) looks.
 3. Where v7 is weak (below --student-ok, wrong product or nothing found),
    --shots real ISP-960 pictures are saved, labelled with what the
    reference saw (boxes mapped with isp960.py, tightened on the 960
    picture itself); where v7 is sure, one picture is kept as well, so the
    next training does not unlearn what works.
 4. "Nimm es auf und lege es auf die nächste Markierung".

Saved in --out/<article>/<article>_<place>_<n>: .jpg (960x544, as the ISP
delivers it), .json (boxes [[article, x0, y0, x1, y1], ...], scores of both
models, place), and _ref.jpg (the 1920 picture the reference judged).
tools/fetch_captures.sh brings them to dataset/products/_isp960/, where
tools/build_products.py takes them as training pictures.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np

from . import style
from .camera import Camera, Screen, die, screen_size, to_bgrx
from .detector import Detector
from .isp960 import boxes_to_isp960
from .photo_cli import find_products, read_products

OUT = "/var/lib/phytec-detect/capture/_isp960"
MODELS = "/opt/phytec-detect/models"
# places as fractions of the 1920x1080 picture: the corners and the drop
# zone first (where the ISP-960 view differs most), then middle and edges
PLACES = [("Ablagefeld", 0.86, 0.80), ("oben links", 0.10, 0.14), ("oben rechts", 0.90, 0.14),
          ("unten links", 0.10, 0.86), ("unten rechts", 0.90, 0.86), ("Mitte", 0.50, 0.50),
          ("oben", 0.50, 0.14), ("links", 0.10, 0.50), ("rechts", 0.90, 0.50), ("unten", 0.50, 0.86)]
PLACE_R = 0.12  # the product's centre must lie within this (of the width) of the mark


def _iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def tighten(gray960, boxes):
    """Mapped boxes -> the dark object found right there in the 960 picture
    (the mapping is extrapolated in the far corners); the mapped box where
    nothing fits."""
    big = cv2.resize(gray960, (gray960.shape[1] * 2, gray960.shape[0] * 2),
                     interpolation=cv2.INTER_LINEAR)  # find_products works on 1920-ish pictures
    found = []
    for rect in find_products(big):
        pts = cv2.boxPoints(rect) / 2
        found.append((*pts.min(0), *pts.max(0)))
    out = []
    for cid, x0, y0, x1, y1 in boxes:
        best = max(found, key=lambda f: _iou(f, (x0, y0, x1, y1)), default=None)
        if best is not None and _iou(best, (x0, y0, x1, y1)) > 0.5:
            x0, y0, x1, y1 = best
        out.append((cid, float(x0), float(y0), float(x1), float(y1)))
    return out


class App:
    def __init__(self, args):
        self.args = args
        self.names = dict(read_products(args.products))
        self.teacher = Detector("hailo", os.path.join(args.models, args.teacher), conf=0.3)
        self.student = Detector("hailo", os.path.join(args.models, args.student), conf=0.2)
        self.cam, self.mode = None, None
        self.screen = None
        self.sw, self.sh = screen_size()
        self.stop = threading.Event()
        self.log = []  # (article, place, student score, saved)

    def camera(self, mode):
        """'ref' = 1920x1080 (reference path), '960' = the ISP's scaled output."""
        if self.mode != mode:
            if self.cam is not None:
                self.cam.release()
            self.cam = Camera(self.args.source, isp=True, isp_width=960 if mode == "960" else None)
            self.mode = mode
            for _ in range(4):  # let the ISP's exposure settle after the switch
                self.cam.read()
        ok, frame = self.cam.read()
        if not ok:
            die("could not read from the camera")
        return frame

    def show(self, gray, title, line, boxes=(), place=None, color=style.PETROL):
        view = to_bgrx(cv2.resize(gray, (self.sw, self.sh), interpolation=cv2.INTER_AREA))
        sx, sy = self.sw / gray.shape[1], self.sh / gray.shape[0]
        if place is not None:
            _, px, py = place
            cv2.circle(view, (int(px * self.sw), int(py * self.sh)), int(PLACE_R * self.sw),
                       style.ORANGE + (255,), 4, cv2.LINE_AA)
        for text, (x0, y0, x1, y1), c in boxes:
            cv2.rectangle(view, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy)),
                          c + (255,), 3)
            style.tag(view, int(x0 * sx), int(y0 * sy) - 2, text, bg=c)
        style.header(view, title, title=("PHYTEC", "v7 verbessern"))
        patch = style.text_patch(line, 26, "Medium", bg=color, pad=(16, 10))
        style.blit(view, patch, 0, self.sh - patch.shape[0])
        if self.screen is None:
            self.screen = Screen(self.sw, self.sh)
        self.screen.show(view)

    def name(self, art):
        return self.names.get(art) or art

    def wait_teacher(self, art, k, place, n_places):
        """Until the reference sees `art` sure and still on the place."""
        cls = self.teacher.labels.index(art)
        still_since, last = None, None
        t_start = time.monotonic()
        while not self.stop.is_set():
            frame = self.camera("ref")
            dets = self.teacher(frame)
            h, w = frame.shape[:2]
            px, py = place[1] * w, place[2] * h
            mine = [d for d in dets if int(d[5]) == cls and d[4] >= self.args.teacher_conf
                    and np.hypot((d[0] + d[2]) / 2 - px, (d[1] + d[3]) / 2 - py) < PLACE_R * w]
            now = time.monotonic()
            if mine:
                box = mine[0][:4]
                moved = last is None or np.abs(box - last).max() > 0.01 * w
                still_since = now if moved or still_since is None else still_since
                last = box
            else:
                still_since, last = None, None
            other = [d for d in dets if d[4] >= 0.5]
            boxes = [(f"{self.name(self.teacher.labels[int(d[5])])} {d[4] * 100:.0f} %", d[:4],
                      style.GREEN if int(d[5]) == cls else style.PETROL) for d in other]
            held = 0.0 if still_since is None else now - still_since
            if mine:
                line = (f"{self.name(art)} erkannt ({mine[0][4] * 100:.0f} %) -- "
                        f"ruhig liegen lassen {max(0.0, self.args.hold - held):.1f} s")
            else:
                line = f"Lege {self.name(art)} auf die Markierung: {place[0]}"
            self.show(frame, f"{self.name(art)}  ·  Markierung {k + 1}/{n_places}: {place[0]}",
                      line, boxes, place)
            if mine and held >= self.args.hold:
                return frame, dets
            if now - t_start > self.args.give_up:
                return None, None
        return None, None

    def judge_and_save(self, art, place, ref_frame, ref_dets):
        """Switch to ISP-960, let v7 look, save where it is weak."""
        cls_t = self.teacher.labels.index(art)
        shown = [d for d in ref_dets if d[4] >= 0.5]
        mapped = boxes_to_isp960([(self.teacher.labels[int(d[5])], *d[:4]) for d in shown])
        scores, saved = [], 0
        out_dir = os.path.join(self.args.out, art)
        os.makedirs(out_dir, exist_ok=True)
        base_n = len([f for f in os.listdir(out_dir) if f.endswith("_ref.jpg")])
        for shot in range(self.args.shots):
            g = self.camera("960")
            labels = tighten(g, mapped)
            target = next((b for b in labels if b[0] == art), None)
            student = self.student(g)
            best = 0.0
            if target is not None:
                for d in student:
                    if _iou(d[:4], target[1:]) > 0.5:
                        name = self.student.labels[int(d[5])]
                        best = max(best, float(d[4]) if name == art else 0.0)
            scores.append(best)
            weak = best < self.args.student_ok
            if weak or shot == 0:  # weak: all shots; sure: one, so nothing is unlearnt
                n = base_n + 1
                stem = os.path.join(out_dir, f"{art}_{place[0].replace(' ', '-')}_{n:03d}_{shot}")
                cv2.imwrite(stem + ".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if shot == 0:
                    cv2.imwrite(stem.rsplit("_", 1)[0] + "_ref.jpg", ref_frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(stem + ".json", "w", encoding="utf-8") as f:
                    json.dump({"boxes": [list(b) for b in labels], "article": art,
                               "place": place[0], "student_score": best,
                               "teacher_score": float(max(d[4] for d in shown
                                                          if int(d[5]) == cls_t)),
                               "student": self.args.student, "teacher": self.args.teacher,
                               "size": [g.shape[1], g.shape[0]], "path": "isp960",
                               "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
                saved += 1
            boxes = [(f"{self.name(self.student.labels[int(d[5])])} {d[4] * 100:.0f} %", d[:4],
                      style.GREEN if d[4] >= self.args.student_ok else style.ORANGE)
                     for d in student if d[4] >= 0.3]
            self.show(g, f"{self.name(art)}  ·  {place[0]}  ·  ISP-960",
                      f"v7: {best * 100:.0f} %  " + ("-- schwach, Bild gespeichert" if weak
                                                     else "-- sicher"),
                      boxes, color=style.ORANGE if weak else style.GREEN)
            if not weak and shot > 0:
                break
        return float(np.median(scores)), saved

    def wait_moved(self, art, place, seconds=2.0):
        """Show the result and wait until the product left the place."""
        cls = self.teacher.labels.index(art)
        t0 = time.monotonic()
        while not self.stop.is_set():
            frame = self.camera("ref")
            dets = self.teacher(frame)
            h, w = frame.shape[:2]
            there = any(int(d[5]) == cls and d[4] >= 0.5 and np.hypot(
                (d[0] + d[2]) / 2 - place[1] * w, (d[1] + d[3]) / 2 - place[2] * h) < PLACE_R * w
                for d in dets)
            self.show(frame, f"{self.name(art)}  ·  {place[0]} erledigt",
                      "Nimm es auf und lege es auf die nächste Markierung", place=place)
            if not there and time.monotonic() - t0 > seconds:
                return

    def run(self):
        arts = [a for a in self.args.articles.split(",") if a] if self.args.articles else \
            [a for a, _ in read_products(self.args.products)]
        arts = [a for a in arts if a in self.teacher.labels and a in self.student.labels]
        places = PLACES[:self.args.places]
        for art in arts:
            for k, place in enumerate(places):
                if self.stop.is_set():
                    return
                frame, dets = self.wait_teacher(art, k, place, len(places))
                if frame is None:
                    print(f"{art} {place[0]}: not recognised by the reference -- skipped", flush=True)
                    continue
                score, saved = self.judge_and_save(art, place, frame, dets)
                self.log.append((art, place[0], score, saved))
                print(f"{art} {place[0]}: v7 {score * 100:.0f} %, {saved} saved", flush=True)
                self.wait_moved(art, place)

    def close(self):
        if self.cam is not None:
            self.cam.release()
        if self.screen is not None:
            self.screen.close()
        for det in (self.teacher, self.student):
            for m in det.models:
                m.backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phytec-improve", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--articles", default="",
                        help="comma-separated article numbers, weakest first (default: all of --products)")
    parser.add_argument("--products", default="/var/lib/phytec-detect/products.csv")
    parser.add_argument("--models", default=MODELS)
    parser.add_argument("--teacher", default="products_v6_n")
    parser.add_argument("--student", default="products_v7_n")
    parser.add_argument("--teacher-conf", type=float, default=0.9)
    parser.add_argument("--student-ok", type=float, default=0.9)
    parser.add_argument("--hold", type=float, default=1.0, help="seconds still before judging")
    parser.add_argument("--shots", type=int, default=3, help="pictures per weak place")
    parser.add_argument("--places", type=int, default=len(PLACES), help="first N places")
    parser.add_argument("--give-up", type=float, default=90.0,
                        help="seconds without the reference recognising the product: next place")
    parser.add_argument("--source", default="csi1")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args(argv)

    app = App(args)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: app.stop.set())
    try:
        app.run()
    finally:
        app.close()
        weak = [(a, p, s) for a, p, s, _ in app.log if s < args.student_ok]
        print(f"done: {len(app.log)} places judged, {len(weak)} weak, "
              f"{sum(n for *_, n in app.log)} pictures saved in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
