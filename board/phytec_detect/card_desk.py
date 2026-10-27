"""The blue card zone: a visitor lays a product and a business card there,
the demo reads the name and thanks the visitor; sales gets the contact.

Whenever the zone's part of the detection picture has changed and then
lain still for HOLD_S, it goes to the card reader (phytec_detect.ocr) in a
thread of its own, the detected products blanked out -- independent of
the detector, which may miss a product the card half hides. An unchanged
zone is not read again. The name goes when the card or the product is
taken out (a patch of the zone differs from the picture it was read in)
or after SHOW_S at the latest; the same card lying on is not greeted again.

A visit is saved in LEADS_DIR only with a name and a product in the zone
(detected there within SEEN_S of the read -- the card may hide it): the
zone picture (JPEG) and a line in leads.csv (time; first name; surname;
article; product; all text read). That is personal data of visitors: delete it after the
fair, and say at the stand that cards laid there are kept for contact.
"""

import csv
import os
import threading
import time

import cv2
import numpy as np

HOLD_S = 1.0               # still for this long: read
MOVED, CHANGED = 4.0, 8.0  # mean grey difference (0..255) of the zone, 64 x 64
RETRIES, RETRY_S = 3, 3.0  # no name: read the still zone again (exposure settling, a hand)
SHOW_S = 30.0              # the banner at most this long
TAKEN = 0.03               # this share of the zone changed by > 25 grey levels: taken out
SEEN_S = 3.0               # a product detected in the zone this recently lies there
LEADS_DIR = "/var/lib/phytec-detect/leads"


class CardDesk:
    def __init__(self, labels, names, model_dir, leads_dir=LEADS_DIR):
        self.labels, self.names, self.leads_dir = labels, names, leads_dir
        self.reader = None
        self.error = None
        self.name = None          # (first, last) of the visitor now at the zone
        self.version = 0          # changes whenever the banner does
        self._prev, self._still_since = None, 0.0
        self._read_small, self._name_small = None, None
        self._busy = False
        self._tries, self._last_read = 0, 0.0
        self._shown_at = 0.0
        self._greeted = None      # greeted, card not taken away since
        self.card_quad = None     # the card's outline (detection picture), with the name
        self._seen = {}           # class -> last time detected in the zone
        self._name_classes = ()

        def load():  # the NPU compiles its graph on the first run (~15 s)
            try:
                from .ocr import CardReader
                reader = CardReader(model_dir)
                reader(np.full((200, 320), 255, np.uint8))
                self.reader = reader
                print(f"card reader ready ({reader.det.kind} / {reader.rec.kind})", flush=True)
            except Exception as error:  # noqa: BLE001 -- the demo runs on without names
                self.error = f"{type(error).__name__}: {error}"
                print(f"card reader off: {self.error}", flush=True)
        threading.Thread(target=load, daemon=True, name="pd-ocr-load").start()

    def update(self, gray, region, products):
        """gray: detection picture; region: the zone in it (x0, y0, x1, y1);
        products: (N, 6) detection boxes in the zone (same picture)."""
        if gray is None:
            return
        now = time.monotonic()
        x0, y0, x1, y1 = region
        zone = gray[y0:y1, x0:x1]
        small = cv2.resize(zone if zone.ndim == 2 else cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY),
                           (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._prev is None or np.abs(small - self._prev).mean() > MOVED:
            self._still_since = now  # hands, a card being laid: wait
        self._prev = small
        for c in products[:, 5]:
            self._seen[int(c)] = now
        if self.name is not None and (
                now - self._shown_at > SHOW_S or
                (self._name_small is not None
                 and (np.abs(small - self._name_small) > 25).mean() > TAKEN)):
            # card or product taken out (even while the hand is still there),
            # or shown long enough: this picture is done, no new greeting
            self.name, self._name_small, self.card_quad = None, None, None
            self._tries = RETRIES
            self.version += 1
        if now - self._still_since < HOLD_S:
            return
        if self._busy or self.reader is None:
            return
        if self._read_small is not None and np.abs(small - self._read_small).mean() <= MOVED:
            # read this picture already: again only while no name came out
            if self.name is not None or self._tries >= RETRIES or now - self._last_read < RETRY_S:
                return
            self._tries += 1
        else:
            self._tries = 0
        self._last_read = now
        crop = zone.copy()
        for bx0, by0, bx1, by1 in products[:, :4].astype(int):  # the products: no text to read
            cv2.rectangle(crop, (bx0 - x0 - 4, by0 - y0 - 4), (bx1 - x0 + 4, by1 - y0 + 4),
                          int(np.median(crop)), -1)
        self._busy, self._read_small = True, small
        classes = tuple(sorted(c for c, t in self._seen.items() if now - t <= SEEN_S))
        threading.Thread(target=self._read, args=(crop, small, (x0, y0), classes), daemon=True,
                         name="pd-ocr").start()

    def _read(self, crop, small, offset, classes):
        try:
            from .ocr import name_from
            t = time.perf_counter()
            lines, rects = self.reader(crop)
            name = name_from(lines, rects)
            print(f"card: {name or 'no name'} in {time.perf_counter() - t:.1f} s "
                  f"({' | '.join(t for t, _ in lines if t)})", flush=True)
            if not name:
                self._greeted = None  # zone without a card: the next one is greeted
            elif name == self._greeted:
                pass  # still that card (only the product taken, or after SHOW_S)
            elif name != self.name:
                self._greeted = name
                self.name, self._name_small = name, small
                self._name_classes = classes
                self._shown_at = time.monotonic()
                self.card_quad = _card_outline(rects, lines, name, offset)
                self.version += 1
                if classes:  # a card alone is no lead: only card and product together
                    self._save(crop, name, classes, lines)
                else:
                    print("card: no product in the zone -- not saved", flush=True)
        except Exception as error:  # noqa: BLE001
            print(f"card reader: {type(error).__name__}: {error}", flush=True)
        finally:
            self._busy = False

    def _save(self, crop, name, classes, lines):
        try:
            os.makedirs(self.leads_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            arts = [self.labels[c] if c < len(self.labels) else str(c) for c in classes]
            cv2.imwrite(os.path.join(self.leads_dir, f"{stamp}_{'+'.join(arts)}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            path = os.path.join(self.leads_dir, "leads.csv")
            new = not os.path.exists(path)
            with open(path, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter=";")
                if new:
                    w.writerow(["Zeit", "Vorname", "Nachname", "Artikelnummer", "Produkt",
                                "Text der Karte"])
                for art in arts:
                    w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name[0], name[1], art,
                                self.names.get(art, art), " | ".join(t for t, _ in lines if t)])
        except OSError as error:
            print(f"lead not saved: {error}", flush=True)

    def product_classes(self):
        """The products last detected in the zone while the name was read."""
        return self._name_classes if self.name is not None else ()

    def banner(self):
        return None if self.name is None else self.name


def _line_axis(rect):
    """Unit vector along a text line (pointing right)."""
    p = cv2.boxPoints(rect)
    e1, e2 = p[1] - p[0], p[2] - p[1]
    u = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    u = u / max(np.hypot(*u), 1e-6)
    return -u if u[0] < 0 else u


def _card_outline(rects, lines, name, offset):
    """The card around its text: the lines near the name line and parallel
    to it (a card's print all runs one way; the module's, if any was read,
    lies apart or turned), as a rectangle along the name line with the
    card's usual margins -> (4, 2) picture pixels."""
    at = [i for i, (t, _) in enumerate(lines) if name[1] in t]
    if not rects or not at:
        return None
    name_rect = rects[at[0]]
    c = np.float32(name_rect[0])
    u = _line_axis(name_rect)
    v = np.float32([-u[1], u[0]])
    reach = 14 * max(min(name_rect[1]), 4)  # a card is ~15 name heights wide
    pts = []
    for r in rects:
        if np.hypot(*(np.float32(r[0]) - c)) < reach and abs(float(_line_axis(r) @ u)) > 0.985:
            pts.append(cv2.boxPoints(r))
    pts = np.concatenate(pts).astype(np.float32) - c
    a, b = pts @ u, pts @ v
    (a0, a1), (b0, b1) = (a.min(), a.max()), (b.min(), b.max())
    ma, mb = 0.07 * (a1 - a0), 0.12 * (b1 - b0)  # print keeps clear of the edge
    a0, a1, b0, b1 = a0 - ma, a1 + ma, b0 - mb, b1 + mb
    quad = np.float32([c + a0 * u + b0 * v, c + a1 * u + b0 * v,
                       c + a1 * u + b1 * v, c + a0 * u + b1 * v])
    return quad + np.float32(offset)


def zone_in_picture(zone, to_screen, size, step=12):
    """The zone (screen pixels) in the detection picture: its points that
    the screen mapping puts into the zone -> bounding box (x0, y0, x1, y1).
    to_screen: (N, 6) boxes in the picture -> the same on the screen."""
    w, h = size
    xs, ys = np.meshgrid(np.arange(0, w, step, dtype=np.float32),
                         np.arange(0, h, step, dtype=np.float32))
    pts = np.stack([xs.ravel(), ys.ravel()], 1)
    boxes = np.zeros((len(pts), 6), np.float32)
    boxes[:, :2], boxes[:, 2:4] = pts, pts + 1
    on = to_screen(boxes)
    cx, cy = (on[:, 0] + on[:, 2]) / 2, (on[:, 1] + on[:, 3]) / 2
    inside = (cx >= zone[0]) & (cx <= zone[2]) & (cy >= zone[1]) & (cy <= zone[3])
    if not inside.any():
        return None
    p = pts[inside]
    return (int(max(0, p[:, 0].min())), int(max(0, p[:, 1].min())),
            int(min(w, p[:, 0].max() + step)), int(min(h, p[:, 1].max() + step)))
