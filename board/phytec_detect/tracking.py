"""Which products someone has picked up from the table.

The demo table holds many products that mostly lie still, so the tracker
keeps an inventory of where each one rests. Two independent cues mark a
product as picked up:

  taken   it rested at a spot, is no longer detected there, and the spot is
          empty now: it looks like the table that was remembered around it.
          Works when it is carried out of view -- the empty spot says which
          product is gone. An arm reaching across other products covers
          their surrounding table too, so they count as covered, not taken.
  lifted  it is still in view but clearly larger than while resting: the
          camera looks straight down, a product lifted towards it grows.
          Measured as the dark-pixel area inside the box (a box alone
          changes with rotation: a square turned 45 degrees has a 2x box).

A taken product stays in the list until it comes back: detected again at
its spot, or another product of the same class settling down anywhere --
or a product of another class settling where the taken one was last seen
(look-alikes like AM62x and AM62x DSC swap class when put down).
Short detection dropouts do nothing, because the spot is not empty then.
"""

import time

import cv2
import numpy as np

REST_S = 2.0          # still this long -> resting, its spot is remembered
STILL = 0.06          # max centre motion between frames (fraction of box size)
MISSING_S = 0.7       # resting product undetected this long -> check its spot
EMPTY_S = 0.5         # spot must look empty this long before "taken"
LIFT_ON = 1.45        # area ratio to count as lifted ...
LIFT_OFF = 1.2        # ... and to count as put down again
LIFT_BOX = 1.25       # the box must have grown too (a hand next to a product
                      # grows its dark area, not its box; turning it on the
                      # table grows the box, not the area)
CONFIRM = 3           # consecutive frames for a lifted/put-down change
FORGET_S = 1.5        # a never-resting track unseen this long is dropped
AREA_EVERY_S = 0.5    # a still product's dark area is re-measured this often
                      # (only to keep learning its resting size) -- every
                      # frame while it could be lifted or is lifted
TAKEN_MAX_S = 120.0   # give up on a taken product after this long (info panel closes;
                      # a guest's module that never comes back must not stay forever)


def dark_area(gray, box):
    """Pixel area of the product inside `box` on the light table.

    The threshold comes from the box plus a margin of table (Otsu needs both
    product and table; a tight box is all product). The pixels are counted
    inside the box only -- counted in the margin too, a hand or sleeve next
    to the product merged with it and made it look lifted."""
    h, w = gray.shape[:2]
    bx0, by0, bx1, by1 = box
    mx, my = 0.15 * (bx1 - bx0), 0.15 * (by1 - by0)
    x0, y0, x1, y1 = (int(round(v)) for v in (bx0 - mx, by0 - my, bx1 + mx, by1 + my))
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    crop = gray[y0:y1, x0:x1]
    step = max(1, int(max(crop.shape) / 160))  # measure on <= ~160 px
    small = crop[::step, ::step]
    thres, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ix0, iy0 = max(int(bx0), 0), max(int(by0), 0)
    ix1, iy1 = min(int(bx1), w), min(int(by1), h)
    inner = gray[iy0:iy1:step, ix0:ix1:step]
    if inner.size == 0:
        return 0.0
    mask = (inner <= thres).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max()) * step * step


def _box_area(box):
    return max(float(box[2] - box[0]), 1.0) * max(float(box[3] - box[1]), 1.0)


def _region_stats(gray, box, grow_in, grow_out):
    """Mean and std of the pixels between `box` grown by grow_in and by
    grow_out (fractions of its size); grow_in=None means the whole inner box."""
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0

    def rect(g):
        return (int(max(x0 - g * bw, 0)), int(max(y0 - g * bh, 0)),
                int(min(x1 + g * bw, w)), int(min(y1 + g * bh, h)))
    ox0, oy0, ox1, oy1 = rect(grow_out)
    if ox1 - ox0 < 4 or oy1 - oy0 < 4:
        return None
    step = max(1, int(max(ox1 - ox0, oy1 - oy0) / 160))
    crop = gray[oy0:oy1:step, ox0:ox1:step].astype(np.float32)
    mask = np.ones(crop.shape, bool)
    if grow_in is not None:
        ix0, iy0, ix1, iy1 = rect(grow_in)
        mask[(iy0 - oy0) // step:(iy1 - oy0) // step, (ix0 - ox0) // step:(ix1 - ox0) // step] = False
    vals = crop[mask]
    return (float(vals.mean()), float(vals.std())) if vals.size else None


def table_ring(gray, box):
    """The table just around a product: (mean, std) of a ring outside it."""
    return _region_stats(gray, box, 0.15, 0.45)


def spot_state(gray, box, ring_ref):
    """'empty', 'occupied' or 'covered' for the spot a product rested on.

    Empty means the table is visible again: the ring around the spot still
    looks like the table did (so no arm or hand lies across it) and the spot
    itself now looks like that table -- same brightness, no structure. An
    arm reaching over a product covers the ring as well (it is bigger than
    the product), which makes the spot 'covered': nothing is decided then.
    """
    if ring_ref is None:
        return "covered"
    ring_mean, ring_std = ring_ref
    ring = table_ring(gray, box)
    inner = _region_stats(gray, box, None, -0.1)
    if ring is None or inner is None:
        return "covered"
    if abs(ring[0] - ring_mean) > 0.12 * ring_mean or ring[1] > 2.5 * ring_std + 6:
        return "covered"
    if abs(inner[0] - ring[0]) < 0.12 * ring[0] and inner[1] < 2.0 * ring_std + 8:
        return "empty"
    return "occupied"


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Track:
    _next = 1

    def __init__(self, det, area, now):
        self.id, Track._next = Track._next, Track._next + 1
        self.cls = int(det[5])
        self.box = det[:4].copy()
        self.score = float(det[4])
        self.area = area
        self.area_at = now          # when `area` was measured
        self.rest_area = None       # learnt while still
        self.rest_box = None        # where it rests
        self.rest_box_area = None   # its box area while resting (kept when lifted)
        self.ring = None            # (mean, std) of the table around rest_box
        self.empty_since = None     # spot looked empty since
        self.still_since = now
        self.lifted = False
        self.taken = False
        self.picked_at = None       # when it became lifted/taken
        self.put_down = False       # was lifted itself, now settling again
        self._votes = 0
        self.seen = now

    @property
    def picked(self):
        return self.lifted or self.taken

    @property
    def resting(self):
        return self.rest_box is not None and not self.picked

    @property
    def centre(self):
        return (self.box[:2] + self.box[2:4]) / 2


class TableTracker:
    def __init__(self):
        self.last_update = None
        self.tracks = []
        self.class_rest = {}   # cls -> median resting area of that class

    def update(self, gray, dets, now=None):
        """Feed one frame's detections (pixels of `gray`); returns the tracks."""
        now = time.monotonic() if now is None else now
        self.last_update = now
        live = [t for t in self.tracks if not t.taken]
        pairs = sorted(((_iou(t.box, dets[j, :4]), i, j)
                        for i, t in enumerate(live) for j in range(len(dets))
                        if t.cls == int(dets[j, 5])), reverse=True)
        used_t, used_d = set(), set()
        for iou, i, j in pairs:
            if iou < 0.2 or i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            self._step(live[i], dets[j], gray, now)
        for j in range(len(dets)):
            if j not in used_d:
                self._new_detection(dets[j], gray, now)

        for t in self.tracks:
            if t.taken or t.seen == now or t.rest_box is None or now - t.seen <= MISSING_S:
                t.empty_since = None
                continue
            if spot_state(gray, t.rest_box, t.ring) != "empty":
                t.empty_since = None  # still there, or covered by an arm: wait
            elif t.empty_since is None:
                t.empty_since = now
            elif now - t.empty_since >= EMPTY_S:
                t.taken, t.lifted = True, False
                t.picked_at = t.picked_at or now
        # Keep: taken products (for a while), resting products (a resting one
        # whose spot is not empty is just covered or missed), and recent ones.
        self.tracks = [t for t in self.tracks
                       if (t.taken and now - t.picked_at < TAKEN_MAX_S)
                       or (not t.taken and t.rest_box is not None and now - t.seen < TAKEN_MAX_S)
                       or (not t.taken and now - t.seen < FORGET_S)]
        self._learn_class_rest()
        return self.tracks

    def _new_detection(self, det, gray, now):
        cls = int(det[5])
        # The product of a taken track coming back to its spot?
        for t in self.tracks:
            if t.taken and t.cls == cls and t.rest_box is not None and \
                    _iou(t.rest_box, det[:4]) > 0.3:
                t.taken, t.picked_at, t.seen = False, None, now
                t.box = det[:4].copy()
                return
        self.tracks.append(Track(det, dark_area(gray, det[:4]), now))

    def _step(self, t, det, gray, now):
        size = max(det[2] - det[0], det[3] - det[1], 1.0)
        moved = np.linalg.norm((det[:2] + det[2:4]) / 2 - t.centre) / size
        t.box, t.score, t.seen = det[:4].copy(), float(det[4]), now
        if moved > STILL:
            t.still_since = now
        ref = t.rest_area or self.class_rest.get(t.cls)
        # Lifted needs the box grown by LIFT_BOX: while it has not grown (and
        # the product is not lifted) the area cannot change the decision, so
        # it is measured only now and then to keep learning the resting size.
        box_ref = _box_area(t.rest_box) if t.rest_box is not None else t.rest_box_area
        measure = (t.lifted or ref is None or not t.area or not box_ref
                   or _box_area(t.box) / box_ref >= LIFT_BOX
                   or now - t.area_at >= AREA_EVERY_S)
        if not measure:
            t._votes = 0  # same as the full check: not lifted, stays so
            return
        t.area, t.area_at = dark_area(gray, t.box), now
        plausible_rest = ref is None or t.area < LIFT_OFF * ref
        if not t.lifted and moved < STILL and plausible_rest and t.area:
            # learn what this product looks like on the table -- never from a
            # size that already looks lifted (held still in the air)
            t.rest_area = t.area if t.rest_area is None else 0.8 * t.rest_area + 0.2 * t.area
            if now - t.still_since >= REST_S:
                if t.rest_box is None and not t.put_down:
                    self._put_back(t.cls, t.box)
                t.put_down = False
                if t.rest_box is None or t.ring is None or _iou(t.rest_box, t.box) < 0.9:
                    t.ring = table_ring(gray, t.box)  # the table as it is around it
                t.rest_box = t.box.copy()
                t.rest_box_area = _box_area(t.box)
        if ref is None or not t.area:
            return
        ratio = t.area / ref
        box_ref = _box_area(t.rest_box) if t.rest_box is not None else t.rest_box_area
        box_ratio = _box_area(t.box) / box_ref if box_ref else LIFT_BOX
        want = (ratio > LIFT_ON and box_ratio >= LIFT_BOX) if not t.lifted else ratio > LIFT_OFF
        if want != t.lifted:
            t._votes += 1
            if t._votes >= CONFIRM:
                t.lifted, t._votes = want, 0
                t.picked_at = now if want else None
                if not want:
                    # put down: wait until it rests again; settling then is
                    # this product returning, not another one coming back
                    t.rest_box, t.put_down, t.still_since = None, True, now
        else:
            t._votes = 0

    def _put_back(self, cls, box):
        """A product of class `cls` settled on the table at `box`: if one of
        that class was taken, it has been put back. Otherwise a taken one
        last seen right there (within its size) is the same product,
        recognised as a look-alike now."""
        taken = [t for t in self.tracks if t.taken and t.cls == cls]
        if not taken:
            centre = (box[:2] + box[2:4]) / 2
            taken = [t for t in self.tracks if t.taken and np.linalg.norm(
                t.centre - centre) < max(t.box[2] - t.box[0], t.box[3] - t.box[1], 1.0)]
        if taken:
            self.tracks.remove(min(taken, key=lambda t: t.picked_at))

    def _learn_class_rest(self):
        by_cls = {}
        for t in self.tracks:
            if t.rest_area and t.resting:
                by_cls.setdefault(t.cls, []).append(t.rest_area)
        for cls, areas in by_cls.items():
            self.class_rest[cls] = float(np.median(areas))

    def picked(self):
        """Lifted or taken products, most recently picked first."""
        return sorted((t for t in self.tracks if t.picked), key=lambda t: -t.picked_at)


