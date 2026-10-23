"""Live module detection on HDMI.

Two-camera setup (default when a live camera is found on CSI2):

  detection camera (CSI1, VM-017 mono, 2592x1944) -> tiles -> backend(s)
  live camera      (CSI2, VM-016 colour)          -> screen, with the
                   detections mapped in through the table-plane
                   calibration (phytec-calibrate)

Without a live camera or calibration the detection camera itself is shown.

    phytec-detect                         # Hailo-8 | NPU side by side
    phytec-detect -b hailo                # Hailo-8 only, fullscreen
    phytec-detect --tile-scale 0.5        # quarter of the tiles, coarser
    phytec-detect --tile-scale 0          # no tiling: whole frame -> 640
    phytec-detect -m model -m coco        # models stacked on one backend
    phytec-detect --detect-cam clip.mp4   # a recording instead of the camera

Each backend runs on its own thread, always on the newest detection frame;
the screen runs at the live camera's rate and shows each backend's latest
detections, so the slower accelerator visibly lags.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
import types

import cv2
import numpy as np

from .backends import BACKENDS, NAMES
from .card_desk import zone_in_picture
from .calib import PlaneMap, frame_to_sensor, sensor_to_frame
from .camera import (Camera, OverlayScreen, Screen, die, find_sensor, isp_size,
                     screen_size, to_bgrx)
from .detector import Detector
from .info_panel import InfoPanel
from .postprocess import Smoother, draw, module_quad
from . import style
from .search import SearchState
from .tiling import Tiler, find_image_circle
from .tracking import TableTracker, _iou

DEFAULT_MODEL_DIR = "/usr/share/phytec-detect/model"
DEFAULT_CALIB = "/etc/phytec-detect/plane_map.json"
DEFAULT_CALIB_ISP = "/etc/phytec-detect/plane_map_isp.json"
# The detection picture, shown without a live camera, is scaled straight to
# the screen width: scaling to 1296 and letting the G2D blow it up to 1920
# again made it look softer than the camera is.


class FrameSource:
    """Reads a camera on its own thread; .latest() is the newest frame.

    on_demand: read a new frame only after the last one was taken. For the
    5 MP detection camera, reading every frame at 47 fps kept the Python
    side busy copying pictures nobody used; its appsink drops stale frames
    anyway, so a read on demand is still the newest picture.
    """

    def __init__(self, cam, on_demand=False):
        self.cam = cam
        self.frame, self.seq, self.fps = None, 0, 0.0
        self.error = None
        self._on_demand = on_demand
        self._taken = threading.Event()
        self._taken.set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=_named("pd-camera", self._loop), daemon=True)
        self._thread.start()

    def _loop(self):
        t0 = time.perf_counter()
        while not self._stop.is_set():
            if self._on_demand and not self._taken.wait(0.1):
                continue
            self._taken.clear()
            ok, frame = self.cam.read()
            if not ok:
                self.error = "could not read from camera"
                return
            self.frame, self.seq = frame, self.seq + 1
            self.fps = self.seq / (time.perf_counter() - t0)

    def latest(self):
        return self.frame, self.seq

    def take(self):
        """latest() for a consumer that processes the frame: lets an
        on-demand source fetch the next one meanwhile."""
        self._taken.set()
        return self.frame, self.seq

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)
        self.cam.release()


class Worker:
    """One detector, always on the source's newest frame, in two stages:
    a prepare thread cuts and letterboxes the next frame's tiles while the
    run thread waits for the accelerator and decodes the previous one --
    the Cortex-A53s and the accelerator work at the same time.

    A third thread tracks the detections (which products are lifted,
    tracking.py) and finds their turned outlines, one frame behind, so the
    accelerator does not wait for it."""

    def __init__(self, detector, source, tiler_factory, track_process=False):
        self.detector, self.source, self._tiler_factory = detector, source, tiler_factory
        self.tracker = TableTracker()
        # tracking in a process of its own (track_proc.py), else in a thread here
        self.track_proc = None
        if track_process:
            from .track_proc import TrackProcess
            self.track_proc = TrackProcess()
        self.snaps = {}  # class -> last still picture of a picked-up product (BGR)
        # The accelerator reports down to 0.2, the smoother decides what is
        # shown (at the demo's threshold): a dip below it is not a dropout.
        self.smoother, self.show_conf = Smoother(), detector.conf
        detector.conf = min(detector.conf, 0.2)
        self.tiler = None
        self.result = None  # (frame, detections, sequence number, lifted, turned outlines)
        self.fps = 0.0
        self.error = None
        self._done = []
        self._job = None  # (frame, seq, prepared) handed from prepare to run
        self._cond = threading.Condition()
        self._track_job = None  # (frame, seq, dets) handed from run to track; newest wins
        self._track_cond = threading.Condition()
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=_named("pd-prepare", self._prepare_loop),
                                          daemon=True),
                         threading.Thread(target=_named("pd-run", self._run_loop), daemon=True),
                         threading.Thread(target=_named("pd-track", self._track_loop),
                                          daemon=True)]
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        with self._track_cond:
            self._track_cond.notify_all()
        for t in self._threads:
            t.join(timeout=5)
        if self.track_proc is not None:
            self.track_proc.close()

    def _prepare_loop(self):
        last = -1
        while not self._stop.is_set():
            with self._cond:  # one prepared frame in flight at most
                while self._job is not None and not self._stop.is_set():
                    self._cond.wait(0.1)
            frame, seq = self.source.take()
            if frame is None or seq == last:
                time.sleep(0.002)
                continue
            last = seq
            if self.tiler is None:
                self.tiler = self._tiler_factory(frame)
            job = (frame, seq, self.detector.prepare(frame, self.tiler))
            with self._cond:
                self._job = job
                self._cond.notify_all()

    def _run_loop(self):
        """Two frames in flight: the next one is started on the accelerator
        before the previous one is decoded, so chip and CPU work at the same
        time. Every frame is decoded exactly as before; when no new frame is
        ready soon, the pending one is finished right away."""
        pending = None  # (frame, seq, submitted)
        while not self._stop.is_set():
            with self._cond:
                deadline = time.monotonic() + (0.005 if pending else 0.1)
                while self._job is None and not self._stop.is_set() \
                        and time.monotonic() < deadline:
                    self._cond.wait(max(0.0, deadline - time.monotonic()))
                job, self._job = self._job, None
                self._cond.notify_all()
            try:
                started = None
                if job is not None:
                    frame, seq, prepared = job
                    started = (frame, seq, self.detector.submit(prepared))
                if pending is None:
                    pending = started
                    continue
                frame, seq, submitted = pending
                pending = started
                dets = self.detector.finish(submitted)
            except Exception as error:  # e.g. HailoRTTimeout: let main() exit
                self.error = error
                return
            if self.smoother is not None:
                dets = self.smoother(dets, self.show_conf)
            now = time.perf_counter()
            self._done = (self._done + [now])[-10:]
            if len(self._done) > 1:
                self.fps = (len(self._done) - 1) / (self._done[-1] - self._done[0])
            # Show the new boxes right away, at the accelerator's rate: the
            # turned outlines and the lifted state of the last tracked frame
            # are carried over (outlines moved with their box); tracking
            # catches up a frame or two later.
            last = self.result
            if last is not None:
                self.result = (frame, dets, seq, last[3], carry_quads(last[1], last[4], dets))
            with self._track_cond:
                self._track_job = (frame, seq, dets)
                self._track_cond.notify_all()

    def _quads(self, gray, dets):
        """Turned outlines (module_quad) -- reused from the previous frame for
        a product whose box and picture have not changed: a 16x16 thumbnail
        of the box, compared to the last one, says whether anything did
        (a hand, a shadow, a turn), and then it is measured again."""
        cache, out, new_cache = getattr(self, "_quad_cache", []), [], []
        for d in dets:
            x0, y0, x1, y1 = (int(round(v)) for v in d[:4])
            crop = gray[max(y0, 0):max(y1, 1), max(x0, 0):max(x1, 1)]
            thumb = cv2.resize(crop, (16, 16), interpolation=cv2.INTER_AREA).astype(np.int16) \
                if crop.size else None
            quad = None
            for c_box, c_cls, c_thumb, c_quad in cache:
                if c_cls == int(d[5]) and np.abs(c_box - d[:4]).max() <= QUAD_BOX_PX \
                        and thumb is not None and c_thumb is not None \
                        and np.abs(thumb - c_thumb).mean() <= QUAD_THUMB_DIFF:
                    quad = c_quad  # same product, same place, same picture
                    thumb, box = c_thumb, c_box  # compare against when it was measured
                    break
            else:
                box = d[:4].copy()
            if quad is None:
                quad = module_quad(gray, d)
                box = d[:4].copy()
            out.append(quad)
            new_cache.append((box, int(d[5]), thumb, quad))
        self._quad_cache = new_cache
        return np.array(out, np.float32).reshape(-1, 4, 2)

    def _snapshot(self, gray, dets, quads):
        _snap_track(self, gray, dets, quads)

    def _track_loop(self):
        while not self._stop.is_set():
            with self._track_cond:
                while self._track_job is None and not self._stop.is_set():
                    self._track_cond.wait(0.1)
                job, self._track_job = self._track_job, None
            if job is None:
                continue
            frame, seq, dets = job
            try:
                t0 = time.perf_counter()
                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                if self.track_proc is not None:
                    lifted, quads, _ = self.track_proc.run(gray, dets)
                    self.snaps = self.track_proc.snaps
                else:
                    self.tracker.update(gray, dets)
                    lifted = np.array([[*t.box, t.score, t.cls] for t in self.tracker.picked()],
                                      np.float32).reshape(-1, 6)
                    quads = self._quads(gray, dets)
                    self._snapshot(gray, dets, quads)
                self.detector.times["track"].append(time.perf_counter() - t0)
            except Exception as error:
                self.error = error
                return
            self.result = (frame, dets, seq, lifted, quads)


QUAD_BOX_PX = 2.0      # box corners moved at most this much (detection jitter)
QUAD_THUMB_DIFF = 3.0  # mean grey-level change of the 16x16 thumbnail


def _snap_track(worker, gray, dets, quads):
    """One still picture per resting product, taken once when it has come
    to rest (again only after it was moved or turned): when it is picked
    up, the info panel shows how it lay on the table instead of the
    reference photo."""
    for t in worker.tracker.tracks:
        if not t.resting or t.seen != worker.tracker.last_update:
            continue
        old = getattr(t, "snap_box", None)
        if old is not None and _iou(old, t.rest_box) >= 0.9:
            continue  # already taken here
        j = next((j for j, d in enumerate(dets)
                  if int(d[5]) == t.cls and np.allclose(d[:4], t.box)), None)
        if j is not None:
            t.snap, t.snap_box = straight_crop(gray, quads[j]), t.rest_box.copy()
    worker.snaps = {t.cls: t.snap for t in worker.tracker.picked()
                    if getattr(t, "snap", None) is not None}


def _named(name, target):
    """Thread target that sets its OS thread name (top -H, /proc/*/task/*/comm)
    -- to see which stage uses the CPU."""
    def run():
        try:
            import ctypes
            ctypes.CDLL(None).prctl(15, name.encode()[:15], 0, 0, 0)  # PR_SET_NAME
        except (OSError, AttributeError):
            pass
        target()
    return run


def carry_quads(old_dets, old_quads, dets):
    """Outlines for new boxes from the previous ones: same class and
    overlapping -> the old outline shifted by the box centre's move; else
    the box corners."""
    out = np.empty((len(dets), 4, 2), np.float32)
    for i, (x0, y0, x1, y1, _, cls) in enumerate(dets):
        out[i] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        best, best_iou = None, 0.3
        for j, (a0, b0, a1, b1, _, c) in enumerate(old_dets):
            if int(c) != int(cls):
                continue
            iw = min(x1, a1) - max(x0, a0)
            ih = min(y1, b1) - max(y0, b0)
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            iou = inter / ((x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter)
            if iou > best_iou:
                best, best_iou = j, iou
        if best is not None:
            a0, b0, a1, b1 = old_dets[best][:4]
            shift = [(x0 + x1 - a0 - a1) / 2, (y0 + y1 - b0 - b1) / 2]
            out[i] = old_quads[best] + shift
    return out


def load_products(model_dirs):
    """products.json of the model directories (tools/product_info.py)."""
    info = {}
    for d in model_dirs:
        path = os.path.join(d, "products.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.values():  # charts, pictures: relative to the model dir
                for key in ("chart", "image"):
                    if entry.get(key):
                        entry[key] = os.path.join(d, entry[key])
            info.update(data)
    return info


def zone_rect(text, width, height):
    """'0.02,0.6,0.26,0.96' -> screen pixels (x0, y0, x1, y1); 'off' -> None."""
    if not text or text == "off":
        return None
    x0, y0, x1, y1 = (float(v) for v in text.split(","))
    return int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)


def in_zone(dets, zone):
    """Indices of the detections whose centre lies in the zone."""
    if zone is None or not len(dets):
        return []
    cx, cy = (dets[:, 0] + dets[:, 2]) / 2, (dets[:, 1] + dets[:, 3]) / 2
    return list(np.flatnonzero((cx >= zone[0]) & (cx <= zone[2]) &
                               (cy >= zone[1]) & (cy <= zone[3])))


def straight_crop(frame, quad, out_w=300):
    """The product inside its turned outline (detection camera), turned
    straight with a margin, long side horizontal -- the panel's live picture."""
    rect = cv2.minAreaRect(quad.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h, angle = h, w, angle + 90
    w, h = w * 1.2, h * 1.2
    if w < 4 or h < 4:
        return None
    out_h = max(1, int(round(out_w * h / w)))
    box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.float32)
    dst = np.float32([[0, out_h], [0, 0], [out_w, 0], [out_w, out_h]])
    crop = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(box, dst), (out_w, out_h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop[..., :3]


def draw_zone(canvas, zone, busy, color=style.ORANGE,
              text=("Produkt hier ablegen", "Erkannt")):
    """A drop zone: dashed frame and a label, solid while something lies in it."""
    x0, y0, x1, y1 = zone
    bgra = color + (255,)
    if busy:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), bgra, 5)
    else:
        for a, b, horizontal in (((x0, y0), (x1, y0), True), ((x0, y1), (x1, y1), True),
                                 ((x0, y0), (x0, y1), False), ((x1, y0), (x1, y1), False)):
            length = (b[0] - a[0]) if horizontal else (b[1] - a[1])
            for t in range(0, length, 36):
                e = min(t + 20, length)
                p = (a[0] + t, a[1]) if horizontal else (a[0], a[1] + t)
                q = (a[0] + e, a[1]) if horizontal else (a[0], a[1] + e)
                cv2.line(canvas, p, q, bgra, 4)
    tag = style.text_patch(text[1] if busy else text[0], 20, "Medium", bg=color, pad=(10, 5))
    style.blit(canvas, tag, x0, y0 - tag.shape[0])


CARD_ZONE_TEXT = ("Produkt + Visitenkarte hier ablegen", "Erkannt")
DROP_ZONE_TEXT = ("Produkt hier ablegen \u00b7 zwei zum Vergleich", "Erkannt")


def draw_card(canvas, quad, name):
    """The business card's outline in blue, tagged with the person's name."""
    pts = np.round(quad).astype(np.int32)
    cv2.polylines(canvas, [pts], True, style.BLUE + (255,), 4, cv2.LINE_AA)
    top = pts[np.argsort(pts[:, 1])[:2]]
    x, y = int(top[:, 0].min()), int(top[:, 1].min())
    tag = style.text_patch(f"{name[0]} {name[1]}", 20, "Medium", bg=style.BLUE, pad=(10, 5))
    style.blit(canvas, tag, x, max(0, y - tag.shape[0]))


def draw_banner(canvas, name, fraction):
    """'Thank you <name>' in PHYTEC petrol across the top of the info panel."""
    w = canvas.shape[1]
    x0 = w - int(w * fraction)
    first = style.text_patch(f"Thank you {name[0]} {name[1]}", 36, "Bold", pad=(28, 0))
    second = style.text_patch("We will contact you shortly.", 26, "Regular", pad=(28, 0))
    h = 22 + first.shape[0] + 6 + second.shape[0] + 22
    y0 = style.HEADER_H
    cv2.rectangle(canvas, (x0, y0), (w - 1, y0 + h), style.PETROL + (255,), -1)
    cv2.rectangle(canvas, (x0, y0 + h - 4), (w - 1, y0 + h), style.WHITE + (255,), -1)
    style.blit(canvas, first, x0, y0 + 22)
    style.blit(canvas, second, x0, y0 + 22 + first.shape[0] + 6)


def display_labels(labels, products):
    return [products.get(l, {}).get("name") or l for l in labels]


def to_view(dets, det_cam, view_cam, plane):
    """Detection-camera frame boxes -> live-camera frame boxes, through
    sensor pixels and the table-plane mapping."""
    if not len(dets):
        return dets
    d = dets.copy()
    d[:, :2] = frame_to_sensor(d[:, :2], det_cam)
    d[:, 2:4] = frame_to_sensor(d[:, 2:4], det_cam)
    d = plane.boxes(d)
    d[:, :2] = sensor_to_frame(d[:, :2], view_cam)
    d[:, 2:4] = sensor_to_frame(d[:, 2:4], view_cam)
    return d


def quads_to_view(quads, det_cam, view_cam, plane):
    """Turned outlines (N, 4, 2) from the detection to the live picture."""
    if not len(quads):
        return quads
    pts = frame_to_sensor(quads.reshape(-1, 2), det_cam)
    pts = sensor_to_frame(plane(pts), view_cam)
    return pts.reshape(-1, 4, 2).astype(np.float32)


def _backend_text(worker):
    det = worker.detector
    name = getattr(det, "display_name", None) or NAMES[det.backend_name]
    tiles = f", {len(worker.tiler.tiles)} tiles" if worker.tiler and len(worker.tiler.tiles) > 1 else ""
    net = f" + {det.ms('net'):.0f} ms LAN" if getattr(det, "show_net", False) else ""
    # (no ms per frame otherwise: with two frames in flight the time from
    # start to result is latency, not what a frame costs)
    return f"{name}  {worker.fps:4.1f} fps{net}{tiles}"


def _bar_text(workers):
    """Header text: every backend with its speed, then the class count."""
    workers = workers if isinstance(workers, (list, tuple)) else [workers]
    n = workers[0].detector.nc  # classes of the running model(s) = products it was trained on
    return "  |  ".join(_backend_text(w) for w in workers) + \
        f"  \u00b7  {n} Produkt{'e' if n != 1 else ''} trainiert"


def _aspect(text):
    """'16:9' -> 1.777..., None -> None."""
    if not text:
        return None
    w, _, h = text.partition(":")
    return float(w) / float(h or 1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phytec-detect", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-b", "--backend", action="append", choices=BACKENDS,
                        help="backend(s) to show, one or two (default: hailo + npu)")
    parser.add_argument("-m", "--model-dir", action="append",
                        help=f"model directory, repeatable to stack models on one "
                             f"backend (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--hef", help="explicit HEF (default: *_mz.hef, else *.hef)")
    parser.add_argument("-c", "--conf", type=float, default=0.4,
                        help="minimum score drawn (default: 0.4)")
    parser.add_argument("--detect-cam", default="csi1", help="csi1/csi2 or a video file")
    parser.add_argument("--aspect", default=None,
                        help="read a full-width band of this aspect from the detection "
                             "sensor, e.g. 16:9 to fill the screen (default: whole sensor)")
    parser.add_argument("--crop-y", type=float, default=0.5,
                        help="where that band sits: 0 top, 0.5 centre, 1 bottom")
    parser.add_argument("--view-cam", default="auto",
                        help="live camera: csi2, csi1, none, or auto (CSI2 if present)")
    parser.add_argument("--calib", default=DEFAULT_CALIB,
                        help=f"table-plane mapping from phytec-calibrate ({DEFAULT_CALIB})")
    parser.add_argument("--isp", action="store_true",
                        help="both cameras through the i.MX8MP ISP in its current mode (e.g. "
                             "1920x1080 each) instead of raw; mapping from "
                             "phytec-calibrate --isp (--aspect/--crop-y do not apply)")
    parser.add_argument("--tile-scale", type=float, default=1.0,
                        help="scale before tiling (1 = full resolution, 0.5 = half; "
                             "0 = no tiling, whole frame to 640)")
    parser.add_argument("--box-labels", action="store_true",
                        help="write the product name at every frame (default: frames only)")
    parser.add_argument("--voice", nargs="?", const="Jabra", default=None, metavar="MIC",
                        help="voice commands ('Phytec, wo ist der i.MX6?') with Whisper on the "
                             "Hailo-8, from this microphone (default: Jabra)")
    parser.add_argument("--axis-boxes", action="store_true",
                        help="plain axis-aligned boxes instead of frames turned like the module")
    parser.add_argument("--det-fps", type=int, default=20,
                        help="with --isp: frames per second the ISP delivers for detection "
                             "(the accelerator takes ~16; every frame read costs CPU), 0 = 30")
    parser.add_argument("--lift", action="store_true",
                        help="picking a product up shows its info (thick orange frame, still "
                             "picture); default: only the zones do -- one product laid in "
                             "the orange zone its info, two the comparison")
    parser.add_argument("--track-thread", action="store_true",
                        help="tracking in a thread of the demo instead of a process of its own "
                             "(on the Hailo board: ~1-2 fps more detection, but tracking "
                             "170 instead of 88 ms -- half the frames tracked)")
    parser.add_argument("--in-process", action="store_true",
                        help="run the accelerator in this process (default: a process of "
                             "its own per backend, ~50 %% more fps on the Hailo-8)")
    parser.add_argument("--remote", metavar="HOST[:PORT]",
                        help="for -b remote: the board running "
                             "'python3 -m phytec_detect.remote serve' (its accelerator, our camera)")
    parser.add_argument("--warn-below", type=float, default=0.8,
                        help="frame and tag in orange below this score (0 = off)")
    parser.add_argument("--quality-dir", default="/var/lib/phytec-detect/quality",
                        help="record unsure / confused detections there (quality.py), "
                             "'' = off")
    parser.add_argument("--isp-width", type=int, default=0,
                        help="with --isp: the ISP scales the detection picture to this width, "
                             "e.g. 960 (~52 ms less CPU per frame, own calibration "
                             "plane_map_isp_w960.json). Off by default until a model is "
                             "trained on ISP-scaled pictures: products_v6_n (CPU-scaled "
                             "training) lost confidence on some boards with it")
    parser.add_argument("--zone", default="0.02,0.6,0.26,0.96",
                        help="drop zone x0,y0,x1,y1 as fractions of the screen (a product laid "
                             "there shows its info at once, with its live picture), 'off' = none")
    parser.add_argument("--card-zone", default="0.02,0.12,0.26,0.48",
                        help="card zone x0,y0,x1,y1 as fractions of the screen (blue: a product "
                             "and a business card laid there), 'off' = none")
    parser.add_argument("--ocr-models", default="/opt/phytec-detect/models/ocr",
                        help="business card reader models (phytec_detect.ocr) for the card zone")
    parser.add_argument("--leads-dir", default="/var/lib/phytec-detect/leads",
                        help="where the card zone saves the visitors' names and cards")
    parser.add_argument("--no-panel", action="store_true",
                        help="never slide in product info (products.json) for lifted products")
    parser.add_argument("-n", "--frames", type=int, help="stop after N screen frames")
    args = parser.parse_args(argv)

    backends = args.backend or ["hailo", "npu"]
    if len(backends) > 2:
        die("at most two backends side by side")

    if args.isp and args.calib == DEFAULT_CALIB:
        from .calibrate_cli import isp_map_path
        args.calib = isp_map_path(args.isp_width) if args.isp_width else DEFAULT_CALIB_ISP
    view_cam = args.view_cam
    if view_cam == "auto":
        view_cam = "csi2" if find_sensor(2) and args.detect_cam != "csi2" else "none"
    plane = None
    if view_cam != "none":
        if os.path.exists(args.calib):
            plane = PlaneMap.load(args.calib)
        else:
            print(f"no calibration at {args.calib} -- showing the detection camera; "
                  f"run phytec-calibrate", file=sys.stderr)
            view_cam = "none"

    def tiler_factory(frame):
        if args.tile_scale <= 0:
            return None
        return Tiler(frame.shape, scale=args.tile_scale, circle=find_image_circle(frame)
                     if frame.ndim == 2 else None)

    model_dirs = args.model_dir or [DEFAULT_MODEL_DIR]
    products = load_products(model_dirs)

    workers, sources = [], []
    screen = None
    if args.isp and view_cam != "none":
        # Live picture first: its compositor needs large contiguous DMA
        # buffers, which are hard to get once the cameras, the Hailo and
        # Whisper have broken the CMA area up ("failed to activate bufferpool").
        # Whisper alone takes ~230 MB of it, so wait until the pipeline runs
        # (its pools are allocated) before anything else starts.
        for attempt in range(5):
            screen = OverlayScreen(int(view_cam[-1]), isp_size(int(view_cam[-1])) or (1920, 1080))
            screen.show(np.zeros((screen.size[1], screen.size[0], 4), np.uint8))
            t0 = time.monotonic()
            while screen.frames < 5 and not screen.error and time.monotonic() - t0 < 5:
                time.sleep(0.05)
            if screen.frames >= 5 and not screen.error:
                break
            print(f"live picture: {screen.error or 'no frames'} -- try {attempt + 2}", flush=True)
            screen.close()
            time.sleep(2)
        else:
            die("live picture does not start", code=3)
        t_screen = time.monotonic()
    try:
        if args.isp:
            det_cam = Camera(args.detect_cam, isp=True, isp_width=args.isp_width or None,
                             isp_fps=args.det_fps or None)
        else:
            det_cam = Camera(args.detect_cam, aspect=_aspect(args.aspect), crop_y=args.crop_y)
        det_source = FrameSource(det_cam, on_demand=True)
        sources.append(det_source)
        view_source = view_geo = None
        if view_cam != "none" and args.isp:
            # live picture: ISP -> G2D -> screen, never through Python; only
            # its geometry is needed to place the frames
            w, h = isp_size(int(view_cam[-1])) or (1920, 1080)
            view_geo = types.SimpleNamespace(crop=((2592 - w) // 2, (1944 - h) // 2, w, h),
                                             scale=screen_size()[0] / w, size=(w, h))
        elif view_cam != "none":
            view_source = FrameSource(Camera(view_cam, crop=plane.view_crop))
            sources.append(view_source)
        for name in backends:
            print(f"loading {NAMES[name]} ...", flush=True)
            if name == "remote":
                from .remote import RemoteDetector
                if not args.remote:
                    raise ValueError("-b remote needs --remote host[:port]")
                det = RemoteDetector(args.remote, conf=args.conf)
            elif args.in_process:
                det = Detector(name, model_dirs, conf=args.conf, hef=args.hef)
            else:  # its own process: no GIL shared with drawing and tracking
                from .remote import spawn_local
                det = spawn_local(name, model_dirs, hef=args.hef, conf=args.conf)
            print(f"  {', '.join(det.paths)} ({det.load_s:.1f} s)", flush=True)
            workers.append(Worker(det, det_source, tiler_factory,
                                  track_process=not (args.track_thread or args.in_process)))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        for w in workers:
            w.stop()
            w.detector.close()
        for s in sources:
            s.stop()
        die(f"phytec-detect: {error}")

    listener = None

    def start_voice():
        try:
            from .whisper_voice import Listener
            return Listener(args.voice, log=lambda m: print(m, flush=True))
        except Exception as error:  # noqa: BLE001 -- the demo runs without voice
            print(f"voice: not available: {type(error).__name__}: {error}", flush=True)
            return None
    # Whisper takes ~230 MB of CMA: with the live picture on the G2D it starts
    # only once that has settled (its pools re-allocate when the window goes
    # fullscreen, and then failed for lack of contiguous memory).
    voice_pending = bool(args.voice)
    if voice_pending and screen is None:
        listener, voice_pending = start_voice(), False

    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    snap = threading.Event()  # kill -USR1: save the screen to /tmp/phytec-detect.png
    signal.signal(signal.SIGUSR1, lambda *_: snap.set())

    labels = workers[0].detector.labels
    search = SearchState([(l, products.get(l, {}).get("name") or l) for l in labels])
    resets = 0
    canvas = panel = None
    zone = card_zone = card_region = cards = None
    banner_seen = None
    zone_classes, live_result = [], None
    screen_tries = 0
    frames, shown_seq, last_draw, last_state = 0, -1, 0.0, None
    sig, last_sig = [], None
    quality, quality_result = None, None
    if args.quality_dir and workers[0].smoother is not None:
        try:
            from .quality import QualityRecorder
            quality = QualityRecorder(args.quality_dir, workers[0].detector.labels,
                                      {a: p.get("name", a) for a, p in products.items()})
        except OSError as error:
            print(f"quality recorder off: {error}", flush=True)
    timing = os.environ.get("PHYTEC_DETECT_TIMING") == "1"
    tsum = {}

    def tick(name, t0):
        if timing:
            tsum[name] = tsum.get(name, 0.0) + time.perf_counter() - t0
        return time.perf_counter()
    try:
        while not stop.is_set() and (args.frames is None or frames < args.frames):
            for s in sources:
                if s.error:
                    die(s.error)
            if voice_pending and screen.frames > 150 and not screen.error \
                    and time.monotonic() - t_screen > 5:
                listener, voice_pending = start_voice(), False
            if view_geo is not None and screen is not None and screen.error:
                # right after a restart the camera or the display may still be
                # held by the previous run: a few new tries before giving up
                screen_tries += 1
                if screen_tries > 5:
                    die(f"live picture: {screen.error}", code=3)
                print(f"live picture: {screen.error} -- try {screen_tries} again", flush=True)
                screen.close()
                time.sleep(2)
                screen = OverlayScreen(int(view_cam[-1]), view_geo.size)
                t_screen = time.monotonic()
            for w in workers:
                if w.error is not None:
                    # Exit non-zero so systemd (Restart=on-failure) starts over
                    # instead of showing a picture without detections.
                    die(f"{NAMES[w.detector.backend_name]} failed: "
                        f"{type(w.error).__name__}: {w.error}", code=3)

            if view_geo is not None:  # overlay layer only, at most ~25 times a second
                # 25 a second while the panel slides, else 15: the frames change
                # only with the detector (~12/s), and every redraw holds the GIL
                # the detector thread needs (two results per frame: detector and
                # tracker)
                wait = (0.04 if panel is not None and panel.moving() else 0.066) \
                    - (time.monotonic() - last_draw)
                if wait > 0:
                    time.sleep(wait)
                # Redraw only when something changed: new detections, the
                # panel sliding, the voice dot, a search -- drawing the layer
                # costs the tracking thread GIL time.
                state = (tuple(id(w.result) for w in workers),
                         panel is not None and panel.moving(),
                         listener is not None and time.monotonic() < listener.awake_until,
                         search.text, int(time.monotonic()),  # and once a second anyway
                         cards.version if cards is not None else 0)
                if state == last_state:
                    time.sleep(0.005)  # nothing new: wait, do not spin (it held a
                    continue           # core and the GIL the detector threads need)
                last_state = state
                last_draw = time.monotonic()
                frame = None
            else:
                frame, seq = view_source.latest() if view_source else det_source.take()
                if frame is None or seq == shown_seq:
                    time.sleep(0.003)
                    continue
                shown_seq = seq
            t_ = time.perf_counter()
            if view_geo is not None:
                base = None
            elif view_source is None:  # detection picture itself, scaled for the screen
                view_scale = min(1.0, screen_size()[0] / frame.shape[1])
                base = to_bgrx(cv2.resize(frame, None, fx=view_scale, fy=view_scale,
                                          interpolation=cv2.INTER_AREA))
            else:
                base = frame
            if canvas is None:
                if view_geo is not None:
                    w_, h = screen.size
                    canvas = np.zeros((h, w_, 4), np.uint8)
                else:
                    h, w_ = base.shape[:2]
                    canvas = np.zeros((h, w_ * len(workers), 4), np.uint8)
                    screen = Screen(canvas.shape[1], h)
                panel = None if args.no_panel else InfoPanel(products, workers[0].detector.labels)
                zone = zone_rect(args.zone, w_, canvas.shape[0])
                card_zone = zone_rect(args.card_zone, w_, canvas.shape[0])
                if card_zone is not None and os.path.isdir(args.ocr_models):
                    from .card_desk import CardDesk
                    cards = CardDesk(workers[0].detector.labels,
                                     {a: p.get("name", a) for a, p in products.items()},
                                     args.ocr_models, args.leads_dir)
            # panel fully in and unchanged: its half of the layer stays as it is
            banner = cards.banner() if cards is not None else None
            keep_from = panel.static_from(canvas.shape[1]) \
                if panel is not None and base is None and len(workers) == 1 \
                and banner == banner_seen else None  # a new banner: the panel again
            banner_seen = banner
            for i, worker in enumerate(workers):
                if view_geo is not None and i > 0:
                    continue  # one live picture: frames of the first backend only
                half = canvas[:, i * w_:(i + 1) * w_] if keep_from is None \
                    else canvas[:, :keep_from]
                half[:] = 0 if base is None else base  # transparent layer, or the picture
                t_ = tick("clear", t_)
                if i == 0:
                    sig = []  # what is drawn, to skip pushing an unchanged layer
                result = worker.result
                if result is None:
                    style.header(half, f"{NAMES[worker.detector.backend_name]} startet \u2026")
                    continue
                dets, lifted, quads = result[1], result[3], result[4]
                if not args.lift:  # picking up does nothing: only the zones
                    lifted = lifted[:0]
                if i == 0 and quality is not None and result is not quality_result:
                    quality_result = result  # once per detection result
                    quality.observe(result[0], worker.smoother.spots)
                wanted = search.poll()  # "Wo ist der i.MX6?": only those, else all
                if search.resets != resets:  # "Reset": forget the table, start afresh
                    resets = search.resets
                    for w in workers:
                        w.tracker = TableTracker()
                        if w.track_proc is not None:
                            w.track_proc.reset()
                    print("reset: table forgotten", flush=True)
                if wanted is not None:
                    keep = np.isin(dets[:, 5].astype(int), list(wanted))
                    dets, quads = dets[keep], quads[keep]
                    lifted = lifted[np.isin(lifted[:, 5].astype(int), list(wanted))]
                det_quads = quads  # detection-camera outlines (before mapping to the view)
                det_dets = dets
                t_ = tick("search/reset", t_)
                if view_source is None and view_geo is None:
                    dets, lifted = dets.copy(), lifted.copy()
                    dets[:, :4] *= view_scale
                    lifted[:, :4] *= view_scale
                    quads = quads * view_scale
                else:
                    view = view_geo or view_source.cam
                    quads = quads_to_view(quads, det_source.cam, view, plane)
                    dets, lifted = (to_view(d, det_source.cam, view, plane)
                                    for d in (dets, lifted))
                t_ = tick("map to view", t_)
                if i == 0 and (zone is not None or card_zone is not None):
                    # both zones show a product's info at once, with its live picture
                    inside = in_zone(dets, zone)
                    in_card = in_zone(dets, card_zone)
                    shown_now = inside + [k for k in in_card if k not in inside]
                    zone_classes = [int(dets[k, 5]) for k in shown_now]
                    if panel is not None and shown_now and live_result is not result:
                        live_result = result  # one new crop per detection result
                        k = shown_now[0]
                        crop = straight_crop(result[0], det_quads[k]) \
                            if result[0] is not None and k < len(det_quads) else None
                        if crop is not None:
                            panel.set_live(int(dets[k, 5]), crop)
                    if zone is not None:
                        two = len({int(dets[k, 5]) for k in inside}) >= 2
                        draw_zone(half, zone, bool(inside), text=DROP_ZONE_TEXT[:1] + (
                            "Vergleich" if two else "Erkannt",))
                    if card_zone is not None:
                        draw_zone(half, card_zone, bool(in_card), style.BLUE, CARD_ZONE_TEXT)
                    if cards is not None and result[0] is not None:
                        if card_region is None:  # the zone in the detection picture, once
                            if view_source is None and view_geo is None:
                                def to_screen(b, s_=view_scale):
                                    b = b.copy()
                                    b[:, :4] *= s_
                                    return b
                            else:
                                def to_screen(b, v_=view):
                                    return to_view(b, det_source.cam, v_, plane)
                            card_region = zone_in_picture(
                                card_zone, to_screen, result[0].shape[1::-1]) or ()
                            print(f"card zone in the detection picture: {card_region}", flush=True)
                        if card_region:
                            cards.update(result[0], card_region, det_dets[in_card])
                        if cards.name is not None and cards.card_quad is not None:
                            q = cards.card_quad[None]
                            q = q * view_scale if view_source is None and view_geo is None \
                                else quads_to_view(q, det_source.cam, view, plane)
                            draw_card(half, q[0], cards.name)
                        if cards.name is not None:  # visitor greeted: their product's info,
                            zone_classes += [c for c in cards.product_classes()  # even when
                                             if c not in zone_classes]  # the card hides it
                    sig.append((bool(inside), bool(in_card)))
                t_ = tick("zone + live crop", t_)
                sig.append((np.round(dets).astype(np.int32).tobytes(),
                            np.round(quads).astype(np.int32).tobytes(),
                            np.round(lifted).astype(np.int32).tobytes()))
                draw(half, dets, display_labels(worker.detector.labels, products)
                     if args.box_labels else None, None if args.axis_boxes else quads,
                     phytec=True, warn_below=args.warn_below or None)
                t_ = tick("draw frames", t_)
                for x0, y0, x1, y1, _, _ in lifted:  # lifted: thick orange frame
                    cv2.rectangle(half, (int(x0) - 4, int(y0) - 4), (int(x1) + 4, int(y1) + 4),
                                  style.ORANGE + (255,), 7)
                if wanted is not None:
                    found = len(dets)
                    style.header(half, f"Suche: {search.text}  \u00b7  "
                                 + (f"{found} gefunden" if found else
                                    ("nicht auf dem Tisch" if wanted else "unbekanntes Produkt")),
                                 accent=style.ORANGE)
                else:
                    # one picture, one set of frames (the first backend's):
                    # the others are compared in the header
                    text = _bar_text(workers if view_geo is not None else worker)
                    style.header(half, text)
                    sig.append(text)
            t_ = tick("lifted + header", t_)
            if panel is not None:
                primary = workers[0].result
                snaps = workers[0].snaps if args.lift else {}
                for cls in list(panel._live):  # left the zone / put back: the photo again
                    if cls not in zone_classes and cls not in snaps and \
                            cls not in panel.shown():  # (not on one missed frame)
                        panel.set_live(cls, None)
                for cls, still in snaps.items():  # (not `snap`: that is the USR1 event)
                    if cls not in zone_classes and panel.picture_of(cls) is not still:
                        panel.set_live(cls, still, tag="AUFGENOMMEN")
                panel.update([int(c) for c in primary[3][:, 5]] if primary and args.lift else [],
                             zone_classes)
                t_ = tick("panel update", t_)
                panel.compose(canvas, intact=keep_from is not None)
                sig.append(panel.state())
            if banner is not None:
                draw_banner(canvas, banner, panel.fraction if panel is not None else 0.5)
                sig.append(banner)
                t_ = tick("panel compose", t_)
            sig.append((search.text, listener is not None and time.monotonic() < listener.awake_until))
            if listener is not None and time.monotonic() < listener.awake_until:
                # "Phytec" heard: red dot top right while it listens for the command
                cx, cy = canvas.shape[1] - 50, 100
                cv2.circle(canvas, (cx, cy), 30, (0, 0, 220, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 30, (255, 255, 255, 255), 3, cv2.LINE_AA)
            if snap.is_set():
                snap.clear()
                cv2.imwrite("/tmp/phytec-detect.png", canvas[..., :3])
                if view_geo is not None:  # what the screen really shows
                    screen.grab_to = "/tmp/phytec-detect-screen.png"
                if (view_source or view_geo) is not None and det_source.frame is not None:
                    cv2.imwrite("/tmp/phytec-detect-cam.png", det_source.frame)
            if view_geo is not None and sig == last_sig and frames:
                continue  # same layer as on screen: no 8 MB copy and conversion
            last_sig = sig
            screen.show(canvas)
            t_ = tick("show (push)", t_)
            frames += 1
            if timing and frames % 60 == 0:
                print("main loop per pushed frame: " + ", ".join(
                    f"{k} {v / 60 * 1000:.1f}" for k, v in tsum.items()) + " ms", flush=True)
                tsum.clear()

            if frames % 60 == 0:
                print(f"detect cam {det_source.fps:4.1f} fps"
                      + (f"  live cam {view_source.fps:4.1f} fps" if view_source else "")
                      + (f"  live picture {screen.frames / max(time.monotonic() - t_screen, 1e-3):4.1f} fps (G2D)"
                         if view_geo is not None and screen is not None else "")
                      + "  " + "  ".join(
                          f"{NAMES[w.detector.backend_name]} {w.fps:4.1f} fps "
                          f"[{len(w.tiler.tiles) if w.tiler else 1} tiles] "
                          f"(pre {w.detector.ms('pre'):.0f} / infer {w.detector.ms('infer'):.0f}"
                          f" / post {w.detector.ms('post'):.0f} / track {w.detector.ms('track'):.0f} ms)"
                          for w in workers),
                      flush=True)
    finally:
        if listener is not None:
            listener.stop()
        for w in workers:
            w.stop()
            w.detector.close()
        for s in sources:
            s.stop()
        if screen is not None:
            screen.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
