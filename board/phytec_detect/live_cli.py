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
from .calib import PlaneMap, frame_to_sensor, sensor_to_frame
from .camera import (Camera, OverlayScreen, Screen, die, find_sensor, isp_size, label_frame,
                     screen_size, to_bgrx)
from .detector import Detector
from .info_panel import InfoPanel
from .postprocess import draw, module_quad
from .search import SearchState
from .tiling import Tiler, find_image_circle
from .tracking import TableTracker

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
        self._thread = threading.Thread(target=self._loop, daemon=True)
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

    def __init__(self, detector, source, tiler_factory):
        self.detector, self.source, self._tiler_factory = detector, source, tiler_factory
        self.tracker = TableTracker()
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
        self._threads = [threading.Thread(target=self._prepare_loop, daemon=True),
                         threading.Thread(target=self._run_loop, daemon=True),
                         threading.Thread(target=self._track_loop, daemon=True)]
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
        while not self._stop.is_set():
            with self._cond:
                while self._job is None and not self._stop.is_set():
                    self._cond.wait(0.1)
                job, self._job = self._job, None
                self._cond.notify_all()
            if job is None:
                continue
            frame, seq, prepared = job
            try:
                dets = self.detector.run(prepared)
            except Exception as error:  # e.g. HailoRTTimeout: let main() exit
                self.error = error
                return
            now = time.perf_counter()
            self._done = (self._done + [now])[-10:]
            if len(self._done) > 1:
                self.fps = (len(self._done) - 1) / (self._done[-1] - self._done[0])
            with self._track_cond:
                self._track_job = (frame, seq, dets)
                self._track_cond.notify_all()

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
                self.tracker.update(gray, dets)
                lifted = np.array([[*t.box, t.score, t.cls] for t in self.tracker.picked()],
                                  np.float32).reshape(-1, 6)
                quads = np.array([module_quad(gray, d) for d in dets],
                                 np.float32).reshape(-1, 4, 2)
                self.detector.times["track"].append(time.perf_counter() - t0)
            except Exception as error:
                self.error = error
                return
            self.result = (frame, dets, seq, lifted, quads)


def load_products(model_dirs):
    """products.json of the model directories (tools/product_info.py)."""
    info = {}
    for d in model_dirs:
        path = os.path.join(d, "products.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                info.update(json.load(f))
    return info


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


def _bar_text(worker):
    det = worker.detector
    tiles = f"{len(worker.tiler.tiles)} tiles, " if worker.tiler and len(worker.tiler.tiles) > 1 else ""
    n = det.nc  # classes of the running model(s) = products it was trained on
    return (f"{NAMES[det.backend_name]}: {worker.fps:4.1f} fps  "
            f"({tiles}inference {det.ms('infer'):.0f} ms)  |  "
            f"{n} Produkt{'e' if n != 1 else ''} trainiert")


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
    parser.add_argument("--no-panel", action="store_true",
                        help="never slide in product info (products.json) for lifted products")
    parser.add_argument("-n", "--frames", type=int, help="stop after N screen frames")
    args = parser.parse_args(argv)

    backends = args.backend or ["hailo", "npu"]
    if len(backends) > 2:
        die("at most two backends side by side")

    if args.isp and args.calib == DEFAULT_CALIB:
        args.calib = DEFAULT_CALIB_ISP
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
    try:
        if args.isp:
            det_cam = Camera(args.detect_cam, isp=True)
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
            det = Detector(name, model_dirs, conf=args.conf, hef=args.hef)
            print(f"  {', '.join(det.paths)} ({det.load_s:.1f} s)", flush=True)
            workers.append(Worker(det, det_source, tiler_factory))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        for w in workers:
            w.stop()
            w.detector.close()
        for s in sources:
            s.stop()
        die(f"phytec-detect: {error}")

    listener = None
    if args.voice:
        try:
            from .whisper_voice import Listener
            listener = Listener(args.voice, log=lambda m: print(m, flush=True))
        except Exception as error:  # noqa: BLE001 -- the demo runs without voice
            print(f"voice: not available: {type(error).__name__}: {error}", flush=True)

    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    snap = threading.Event()  # kill -USR1: save the screen to /tmp/phytec-detect.png
    signal.signal(signal.SIGUSR1, lambda *_: snap.set())

    labels = workers[0].detector.labels
    search = SearchState([(l, products.get(l, {}).get("name") or l) for l in labels])
    resets = 0
    screen = canvas = panel = None
    frames, shown_seq, last_draw = 0, -1, 0.0
    try:
        while not stop.is_set() and (args.frames is None or frames < args.frames):
            for s in sources:
                if s.error:
                    die(s.error)
            if view_geo is not None and screen is not None and screen.error:
                die(f"live picture: {screen.error}", code=3)
            for w in workers:
                if w.error is not None:
                    # Exit non-zero so systemd (Restart=on-failure) starts over
                    # instead of showing a picture without detections.
                    die(f"{NAMES[w.detector.backend_name]} failed: "
                        f"{type(w.error).__name__}: {w.error}", code=3)

            if view_geo is not None:  # overlay layer only, at most ~25 times a second
                wait = 0.04 - (time.monotonic() - last_draw)
                if wait > 0:
                    time.sleep(wait)
                last_draw = time.monotonic()
                frame = None
            else:
                frame, seq = view_source.latest() if view_source else det_source.take()
                if frame is None or seq == shown_seq:
                    time.sleep(0.003)
                    continue
                shown_seq = seq
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
                    screen = OverlayScreen(int(view_cam[-1]), view_geo.size)
                    t_screen = time.monotonic()
                    w_, h = screen.size
                    canvas = np.zeros((h, w_, 4), np.uint8)
                else:
                    h, w_ = base.shape[:2]
                    canvas = np.zeros((h, w_ * len(workers), 4), np.uint8)
                    screen = Screen(canvas.shape[1], h)
                panel = None if args.no_panel else InfoPanel(products, workers[0].detector.labels)
            for i, worker in enumerate(workers):
                half = canvas[:, i * w_:(i + 1) * w_]
                half[:] = 0 if base is None else base  # transparent layer, or the picture
                result = worker.result
                if result is None:
                    label_frame(half, f"{NAMES[worker.detector.backend_name]}: starting ...")
                    continue
                dets, lifted, quads = result[1], result[3], result[4]
                wanted = search.poll()  # "Wo ist der i.MX6?": only those, else all
                if search.resets != resets:  # "Reset": forget the table, start afresh
                    resets = search.resets
                    for w in workers:
                        w.tracker = TableTracker()
                    print("reset: table forgotten", flush=True)
                if wanted is not None:
                    keep = np.isin(dets[:, 5].astype(int), list(wanted))
                    dets, quads = dets[keep], quads[keep]
                    lifted = lifted[np.isin(lifted[:, 5].astype(int), list(wanted))]
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
                draw(half, dets, display_labels(worker.detector.labels, products)
                     if args.box_labels else None, None if args.axis_boxes else quads)
                for x0, y0, x1, y1, _, _ in lifted:  # lifted: thick yellow frame
                    cv2.rectangle(half, (int(x0) - 4, int(y0) - 4), (int(x1) + 4, int(y1) + 4),
                                  (0, 230, 255, 255), 7)
                if wanted is not None:
                    found = len(dets)
                    label_frame(half, f"Suche: {search.text}  -  "
                                + (f"{found} gefunden" if found else
                                   ("nicht auf dem Tisch" if wanted else "unbekanntes Produkt")),
                                color=(0, 230, 0))
                else:
                    label_frame(half, _bar_text(worker))
            if panel is not None:
                primary = workers[0].result
                panel.update([int(c) for c in primary[3][:, 5]] if primary else [])
                panel.compose(canvas)
            if listener is not None and time.monotonic() < listener.awake_until:
                # "Phytec" heard: red dot top right while it listens for the command
                cx, cy = canvas.shape[1] - 50, 100
                cv2.circle(canvas, (cx, cy), 30, (0, 0, 220, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 30, (255, 255, 255, 255), 3, cv2.LINE_AA)
            if snap.is_set():
                snap.clear()
                cv2.imwrite("/tmp/phytec-detect.png", canvas[..., :3])
                if (view_source or view_geo) is not None and det_source.frame is not None:
                    cv2.imwrite("/tmp/phytec-detect-cam.png", det_source.frame)
            screen.show(canvas)
            frames += 1

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
