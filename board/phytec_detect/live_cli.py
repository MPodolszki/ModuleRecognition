"""Live phyCAM detection on HDMI: one backend fullscreen, or two side by side.

    phytec-detect                       # Hailo-8 | NPU side by side
    phytec-detect -b hailo              # Hailo-8 only, fullscreen
    phytec-detect -b hailo -b cpu       # any two backends
    phytec-detect --source clip.mp4     # a video file instead of the camera
    phytec-detect -m model -m coco      # the module model stacked on a COCO model

In side-by-side mode both backends get the newest camera frame on their
own thread. Each half shows the frame its backend last finished, so the
slower side visibly lags -- that, and the numbers in the bar (frame rate
and the accelerator's own inference time), is the comparison.
"""

import argparse
import signal
import sys
import threading
import time

import numpy as np

from .backends import BACKENDS, NAMES
from .camera import Camera, Screen, die, label_frame
from .detector import Detector
from .postprocess import draw

DEFAULT_MODEL_DIR = "/usr/share/phytec-detect/model"


class Worker:
    """One detector on its own thread, always on the newest frame."""

    def __init__(self, detector):
        self.detector = detector
        self.result = None  # (frame, detections, sequence number)
        self.fps = 0.0
        self.error = None  # exception that ended the worker thread
        self._done = []
        self._lock = threading.Lock()
        self._next = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame):
        with self._lock:
            self._next = frame

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self):
        runs = 0
        while not self._stop.is_set():
            with self._lock:
                frame, self._next = self._next, None
            if frame is None:
                time.sleep(0.002)
                continue
            try:
                dets = self.detector(frame)
            except Exception as error:  # e.g. HailoRTTimeout: let main() exit
                self.error = error
                return
            runs += 1
            now = time.perf_counter()
            self._done = (self._done + [now])[-15:]
            if len(self._done) > 1:
                self.fps = (len(self._done) - 1) / (self._done[-1] - self._done[0])
            self.result = (frame, dets, runs)


def _bar_text(worker):
    det = worker.detector
    return (f"{NAMES[det.backend_name]}: {worker.fps:4.1f} fps  "
            f"inference {det.ms('infer'):5.1f} ms")


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
    parser.add_argument("--source", default="csi1", help="csi1 or a video file")
    parser.add_argument("-n", "--frames", type=int, help="stop after N camera frames")
    args = parser.parse_args(argv)

    backends = args.backend or ["hailo", "npu"]
    if len(backends) > 2:
        die("at most two backends side by side")

    workers = []
    try:
        for name in backends:
            print(f"loading {NAMES[name]} ...", flush=True)
            det = Detector(name, args.model_dir or [DEFAULT_MODEL_DIR],
                           conf=args.conf, hef=args.hef)
            print(f"  {', '.join(det.paths)} ({det.load_s:.1f} s)", flush=True)
            workers.append(Worker(det))
        cam = Camera(args.source)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        for w in workers:
            w.stop()
            w.detector.close()
        die(f"phytec-detect: {error}")

    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    screen = canvas = None
    shown = [None] * len(workers)
    frames, start = 0, time.perf_counter()
    try:
        while not stop.is_set() and (args.frames is None or frames < args.frames):
            ok, frame = cam.read()
            if not ok:
                if cam.is_camera:
                    die("could not read from camera")
                break
            for w in workers:
                if w.error is not None:
                    # Exit non-zero so systemd (Restart=on-failure) starts over
                    # instead of showing a camera picture without detections.
                    die(f"{NAMES[w.detector.backend_name]} failed: "
                        f"{type(w.error).__name__}: {w.error}", code=3)
                w.submit(frame)
            frames += 1

            h, w_ = frame.shape[:2]
            if canvas is None:
                canvas = np.zeros((h, w_ * len(workers), 4), np.uint8)
                screen = Screen(canvas.shape[1], h)
            for i, worker in enumerate(workers):
                result = worker.result
                half = canvas[:, i * w_:(i + 1) * w_]
                if result is None:
                    half[:] = frame
                    label_frame(half, f"{NAMES[worker.detector.backend_name]}: starting ...")
                elif result[2] != shown[i]:
                    shown[i] = result[2]
                    half[:] = result[0]
                    draw(half, result[1], worker.detector.labels)
                    label_frame(half, _bar_text(worker))
            screen.show(canvas)

            if frames % 60 == 0:
                cam_fps = frames / (time.perf_counter() - start)
                print(f"camera {cam_fps:5.1f} fps  " + "  ".join(
                    f"{NAMES[w.detector.backend_name]} {w.fps:5.1f} fps "
                    f"(pre {w.detector.ms('pre'):.1f} / infer {w.detector.ms('infer'):.1f} / "
                    f"post {w.detector.ms('post'):.1f} ms)" for w in workers), flush=True)
    finally:
        for w in workers:
            w.stop()
            w.detector.close()
        cam.release()
        if screen is not None:
            screen.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
