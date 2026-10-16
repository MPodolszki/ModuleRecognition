"""Focus aid: live pictures with a sharpness number, to set the lenses by hand.

    phytec-focus                     # both cameras side by side (S/W | colour)
    phytec-focus --source csi1       # one camera on the whole screen
    phytec-focus --zoom 2            # centre detail twice enlarged
    phytec-focus --raw               # raw sensor pictures through the ISI, no ISP

Each camera gets its share of the screen, filled with the centre of its
picture in sensor pixels (1:1, or enlarged with --zoom), so the focus can
be judged by eye. A small picture of the whole frame sits in the corner,
the yellow frame in it marks the detail.

Pictures come through the i.MX8MP ISP (imx8-phycam-isp, with its current
mode and tuning files) unless --raw. With both cameras streaming, the ISP
runs them in 1920x1080 (a 1:1 window from the sensor centre): at the full
2592x1944 on both at once its pictures break up in vertical stripes. Sharpness is the variance of the Laplacian
(edge contrast) in the detail, divided by its mean brightness, so it does
not jump when the exposure changes. Turn the lens until the number peaks:
the bar shows the current value against the best one seen so far (the best
resets with 'r' + Enter on the terminal). Put something with fine detail --
printed text, a PCB, the ArUco sheet -- on the table under the centre.
"""

import argparse
import select
import signal
import sys
import threading
import time

import cv2
import numpy as np

from .camera import (Camera, Screen, die, find_sensor, is_mono, isp_size, screen_size,
                     to_bgrx)

NAMES = {"csi1": "S/W (CSI1)", "csi2": "Farbe (CSI2)"}


def sharpness(gray):
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    _, std = cv2.meanStdDev(lap)
    return float(std[0, 0] ** 2 / max(cv2.mean(gray)[0], 1.0))


def window(size, pane, zoom, center=None):
    """Detail of a (w, h) frame that fills a (w, h) pane at zoom, around
    center (default: the frame centre)."""
    (w, h), (pw, ph) = size, pane
    dw, dh = min(w, pw // zoom) & ~1, min(h, ph // zoom) & ~1
    cx, cy = center if center is not None else (w / 2, h / 2)
    x0 = int(min(max(cx - dw / 2, 0), w - dw)) & ~1
    y0 = int(min(max(cy - dh / 2, 0), h - dh)) & ~1
    return x0, y0, dw, dh


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)


def match_center(img, ref):
    """Where the centre of ref (a small whole frame of the other camera) lies
    in img, in img's relative coordinates (0..1) -- or None. Both cameras look
    at the same table; SIFT copes with their different scale and colour."""
    sift = cv2.SIFT_create(nfeatures=1500)
    ka, da = sift.detectAndCompute(_gray(img), None)
    kb, db = sift.detectAndCompute(_gray(ref), None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None
    pairs = cv2.BFMatcher().knnMatch(db, da, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None
    src = np.float32([kb[m.queryIdx].pt for m in good])
    dst = np.float32([ka[m.trainIdx].pt for m in good])
    hom, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if hom is None or inliers.sum() < 8:
        return None
    rh, rw = ref.shape[:2]
    x, y = cv2.perspectiveTransform(np.float32([[[rw / 2, rh / 2]]]), hom)[0, 0]
    ih, iw = img.shape[:2]
    if not (0 <= x < iw and 0 <= y < ih):
        return None
    return x / iw, y / ih


class FrameReader:
    """Camera -> (centre detail, small whole frame), cut on the CPU."""

    def __init__(self, cam, pane, zoom, thumb_w):
        self.cam, self.pane, self.zoom, self.thumb_w = cam, pane, zoom, thumb_w
        self.size, self.center, self.small = None, None, None

    def read(self):
        ok, frame = self.cam.read()
        if not ok:
            return None
        h, w = frame.shape[:2]
        self.size = (w, h)
        x0, y0, dw, dh = window(self.size, self.pane, self.zoom, self.center)
        self.small = frame[::4, ::4]  # for match_center
        step = max(1, w // self.thumb_w // 2)  # a full-frame INTER_AREA is too slow
        _, _, tw, th = thumb_rect(self.pane, self.thumb_w, self.size)
        thumb = cv2.resize(frame[::step, ::step], (tw, th), interpolation=cv2.INTER_AREA)
        return frame[y0:y0 + dh, x0:x0 + dw].copy(), thumb

    def release(self):
        self.cam.release()


def thumb_rect(pane, thumb_w, size):
    """Where the small whole frame sits in a pane: x, y, w, h."""
    (pw, ph), (w, h) = pane, size
    tw, th = thumb_w & ~1, (h * thumb_w // w) & ~1
    return pw - tw - 10, ph - th - 10, tw, th


class IspReader:
    """Camera through the ISP: the G2D cuts the detail, scales the
    whole frame and composes both into the finished pane, so only that small
    picture reaches the CPU (the full 2592x1944 BGRx frame alone took ~150 ms
    to read, and mapping G2D buffers in Python stalled everything)."""

    def __init__(self, csi, pane, zoom, thumb_w):
        size = isp_size(csi) or (2592, 1944)
        self.size, self.center = size, None
        (w, h), (pw, ph) = size, pane
        x0, y0, dw, dh = window(size, pane, zoom)
        tx, ty, tw, th = thumb_rect(pane, thumb_w, size)
        leaky = "queue leaky=downstream max-size-buffers=1"
        self._cap = cv2.VideoCapture(
            f"imxcompositor_g2d name=c "
            f"sink_0::xpos={(pw - dw * zoom) // 2} sink_0::ypos={(ph - dh * zoom) // 2} "
            f"sink_0::width={dw * zoom} sink_0::height={dh * zoom} sink_0::zorder=0 "
            f"sink_1::xpos={tx} sink_1::ypos={ty} sink_1::width={tw} sink_1::height={th} "
            f"sink_1::zorder=1 ! video/x-raw,format=BGRx,width={pw},height={ph} ! "
            f"appsink drop=true max-buffers=1 sync=false "
            f"v4l2src device=/dev/video-isp-csi{csi} ! "
            f"video/x-raw,format=YUY2,width={w},height={h} ! tee name=t "
            f"t. ! {leaky} ! videocrop left={x0} top={y0} right={w - x0 - dw} "
            f"bottom={h - y0 - dh} ! c.sink_0 "
            f"t. ! {leaky} ! c.sink_1", cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open the ISP on CSI{csi}")

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            return None
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        return frame, None

    def release(self):
        self._cap.release()


class Pane:
    """One camera: reader thread, sharpness of the shown detail, its picture."""

    def __init__(self, source, reader, width, height, zoom):
        self.source, self.reader = source, reader
        self.width, self.height, self.zoom = width, height, zoom
        self.frame, self.seq = None, 0
        self.value, self.best, self._history = 0.0, 0.0, []
        self.failed = False
        self._lock = threading.Lock()

    def run(self, stop):
        while not stop.is_set():
            try:
                got = self.reader.read()
            except Exception as error:  # noqa: BLE001 -- end the tool, not just this thread
                print(f"{self.source}: {error!r}", file=sys.stderr, flush=True)
                got = None
            if got is None:
                self.failed = True
                stop.set()
                return
            with self._lock:
                self.frame, self.seq = got, self.seq + 1

    def update(self):
        """Measure the newest detail; False if there is none yet."""
        with self._lock:
            got = self.frame
        if got is None:
            return False
        detail = got[0]
        if got[1] is None:  # composed pane: measure the detail above the small frame
            detail = detail[130:thumb_rect((self.width, self.height), self.width * 2 // 5,
                                           self.reader.size)[1] - 10]
        gray = detail if detail.ndim == 2 else cv2.cvtColor(detail, cv2.COLOR_BGRA2GRAY)
        self._history = (self._history + [sharpness(gray)])[-8:]  # smooth sensor noise
        self.value = float(np.median(self._history))
        self.best = max(self.best, self.value)
        self._shown = got
        return True

    def draw(self, out, x, label):
        detail, thumb = self._shown
        size = self.reader.size
        tx, ty, tw, th = thumb_rect((self.width, self.height), self.width * 2 // 5, size)
        if thumb is None:  # the G2D composed the pane already
            out[:, x:x + self.width] = to_bgrx(detail)[:self.height, :self.width]
        else:
            if self.zoom > 1:
                detail = cv2.resize(detail, None, fx=self.zoom, fy=self.zoom,
                                    interpolation=cv2.INTER_NEAREST)
            detail = to_bgrx(detail)[:self.height, :self.width]
            oy, ox = (self.height - detail.shape[0]) // 2, (self.width - detail.shape[1]) // 2
            out[oy:oy + detail.shape[0], x + ox:x + ox + detail.shape[1]] = detail
            out[ty:ty + th, x + tx:x + tx + tw] = to_bgrx(thumb)[:th, :tw]

        # mark the detail in the small whole frame
        scale = tw / size[0]
        x0, y0, dw, dh = window(size, (self.width, self.height), self.zoom,
                                self.reader.center)
        cv2.rectangle(out, (x + tx + int(x0 * scale), ty + int(y0 * scale)),
                      (x + tx + int((x0 + dw) * scale), ty + int((y0 + dh) * scale)),
                      (0, 255, 255, 255), 2)
        cv2.rectangle(out, (x + tx - 1, ty - 1), (x + tx + tw, ty + th), (255, 255, 255, 255), 1)

        # header: camera, value, bar
        frac = self.value / self.best if self.best else 0
        color = (0, 255, 0, 255) if frac > 0.95 else (0, 200, 255, 255) if frac > 0.8 \
            else (0, 0, 255, 255)
        cv2.rectangle(out, (x, 0), (x + self.width, 130), (0, 0, 0, 255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, label, (x + 20, 40), font, 1.1, (255, 255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"{100 * frac:3.0f} %", (x + self.width - 230, 60), font, 1.8, color,
                    4, cv2.LINE_AA)
        cv2.putText(out, f"Schaerfe {self.value:6.1f}   bester {self.best:6.1f}",
                    (x + 20, 80), font, 0.8, (200, 200, 200, 255), 2, cv2.LINE_AA)
        bar_w = self.width - 40
        cv2.rectangle(out, (x + 20, 98), (x + 20 + int(bar_w * min(frac, 1)), 120), color, -1)
        cv2.rectangle(out, (x + 20, 98), (x + 20 + bar_w, 120), (200, 200, 200, 255), 1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phytec-focus", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="csi1,csi2",
                        help="cameras side by side, comma separated (csi1, csi2 or video files)")
    parser.add_argument("--zoom", type=int, default=1,
                        help="enlargement of the centre detail (default 1 = sensor pixels)")
    parser.add_argument("--raw", action="store_true",
                        help="raw sensor pictures through the ISI instead of the ISP")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args(argv)

    # OpenCV's own worker threads fight the camera reader threads for the
    # four cores (the screen ran at ~3 fps); one thread per call is faster here.
    cv2.setNumThreads(1)
    sources = [s for s in args.source.split(",") if s]
    sw, sh = screen_size()
    pane_w = (sw // len(sources)) & ~1
    zoom, thumb_w = max(1, args.zoom), pane_w * 2 // 5
    panes = []
    try:
        for source in sources:
            isp = not args.raw and source in ("csi1", "csi2")
            found = find_sensor(int(source[-1])) if isp else None
            if found and not is_mono(found[0]):  # S/W reads faster as GRAY8 via OpenCV
                reader = IspReader(int(source[-1]), (pane_w, sh), zoom, thumb_w)
            else:
                reader = FrameReader(Camera(source, isp=isp), (pane_w, sh), zoom, thumb_w)
            panes.append(Pane(source, reader, pane_w, sh, zoom))
    except RuntimeError as error:
        for pane in panes:
            pane.reader.release()
        die(str(error))
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    snap = threading.Event()  # kill -USR1: save the screen to /tmp/phytec-focus.png
    signal.signal(signal.SIGUSR1, lambda *_: snap.set())
    threads = [threading.Thread(target=p.run, args=(stop,), daemon=True) for p in panes]
    for t in threads:
        t.start()

    path = "RAW (ISI)" if args.raw else "ISP"
    screen, n, last = None, 0, None
    t_print, seq_print = 0.0, [0] * len(panes)
    # The colour detail is cut on the G2D at the sensor centre; the S/W detail
    # (cut on the CPU) follows it, so both show the same spot of the table.
    fixed = next((p for p in panes if isinstance(p.reader, IspReader)), None)
    movable = [p for p in panes if isinstance(p.reader, FrameReader)] if fixed else []

    def align():  # SIFT takes ~0.5 s on the board: off the screen loop
        while not stop.wait(3.0):
            if getattr(fixed, "_shown", None) is None:
                continue
            tx, ty, tw, th = thumb_rect((fixed.width, fixed.height), fixed.width * 2 // 5,
                                        fixed.reader.size)
            ref = fixed._shown[0][ty:ty + th, tx:tx + tw].copy()
            for p in movable:
                if p.reader.small is not None:
                    rel = match_center(np.ascontiguousarray(p.reader.small), ref)
                    if rel is not None:
                        if p.reader.center is None:
                            print(f"{p.source}: detail moved onto the {fixed.source} centre, "
                                  f"at {100 * rel[0]:.0f} % / {100 * rel[1]:.0f} %", flush=True)
                        p.reader.center = (rel[0] * p.reader.size[0], rel[1] * p.reader.size[1])

    if movable:
        threading.Thread(target=align, daemon=True).start()
    out = np.zeros((sh, sw, 4), np.uint8)
    try:
        while not stop.is_set():
            seqs = tuple(p.seq for p in panes)
            if seqs == last:  # nothing new from any camera yet
                stop.wait(0.005)
                continue
            last = seqs
            if not all(p.update() for p in panes):
                continue
            n += 1
            if select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline().strip() == "r":
                for p in panes:
                    p.best = p.value
            if n % 15 == 0:
                now = time.monotonic()
                fps = 15 / (now - t_print) if t_print else 0.0
                cam_fps = [(p.seq - s0) / (now - t_print) if t_print else 0.0
                           for p, s0 in zip(panes, seq_print)]
                t_print, seq_print = now, [p.seq for p in panes]
                print(f"screen {fps:4.1f} fps  cams " +
                      "/".join(f"{f:4.1f}" for f in cam_fps) + "  " + "   ".join(f"{p.source}: sharpness {p.value:6.1f} best {p.best:6.1f} "
                                 f"({100 * p.value / p.best if p.best else 0:5.1f} %)"
                                 for p in panes), flush=True)
            if args.no_display:
                continue
            for i, p in enumerate(panes):
                p.draw(out, i * pane_w, f"{NAMES.get(p.source, p.source)}  {path}")
            for i in range(1, len(panes)):
                cv2.line(out, (i * pane_w, 0), (i * pane_w, sh), (255, 255, 255, 255), 3)
            if snap.is_set():
                snap.clear()
                cv2.imwrite("/tmp/phytec-focus.png", out[..., :3])
            if screen is None:
                screen = Screen(sw, sh)
            screen.show(out)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2)
        for p in panes:
            p.reader.release()
        if screen is not None:
            screen.close()
    if any(p.failed for p in panes):
        die("could not read from camera")
    return 0


if __name__ == "__main__":
    sys.exit(main())
