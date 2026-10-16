"""Focus aid: live picture with a sharpness number, to set the lens by hand.

    phytec-focus                 # detection camera (CSI1)
    phytec-focus --source csi2   # live camera
    phytec-focus --roi 0.3       # measure on the central 30 % only
    phytec-focus --zoom 2        # centre detail twice enlarged
    phytec-focus --raw           # raw sensor picture through the ISI, no ISP

The screen shows the whole sensor picture on the left (the yellow frame
marks the detail) and the centre of the picture in sensor pixels on the
right -- 1:1, or enlarged with --zoom -- so the focus can be judged by eye.

Sharpness is the variance of the Laplacian (edge contrast) in the
measuring window, divided by the window's mean brightness, so it does not
jump when the auto exposure changes. Turn the lens until the number peaks:
the bar shows the current value against the best one seen so far (the best
resets with 'r' + Enter on the terminal). Put something with fine detail --
printed text, a PCB, the ArUco sheet -- on the table under the window.
"""

import argparse
import select
import signal
import sys
import threading

import cv2
import numpy as np

from .camera import Camera, Screen, die, screen_size, to_bgrx


def sharpness(gray):
    g = gray.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(lap.var() / max(g.mean(), 1.0))


def view(frame, value, best, zoom, path):
    """Screen-sized picture: whole frame left, centre detail right, sharpness
    bar under the whole frame."""
    sw, sh = screen_size()
    out = np.zeros((sh, sw, 4), np.uint8)
    h, w = frame.shape[:2]
    left_w = sw // 2
    scale = left_w / w
    small = to_bgrx(cv2.resize(frame, (left_w, int(h * scale)), interpolation=cv2.INTER_AREA))
    out[:small.shape[0], :left_w] = small
    # Centre detail: (sw - left_w) x sh screen pixels = that / zoom sensor pixels.
    dw, dh = (sw - left_w) // zoom, sh // zoom
    x0, y0 = max(0, (w - dw) // 2), max(0, (h - dh) // 2)
    detail = frame[y0:y0 + dh, x0:x0 + dw]
    if zoom > 1:
        detail = cv2.resize(detail, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    detail = to_bgrx(detail)
    out[:detail.shape[0], left_w:left_w + detail.shape[1]] = detail
    cv2.rectangle(out, (int(x0 * scale), int(y0 * scale)),
                  (int((x0 + dw) * scale), int((y0 + dh) * scale)), (0, 255, 255, 255), 2)
    cv2.line(out, (left_w, 0), (left_w, sh), (255, 255, 255, 255), 2)
    frac = value / best if best else 0
    color = (0, 255, 0, 255) if frac > 0.95 else (0, 200, 255, 255) if frac > 0.8 \
        else (0, 0, 255, 255)
    top = small.shape[0] + 40
    text = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, f"Schaerfe {value:6.1f}   bester {best:6.1f}", (20, top + 40), text,
                1.2, (255, 255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"{100 * frac:4.0f} %", (20, top + 130), text, 2.5, color, 5, cv2.LINE_AA)
    cv2.rectangle(out, (20, top + 160), (20 + int((left_w - 40) * min(frac, 1)), top + 200),
                  color, -1)
    cv2.rectangle(out, (20, top + 160), (left_w - 20, top + 200), (200, 200, 200, 255), 1)
    cv2.putText(out, f"{path}   rechts: Bildmitte {'1:1' if zoom == 1 else f'{zoom}x'} (Sensorpixel)",
                (20, top + 250), text, 0.9, (200, 200, 200, 255), 2, cv2.LINE_AA)
    return out


def _aspect(text):
    if not text:
        return None
    w, _, h = text.partition(":")
    return float(w) / float(h or 1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phytec-focus", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="csi1", help="csi1, csi2 or a video file")
    parser.add_argument("--roi", type=float, default=0.4,
                        help="measuring window, fraction of width/height (default 0.4)")
    parser.add_argument("--aspect", default=None, help="sensor band, e.g. 16:9")
    parser.add_argument("--crop-y", type=float, default=0.5)
    parser.add_argument("--zoom", type=int, default=1,
                        help="enlargement of the centre detail (default 1 = sensor pixels)")
    parser.add_argument("--raw", action="store_true",
                        help="raw sensor picture through the ISI instead of the ISP")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args(argv)

    try:
        cam = Camera(args.source, aspect=_aspect(args.aspect), crop_y=args.crop_y,
                     isp=not args.raw and args.source.startswith("csi"))
    except RuntimeError as error:
        die(str(error))
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    screen, best, history, n = None, 0.0, [], 0
    try:
        while not stop.is_set():
            ok, frame = cam.read()
            if not ok:
                die("could not read from camera")
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            h, w = gray.shape
            rw, rh = int(w * args.roi), int(h * args.roi)
            x0, y0 = (w - rw) // 2, (h - rh) // 2
            value = sharpness(gray[y0:y0 + rh, x0:x0 + rw])
            history = (history + [value])[-8:]  # smooth the sensor noise
            value = float(np.median(history))
            best = max(best, value)
            n += 1
            if select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline().strip() == "r":
                best = value
            if n % 15 == 0:
                print(f"sharpness {value:8.1f}   best {best:8.1f}   "
                      f"({100 * value / best if best else 0:5.1f} % of best)", flush=True)
            if args.no_display:
                continue
            shown = view(frame, value, best, args.zoom, "RAW (ISI)" if args.raw else "ISP")
            if screen is None:
                screen = Screen(shown.shape[1], shown.shape[0])
            screen.show(shown)
    finally:
        cam.release()
        if screen is not None:
            screen.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
