"""Collect training pictures with the phyCAM the detector will later run on.

Shows the live camera fullscreen on HDMI and saves a full-resolution JPEG
every --interval seconds (or only on Enter with --interval 0). The saved
pictures come out of exactly the same capture path as the live demo, so
the training data matches what the model sees at run time: full-resolution
GRAY8 from the VM-017, white-balanced colour from a VM-016.

    phytec-capture -o /var/lib/phytec-detect/capture --interval 1.0 --count 200
"""

import argparse
import os
import select
import signal
import sys
import threading
import time

import cv2

from .camera import Camera, Screen, die, label_frame, to_bgrx


def _aspect(text):
    """'16:9' -> 1.777..., None -> None."""
    if not text:
        return None
    w, _, h = text.partition(":")
    return float(w) / float(h or 1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phytec-capture", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out-dir", default="/var/lib/phytec-detect/capture")
    parser.add_argument("-p", "--prefix", default="phycam",
                        help="file name prefix (default: phycam)")
    parser.add_argument("-i", "--interval", type=float, default=1.0,
                        help="seconds between pictures; 0 = only on Enter")
    parser.add_argument("-n", "--count", type=int, default=0,
                        help="stop after this many pictures (0 = until Ctrl+C)")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--isp", action="store_true",
                        help="through the ISP in its current mode, as phytec-detect --isp")
    parser.add_argument("--source", default="csi1", help="csi1, csi2 or a video file")
    parser.add_argument("--aspect", default=None,
                        help="read a full-width band of this aspect from the detection "
                             "sensor, e.g. 16:9 to fill the screen (default: whole sensor)")
    parser.add_argument("--crop-y", type=float, default=0.5,
                        help="where that band sits: 0 top, 0.5 centre, 1 bottom")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    existing = [f for f in os.listdir(args.out_dir) if f.startswith(args.prefix)]
    index = len(existing)

    try:
        cam = (Camera(args.source, isp=True) if args.isp else Camera(args.source, aspect=_aspect(args.aspect), crop_y=args.crop_y))
    except RuntimeError as error:
        die(str(error))

    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    screen = None
    saved, last_save = 0, 0.0
    try:
        while not stop.is_set():
            ok, frame = cam.read()
            if not ok:
                die("could not read from camera")
            now = time.monotonic()

            want = False
            if args.interval > 0:
                want = now - last_save >= args.interval
            elif select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                want = True

            if want:
                index += 1
                path = os.path.join(args.out_dir, f"{args.prefix}_{index:05d}.jpg")
                cv2.imwrite(path, frame if frame.ndim == 2 else frame[..., :3],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved, last_save = saved + 1, now
                print(path, flush=True)
                if args.count and saved >= args.count:
                    break

            if not args.no_display:
                scale = min(1.0, 1296 / frame.shape[1])
                shown = to_bgrx(cv2.resize(frame, None, fx=scale, fy=scale,
                                           interpolation=cv2.INTER_AREA))
                if screen is None:
                    screen = Screen(shown.shape[1], shown.shape[0])
                flash = now - last_save < 0.15
                label_frame(shown, f"capture: {saved} saved -> {args.out_dir}",
                            (0, 0, 255) if flash else (0, 255, 255))
                screen.show(shown)
    finally:
        cam.release()
        if screen is not None:
            screen.close()
    print(f"{saved} picture(s) saved to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
