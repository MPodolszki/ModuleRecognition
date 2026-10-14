"""Collect training pictures with the phyCAM the detector will later run on.

Shows the live camera fullscreen on HDMI and saves a full-resolution JPEG
every --interval seconds (or only on Enter with --interval 0). The saved
pictures come out of exactly the same capture path as the live demo, so
the training data matches what the model sees at run time.

    phytec-capture -o /home/root/capture --interval 1.0 --count 200
"""

import argparse
import os
import select
import signal
import sys
import threading
import time

import cv2

from .camera import Camera, Screen, die, label_frame


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phytec-capture", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out-dir", default="/home/root/capture")
    parser.add_argument("-p", "--prefix", default="phycam",
                        help="file name prefix (default: phycam)")
    parser.add_argument("-i", "--interval", type=float, default=1.0,
                        help="seconds between pictures; 0 = only on Enter")
    parser.add_argument("-n", "--count", type=int, default=0,
                        help="stop after this many pictures (0 = until Ctrl+C)")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--source", default="csi1")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    existing = [f for f in os.listdir(args.out_dir) if f.startswith(args.prefix)]
    index = len(existing)

    try:
        cam = Camera(args.source)
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
                cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved, last_save = saved + 1, now
                print(path, flush=True)
                if args.count and saved >= args.count:
                    break

            if not args.no_display:
                if screen is None:
                    screen = Screen(frame.shape[1], frame.shape[0])
                shown = frame.copy()
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
