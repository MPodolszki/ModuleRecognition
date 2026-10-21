"""Photograph every product of products.csv that has no pictures yet.

    phytec-photo-queue                       # /var/lib/phytec-detect/products.csv
    phytec-photo-queue --csv other.csv

products.csv is the project's list (Artikelnummer;Bezeichnung), copied to
the board next to the captures. Each product without a finished
phytec-photo run (capture/<Artikelnummer>/DONE) is photographed through the
ISP, one after the other. New rows are all it takes, also while the queue
runs: the list is re-read before every product, and the monitor shows it on
the right. Between two products the monitor shows which one is next and
waits, as long as it takes, until someone lays it in the middle -- nobody
has to stay at the table. With everything done the queue keeps watching
the list (--once: it ends). A product with capture/<Artikelnummer>/SKIP
is left out. Stopping the queue stops the current product;
run it again to go on where it stopped.

--round 2 (3, ...) photographs the list again, e.g. the products the demo
recognises badly (quality/nachtrainieren.csv): its own marker DONE_R2, the
new pictures numbered on after the earlier ones.
"""

import argparse
import os
import sys
import time

from . import photo_cli
from .photo_cli import read_products

CAPTURE = "/var/lib/phytec-detect/capture"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phytec-photo-queue", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="/var/lib/phytec-detect/products.csv")
    parser.add_argument("--out", default=CAPTURE)
    parser.add_argument("--raw", action="store_true", help="raw path instead of the ISP")
    parser.add_argument("--round", type=int, default=1,
                        help="2, 3, ...: photograph the products again (marker DONE_R<n>)")
    parser.add_argument("--once", action="store_true",
                        help="end when the list is done instead of waiting for new rows")
    args = parser.parse_args(argv)

    done_marker = "DONE" if args.round == 1 else f"DONE_R{args.round}"

    def todo():  # re-read every time: the list may grow while the queue runs
        return [(a, n) for a, n in read_products(args.csv)
                if not any(os.path.exists(os.path.join(args.out, a, m))
                           for m in (done_marker, "SKIP"))]

    previous, waiting = "", False
    while True:
        pending = todo()
        if not pending:
            if args.once:
                print("all products of the list are photographed", flush=True)
                return 0
            if not waiting:
                print(f"all products of the list are photographed -- watching {args.csv} "
                      f"for new ones", flush=True)
                waiting = True
            time.sleep(10)
            continue
        waiting = False
        article, name = pending[0]
        print(f"next: {article} {name} ({len(pending)} to do)", flush=True)
        folder = os.path.join(args.out, article)
        shots = len([f for f in os.listdir(folder) if f.endswith(".jpg")]) \
            if os.path.isdir(folder) else 0
        if args.round > 1:  # only this round's pictures count for going on
            start_file = os.path.join(folder, f"R{args.round}_START")
            if not os.path.exists(start_file):
                os.makedirs(folder, exist_ok=True)
                with open(start_file, "w") as f:
                    f.write(str(shots))
            with open(start_file) as f:
                shots -= int(f.read().strip() or 0)
        # interrupted after the reference shot: go on where it stopped
        resume = ["--start", str(shots + 1)] if shots >= 1 else []
        rc = photo_cli.main(["--article", article, "--name", name, "--intro", "4",
                             "--wait-start", "--previous", previous, "--list-csv", args.csv,
                             "--out", args.out, "--done-marker", done_marker] + resume
                            + ([] if args.raw else ["--isp"]))
        if rc != 0:
            print("stopped", flush=True)
            return rc
        previous = name or article


if __name__ == "__main__":
    sys.exit(main())
