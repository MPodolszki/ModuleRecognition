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
the list (--once: it ends). Stopping the queue stops the current product;
run it again to go on (that product starts over).
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
    parser.add_argument("--once", action="store_true",
                        help="end when the list is done instead of waiting for new rows")
    args = parser.parse_args(argv)

    def todo():  # re-read every time: the list may grow while the queue runs
        return [(a, n) for a, n in read_products(args.csv)
                if not os.path.exists(os.path.join(args.out, a, "DONE"))]

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
        if os.path.isdir(folder) and os.listdir(folder):  # interrupted: start over
            old = os.path.join(args.out, "_verworfen", f"{article}_unfertig")
            os.makedirs(old, exist_ok=True)
            for name_ in os.listdir(folder):
                os.replace(os.path.join(folder, name_), os.path.join(old, name_))
        rc = photo_cli.main(["--article", article, "--name", name, "--intro", "4",
                             "--wait-start", "--previous", previous, "--list-csv", args.csv,
                             "--out", args.out]
                            + ([] if args.raw else ["--isp"]))
        if rc != 0:
            print("stopped", flush=True)
            return rc
        previous = name or article


if __name__ == "__main__":
    sys.exit(main())
