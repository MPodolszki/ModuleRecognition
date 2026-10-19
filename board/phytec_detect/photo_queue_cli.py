"""Photograph every product of products.csv that has no pictures yet.

    phytec-photo-queue                       # /var/lib/phytec-detect/products.csv
    phytec-photo-queue --csv other.csv

products.csv is the project's list (Artikelnummer;Bezeichnung, deployed
next to the captures). Each product without a finished phytec-photo run
(capture/<Artikelnummer>/DONE) is photographed through the ISP, one after
the other -- new rows in the list are all it takes. Between two products
the monitor shows which one is next and waits, as long as it takes, until
someone lays it in the middle: nobody has to sit in front of it. Stopping the queue stops the current product; run it again to go on
(that product starts over in its folder).
"""

import argparse
import os
import sys

from . import photo_cli

CAPTURE = "/var/lib/phytec-detect/capture"


def read_products(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(";")]
            if len(parts) < 1 or not parts[0] or parts[0].lower() == "artikelnummer":
                continue
            rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phytec-photo-queue", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="/var/lib/phytec-detect/products.csv")
    parser.add_argument("--out", default=CAPTURE)
    parser.add_argument("--raw", action="store_true", help="raw path instead of the ISP")
    args = parser.parse_args(argv)

    todo = [(a, n) for a, n in read_products(args.csv)
            if not os.path.exists(os.path.join(args.out, a, "DONE"))]
    if not todo:
        print("all products of the list are photographed", flush=True)
        return 0
    print("to photograph: " + ", ".join(f"{a} ({n})" for a, n in todo), flush=True)
    previous = ""
    for article, name in todo:
        print(f"next: {article} {name}", flush=True)
        folder = os.path.join(args.out, article)
        if os.path.isdir(folder):  # an interrupted run: start that product over
            old = os.path.join(args.out, "_verworfen", f"{article}_unfertig")
            os.makedirs(old, exist_ok=True)
            for name_ in os.listdir(folder):
                os.replace(os.path.join(folder, name_), os.path.join(old, name_))
        rc = photo_cli.main(["--article", article, "--name", name, "--intro", "4",
                             "--wait-start", "--previous", previous]
                            + ([] if args.raw else ["--isp"]))
        previous = name or article
        if rc != 0:
            print("stopped", flush=True)
            return rc
    print("all products of the list are photographed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
