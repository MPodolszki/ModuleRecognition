#!/usr/bin/env python3
"""Fine-tune YOLO26 (COCO-pretrained) on the PHYTEC module dataset.

    uv run train/train.py --data dataset_synth/data.yaml --model yolo26n --name phyflex_synth_n

Runs end up in runs/detect/<name>/, weights/best.pt is what export/cut_onnx.py
takes next. imgsz stays 640: both the HEF and the TFLite are compiled for a
fixed 640x640 input.
"""

import argparse
import pathlib

from ultralytics import YOLO

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo26n", help="yolo26n / yolo26s / ...")
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", default=None, choices=("ram", "disk"),
                        help="keep the decoded pictures (faster epochs at 960 px)")
    args = parser.parse_args()

    # --model: a size (yolo26n) or a trained best.pt to continue from (fine-tune:
    # far fewer epochs; new classes get a fresh classification output)
    if pathlib.Path(args.model).suffix == ".pt" and pathlib.Path(args.model).exists():
        model = YOLO(args.model)
    else:
        weights = ROOT / "models" / f"{args.model}.pt"
        model = YOLO(str(weights) if weights.exists() else f"{args.model}.pt")
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device=0, project=str(ROOT / "runs" / "detect"),
        name=args.name, exist_ok=True, patience=25, plots=True,
        # Modules lie in any orientation on a desk: allow vertical flips and
        # rotation on top of the default augmentation.
        flipud=0.5, degrees=15.0, close_mosaic=10,
        **({"cache": args.cache} if args.cache else {}),
    )


if __name__ == "__main__":
    main()
