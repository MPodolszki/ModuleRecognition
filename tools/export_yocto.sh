#!/bin/sh
# Copy the demo into the Yocto example recipe (meta-hailo-examples-phytec,
# recipes-examples/product-detection), the way tools/deploy.sh copies it to
# a board: the phytec_detect package, the systemd units, the product lists
# and one model directory.
#
#   tools/export_yocto.sh ~/share/augenblix2fae/Yocto/imx8mp/phyVIP1/sources/meta-hailo-examples-phytec [models/products_v8_n]
#
# Only the files the recipes install are copied: no HAR, calibration set,
# full ONNX or other model versions.
set -eu

LAYER="$1"
MODEL="${2:-models/products_v8_n}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R="$LAYER/recipes-examples/product-detection"
[ -d "$R" ] || { echo "no $R -- is $LAYER meta-hailo-examples-phytec?" >&2; exit 1; }
case "$MODEL" in /*) ;; *) MODEL="$ROOT/$MODEL" ;; esac
NAME="$(basename "$MODEL")"

# code
rm -rf "$R/files/phytec_detect"
mkdir -p "$R/files"
tar -C "$ROOT/board" --exclude=__pycache__ --exclude='*.pyc' -cf - phytec_detect | tar -C "$R/files" -xf -

# units and boot helper (phytec-improve: ISP-960 path, shelved -- not exported)
for f in phytec-detect.service phytec-trial.service phytec-mode.service \
         phytec-camera-retry.service phytec-camera-retry.sh; do
    cp "$ROOT/board/systemd/$f" "$R/files/$f"
done

# product list (phytec-photo-queue) and look-alike scenes (phytec-find)
cp "$ROOT/dataset/products/products.csv" "$R/files/products.csv"
cp "$ROOT/dataset/products/szenen.csv" "$R/files/szenen.csv"

# model
M="$R/demo-product-detection-model"
rm -rf "$M"
mkdir -p "$M/$NAME"
(cd "$MODEL" && tar --exclude='*.npy' --exclude='*.har' --exclude='*_full.onnx' \
    --exclude='*.onnx' --exclude='*_fp32.tflite' --exclude='*.dvm' -cf - .) | tar -C "$M/$NAME" -xf -

echo "exported $(git -C "$ROOT" describe --always --dirty) to $R:"
du -sh "$R/files/phytec_detect" "$M/$NAME"
