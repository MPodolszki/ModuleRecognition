#!/bin/sh
# best.pt -> models/<name>/ with everything the board needs:
#   <name>.onnx (cut, FP32 reference), <name>_int8.tflite (NPU/CPU),
#   <name>_<profile>.hef (Hailo-8), labels.txt, model.json
#
#   tools/build_model.sh runs/detect/phyflex_synth_n/weights/best.pt <calib image dir> [hef profile] [extra compile_hef args]
#
# Both INT8 models are calibrated on the same pictures (calib_640.npy).
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS="$1"; CALIB_SRC="$2"; PROFILE="${3:-int8}"
shift 2; [ $# -gt 0 ] && shift
NAME="$(basename "$(dirname "$(dirname "$WEIGHTS")")")"
OUT="$ROOT/models/$NAME"
export YOLO_AUTOINSTALL=false
IMGSZ="${IMGSZ:-640}"   # network input (square), as trained: IMGSZ=960 tools/build_model.sh ...
CALIB="$OUT/calib_$IMGSZ.npy"

cd "$ROOT"
uv run export/cut_onnx.py "$WEIGHTS" --name "$NAME" --imgsz "$IMGSZ"
uv run tools/product_info.py "$OUT" >/dev/null   # products.json for the info panel
uv run export/make_calib.py "$CALIB_SRC" -o "$CALIB" -n 512 --imgsz "$IMGSZ"
if [ -z "${SKIP_NPU:-}" ]; then   # SKIP_NPU=1: Hailo only, ~10 min less
(cd export/npu && uv run to_tflite.py "$OUT/$NAME.onnx" "$CALIB" \
    --check-image "$(find -L "$CALIB_SRC" -name '*.jpg' | sort | head -1)") 2>&1 \
    | grep -vE '^[EWI]0000|^\s*$'
fi
export/hailo/run.sh "$OUT/$NAME.onnx" "$CALIB" --profile "$PROFILE" "$@" 2>&1 \
    | grep -E '^==|calib\[|Error|error|context_[0-9] utilization|Successful'
ls -la "$OUT"
