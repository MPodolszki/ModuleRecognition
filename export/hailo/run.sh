#!/bin/sh
# Run compile_hef.py inside the Hailo AI SW Suite 2025-10 container
# (DFC 3.33.0 / HailoRT 4.23.0 -- the HailoRT the phyVIP BSP ships).
#
#   export/hailo/run.sh models/phyflex_n/phyflex_n.onnx models/phyflex_n/calib_640.npy [--profile int8 ...]
#
# Paths are relative to the project root, which is mounted at /work. Runs as
# the calling user so the HEF/HAR are not root-owned. A GPU is used when the
# NVIDIA container toolkit is installed (--gpus all); without it the DFC
# optimises on the CPU, which is fine for --profile int8 but slow for mz.
set -eu

IMAGE="${HAILO_IMAGE:-hailo8_ai_sw_suite_2025-10:1}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

GPU=""
if docker info 2>/dev/null | grep -q "nvidia"; then
    GPU="--gpus all"
fi

in_work() { echo "/work/$(realpath --relative-to="$ROOT" "$1")"; }
ONNX="$(in_work "$1")"
CALIB="$(in_work "$2")"
shift 2

# shellcheck disable=SC2086
exec docker run --rm $GPU \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$ROOT":/work -w /tmp \
    "$IMAGE" \
    /local/workspace/hailo_virtualenv/bin/python /work/export/hailo/compile_hef.py \
    "$ONNX" "$CALIB" "$@"
