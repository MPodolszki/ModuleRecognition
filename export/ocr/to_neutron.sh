#!/bin/sh
# det_int8.tflite (export/ocr/to_tflite.py) -> det_neutron.tflite for the
# i.MX95 NPU (Neutron), which phytec_detect.ocr prefers there.
#
#   export/ocr/to_neutron.sh models/ocr/det_int8.tflite [models/ocr/det_neutron.tflite] [converter options]
#
# The Neutron delegate only runs operators that neutron-converter compiled
# offline into NeutronGraph nodes; everything else stays on the CPU. The
# converter has to match the board's Neutron runtime -- 3.1.1 in
# BSP-Yocto-NXP-i.MX95-PD26.1.0 (neutron_3.1.1.bb) -- so it comes from NXP's
# eIQ package index in exactly that version:
#
#   pip install --index-url https://eiq.nxp.com/repository eiq-neutron-sdk==3.1.1
#
# The two TRANSPOSE_CONVs of the DB head (80x128 -> 160x256 -> 320x512) stay
# on the CPU: scheduling them for the NPU ran the converter out of memory
# (killed at 48 GB, after 25 min); without them it takes 5 s and 100 MB and
# still maps 208 of 224 operators (NPU estimate ~5 ms at 1 GHz).
#
# NEUTRON_CONVERTER=/path/to/neutron-converter overrides the one on PATH.
# check_neutron.py compares the result with det_int8.tflite.
set -eu

IN="$1"
OUT="${2:-${IN%_int8.tflite}_neutron.tflite}"
shift $(( $# < 2 ? $# : 2 ))
CONVERTER="${NEUTRON_CONVERTER:-neutron-converter}"

"$CONVERTER" --input "$IN" --output "$OUT" --target imx95 \
    --exclude-operator-types TRANSPOSE_CONV "$@" > "$OUT.log" 2>&1 || {
    tail -20 "$OUT.log" >&2
    exit 1
}
# how much of the graph the NPU takes
grep -E "Operator conversion ratio|Number of Neutron graphs" "$OUT.log" | tail -2 || true
echo "wrote $OUT (log: $OUT.log)"
