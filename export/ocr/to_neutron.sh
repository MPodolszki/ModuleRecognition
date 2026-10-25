#!/bin/sh
# det_int8.tflite (export/ocr/to_tflite.py) -> det_neutron.tflite for the
# i.MX95 NPU (Neutron), which phytec_detect.ocr prefers there.
#
#   export/ocr/to_neutron.sh models/ocr/det_int8.tflite [models/ocr/det_neutron.tflite]
#
# The Neutron delegate only runs operators that neutron-converter compiled
# offline into NeutronGraph nodes; everything else stays on the CPU. The
# converter has to match the board's Neutron runtime -- 3.1.1 in
# BSP-Yocto-NXP-i.MX95-PD26.1.0 (neutron_3.1.1.bb) -- so it comes from NXP's
# eIQ package index in exactly that version:
#
#   pip install --index-url https://eiq.nxp.com/repository eiq-neutron-sdk==3.1.1
#
# NEUTRON_CONVERTER=/path/to/neutron-converter overrides the one on PATH.
set -eu

IN="$1"
OUT="${2:-${IN%_int8.tflite}_neutron.tflite}"
CONVERTER="${NEUTRON_CONVERTER:-neutron-converter}"

"$CONVERTER" --input "$IN" --output "$OUT" --target imx95 > "$OUT.log" 2>&1 || {
    tail -20 "$OUT.log" >&2
    exit 1
}
# how much of the graph the NPU takes
grep -E "Operator conversion ratio|Number of Neutron graphs" "$OUT.log" || true
echo "wrote $OUT (log: $OUT.log)"
