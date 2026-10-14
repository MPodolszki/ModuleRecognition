#!/bin/sh
# Copy phytec_detect and model directories to a board for testing.
#
#   tools/deploy.sh 172.25.32.143 models/phyflex_synth_n [models/yolo26n_coco ...]
#
# Installs to /opt/phytec-detect on the board:
#   /opt/phytec-detect/phytec_detect/      the Python package
#   /opt/phytec-detect/models/<name>/      labels.txt, model.json, *.hef, *_int8.tflite, <name>.onnx
#   /opt/phytec-detect/bin/phytec-{detect,benchmark,capture}
# The bin wrappers run as the weston user, on its Wayland display, like the
# kiosk demos. Large calibration/HAR/full-ONNX files are not copied.
set -eu

BOARD="$1"; shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST=/opt/phytec-detect
SSH="ssh -o BatchMode=yes root@$BOARD"

$SSH "mkdir -p $DEST/models $DEST/bin /var/cache/phytec-detect && chmod 777 /var/cache/phytec-detect"
tar -C "$ROOT/board" --exclude=__pycache__ -cf - phytec_detect | $SSH "tar -C $DEST -xf -"

for dir in "$@"; do
    name="$(basename "$dir")"
    echo "model $name"
    (cd "$dir" && tar --exclude='*.npy' --exclude='*.har' --exclude='*_full.onnx' \
        --exclude='*_fp32.tflite' -cf - .) | $SSH "mkdir -p $DEST/models/$name && tar -C $DEST/models/$name -xf -"
done

for tool in detect:live_cli benchmark:benchmark_cli capture:capture_cli; do
    cmd="${tool%%:*}"; module="${tool#*:}"
    $SSH "cat > $DEST/bin/phytec-$cmd" <<EOF
#!/bin/sh
# Run as weston on its display (camera, Hailo and the NPU are usable there too).
cd /tmp
if [ "\$(id -u)" = 0 ]; then
    exec su weston -s /bin/sh -c 'XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 PYTHONPATH=$DEST exec python3 -m phytec_detect.$module "\$@"' -- phytec-$cmd "\$@"
fi
XDG_RUNTIME_DIR=\${XDG_RUNTIME_DIR:-/run/user/1000} WAYLAND_DISPLAY=\${WAYLAND_DISPLAY:-wayland-0} \\
PYTHONPATH=$DEST exec python3 -m phytec_detect.$module "\$@"
EOF
done
$SSH "chmod +x $DEST/bin/*; ls -R $DEST | head -40"
