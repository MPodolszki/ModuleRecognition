#!/bin/sh
# Copy phytec_detect and model directories to a board for testing.
#
#   tools/deploy.sh 172.25.32.143 models/phyflex_synth_n [models/yolo26n_coco ...]
#
# Installs to /opt/phytec-detect on the board:
#   /opt/phytec-detect/phytec_detect/      the Python package
#   /opt/phytec-detect/models/<name>/      labels.txt, model.json, *.hef, *_int8.tflite, <name>.onnx
#   /opt/phytec-detect/bin/phytec-{detect,benchmark,capture,calibrate,focus,photo}
# The bin wrappers run as the weston user, on its Wayland display, like the
# kiosk demos. Large calibration/HAR/full-ONNX files are not copied.
set -eu

BOARD="$1"; shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST=/opt/phytec-detect
SSH="ssh -o BatchMode=yes root@$BOARD"

$SSH "mkdir -p $DEST/models $DEST/bin /var/cache/phytec-detect /etc/phytec-detect && chmod 777 /var/cache/phytec-detect && chown weston:weston /etc/phytec-detect"
tar -C "$ROOT/board" --exclude=__pycache__ -cf - phytec_detect | $SSH "tar -C $DEST -xf -"

for dir in "$@"; do
    name="$(basename "$dir")"
    echo "model $name"
    (cd "$dir" && tar --exclude='*.npy' --exclude='*.har' --exclude='*_full.onnx' \
        --exclude='*_fp32.tflite' -cf - .) | $SSH "mkdir -p $DEST/models/$name && tar -C $DEST/models/$name -xf -"
done

for tool in detect:live_cli benchmark:benchmark_cli capture:capture_cli calibrate:calibrate_cli focus:focus_cli photo:photo_cli photo-queue:photo_queue_cli; do
    cmd="${tool%%:*}"; module="${tool#*:}"
    $SSH "cat > $DEST/bin/phytec-$cmd" <<EOF
#!/bin/sh
# Run as weston on its display (camera, Hailo and the NPU are usable there too).
# As root, phytec_detect.asuser switches user without a login session, so the
# tool stays inside whatever systemd unit started it (with su it did not).
cd /tmp
if [ "\$(id -u)" = 0 ]; then
    PYTHONPATH=$DEST exec python3 -m phytec_detect.asuser weston phytec_detect.$module "\$@"
fi
XDG_RUNTIME_DIR=\${XDG_RUNTIME_DIR:-/run/user/1000} WAYLAND_DISPLAY=\${WAYLAND_DISPLAY:-wayland-0} \\
PYTHONPATH=$DEST exec python3 -m phytec_detect.$module "\$@"
EOF
done
$SSH "chmod +x $DEST/bin/*; ls -R $DEST | head -40"

# Boot helper: re-bind FPD-Link deserializers that did not lock at boot.
$SSH "cat > $DEST/bin/phytec-camera-retry && chmod +x $DEST/bin/phytec-camera-retry" \
    < "$ROOT/board/systemd/phytec-camera-retry.sh"
$SSH "cat > /etc/systemd/system/phytec-camera-retry.service && systemctl daemon-reload \
    && systemctl enable -q phytec-camera-retry" < "$ROOT/board/systemd/phytec-camera-retry.service"

# Product list for phytec-photo-queue (Artikelnummer;Bezeichnung).
"$ROOT/tools/sync_products.sh" "$BOARD" --once >/dev/null

# Demo, trade-fair training show and the buttons switching between them.
for unit in phytec-detect phytec-trial phytec-mode; do
    $SSH "cat > /etc/systemd/system/$unit.service" < "$ROOT/board/systemd/$unit.service"
done
$SSH "cat > $DEST/bin/phytec-mode && chmod +x $DEST/bin/phytec-mode" <<EOF2
#!/bin/sh
# Switch demo / training by hand (as root): phytec-mode demo|training
PYTHONPATH=$DEST exec python3 -m phytec_detect.mode_cli "\$@"
EOF2
$SSH "systemctl daemon-reload && systemctl enable -q phytec-mode && systemctl restart phytec-mode"
