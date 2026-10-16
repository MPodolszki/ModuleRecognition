#!/bin/sh
# Copy a built model to the board and switch the demo to it -- rolls back to
# the previous model if the demo does not come up with it.
#
#   tools/autodeploy.sh 172.25.32.145 models/products_v3_n
set -u
BOARD="$1"; MODEL_DIR="$2"
NAME="$(basename "$MODEL_DIR")"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -f "$MODEL_DIR/${NAME}_int8.hef" ] || [ ! -f "$MODEL_DIR/${NAME}_int8.tflite" ]; then
    echo "AUTODEPLOY: $MODEL_DIR incomplete, demo left as it is"; ls "$MODEL_DIR"; exit 1
fi
tools/deploy.sh "$BOARD" "$MODEL_DIR" >/dev/null 2>&1 || { echo "AUTODEPLOY: deploy to $BOARD failed"; exit 1; }
ssh -o BatchMode=yes root@"$BOARD" "
cp /etc/default/phytec-detect /etc/default/phytec-detect.prev
sed -i 's|models/[a-z0-9_]*|models/$NAME|' /etc/default/phytec-detect
systemctl restart phytec-detect; sleep 40
if systemctl is-active -q phytec-detect && journalctl -u phytec-detect --since '-30s' -o cat | grep -q 'detect cam'; then
    echo 'AUTODEPLOY: OK, demo runs $NAME'
    journalctl -u phytec-detect --since '-20s' -o cat | grep 'detect cam' | tail -1
else
    echo 'AUTODEPLOY: $NAME did not start, rolled back'
    cp /etc/default/phytec-detect.prev /etc/default/phytec-detect; systemctl restart phytec-detect
fi"
