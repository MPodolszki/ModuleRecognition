#!/bin/sh
# Keep the board's product list (phytec-photo-queue) in step with the
# project's dataset/products/products.csv: copies it whenever it changed.
#
#   tools/sync_products.sh 172.25.32.142            # every 15 s until stopped
#   tools/sync_products.sh 172.25.32.142 --once
set -u
BOARD="$1"; ONCE="${2:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CSV="$ROOT/dataset/products/products.csv"
last=""
while :; do
    sum="$(md5sum "$CSV" | cut -d' ' -f1)"
    if [ "$sum" != "$last" ]; then
        if ssh -o BatchMode=yes -o ConnectTimeout=5 root@"$BOARD" \
            "cat > /var/lib/phytec-detect/products.csv.new && chown weston:weston /var/lib/phytec-detect/products.csv.new && mv /var/lib/phytec-detect/products.csv.new /var/lib/phytec-detect/products.csv" < "$CSV"; then
            echo "$(date +%T) products.csv -> $BOARD ($(grep -c ';' "$CSV") rows incl. header)"
            last="$sum"
        fi
    fi
    [ "$ONCE" = --once ] && exit 0
    sleep 15
done
