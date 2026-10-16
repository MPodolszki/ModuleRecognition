#!/bin/sh
# Fetch phytec-photo / phytec-capture pictures from the board into
# dataset/products/<Artikelnummer>/vm017/ (and _background/).
#
#   tools/fetch_captures.sh 172.25.32.145                 # all articles
#   tools/fetch_captures.sh 172.25.32.145 PCM-070-0F3243I.A3
set -eu
BOARD="$1"; shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC=/var/lib/phytec-detect/capture
ARTICLES="${*:-$(ssh -o BatchMode=yes root@"$BOARD" "ls $SRC")}"
for art in $ARTICLES; do
    case "$art" in
        bg_*|_background) dest="$ROOT/dataset/products/_background/vm017" ;;
        *) dest="$ROOT/dataset/products/$art/vm017" ;;
    esac
    mkdir -p "$dest"
    ssh -o BatchMode=yes root@"$BOARD" "cd $SRC/$art && tar -cf - ." | tar -C "$dest" -xf -
    echo "$art: $(ls "$dest" | grep -c '\.jpg$') pictures -> $dest"
done
