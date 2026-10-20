#!/bin/sh
# Fetch phytec-photo / phytec-capture pictures from the board into the
# project: dataset/products/<Artikelnummer>/vm017/ (raw path) or
# .../vm017_isp/ (phytec-photo --isp, "path": "isp" in the .json), and
# _background/. Existing files are kept; the board's copies stay.
#
#   tools/fetch_captures.sh 172.25.32.142                 # all articles
#   tools/fetch_captures.sh 172.25.32.142 PCM-070-0F3243I.A3
set -eu
BOARD="$1"; shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC=/var/lib/phytec-detect/capture
SSH="ssh -o BatchMode=yes root@$BOARD"
ARTICLES="${*:-$($SSH "ls $SRC" | grep -v '^_raw_alt$\|^_verworfen$')}"
for art in $ARTICLES; do
    case "$art" in
        bg_*|_background) dest="$ROOT/dataset/products/_background/vm017" ;;
        _negatives) dest="$ROOT/dataset/products/_background/negatives_isp" ;;
        _scenes) dest="$ROOT/dataset/products/_scenes" ;;
        *) if $SSH "grep -qs '\"path\": \"isp\"' $SRC/$art/*.json"; then
               dest="$ROOT/dataset/products/$art/vm017_isp"
           else
               dest="$ROOT/dataset/products/$art/vm017"
           fi ;;
    esac
    mkdir -p "$dest"
    $SSH "cd $SRC/$art && tar -cf - ." | tar -C "$dest" --skip-old-files -xf -
    echo "$art: $(ls "$dest" | grep -c '\.jpg$') pictures -> ${dest#$ROOT/}"
done
