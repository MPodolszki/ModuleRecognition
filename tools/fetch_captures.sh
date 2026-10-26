#!/bin/sh
# Fetch phytec-photo / phytec-capture pictures from the board into the
# project: dataset/products/<Artikelnummer>/vm017/ (raw path) or
# .../vm017_isp/ (phytec-photo --isp, "path": "isp" in the .json), and
# _background/. Existing files are kept; the board's copies stay.
#
#   tools/fetch_captures.sh 172.25.32.142                 # all articles
#   tools/fetch_captures.sh 172.25.32.142 PCM-070-0F3243I.A3
#
# CAM / SINCE: the detection camera changed (VM-017 -> VM-117 on
# 2026-10-23 11:30). Pictures taken from SINCE on go to .../<CAM>/ resp.
# .../<CAM>_isp/ instead, so the two cameras' pictures never mix:
#   CAM=vm117 SINCE="2026-10-23 11:30" tools/fetch_captures.sh 172.25.32.142
set -eu
BOARD="$1"; shift
CAM="${CAM:-vm017}"
SINCE="${SINCE:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC=/var/lib/phytec-detect/capture
SSH="ssh -o BatchMode=yes root@$BOARD"
ARTICLES="${*:-$($SSH "ls $SRC" | grep -v '^_raw_alt$\|^_verworfen$')}"
for art in $ARTICLES; do
    case "$art" in
        bg_*|_background) dest="$ROOT/dataset/products/_background/$CAM" ;;
        _negatives) dest="$ROOT/dataset/products/_background/negatives_isp" ;;
        _scenes) dest="$ROOT/dataset/products/_scenes" ;;
        _isp960) dest="$ROOT/dataset/products/_isp960"   # phytec-improve, one folder per article
                 mkdir -p "$dest"
                 $SSH "cd $SRC/_isp960 && tar -cf - ." | tar -C "$dest" --skip-old-files -xf -
                 echo "_isp960: $(find "$dest" -name '*.jpg' ! -name '*_ref.jpg' | wc -l) pictures -> ${dest#$ROOT/}"
                 continue ;;
        *) if $SSH "grep -qs '\"path\": \"isp\"' $SRC/$art/*.json"; then
               dest="$ROOT/dataset/products/$art/${CAM}_isp"
           else
               dest="$ROOT/dataset/products/$art/$CAM"
           fi ;;
    esac
    if [ -n "$SINCE" ]; then  # only what was taken from SINCE on
        files="$($SSH "cd $SRC/$art && touch -d '$SINCE' /tmp/.fetch_since && find . -type f -newer /tmp/.fetch_since")"
        [ -n "$files" ] || continue
        mkdir -p "$dest"
        $SSH "cd $SRC/$art && tar -cf - $(echo $files)" | tar -C "$dest" --skip-old-files -xf -
    else
        mkdir -p "$dest"
        $SSH "cd $SRC/$art && tar -cf - ." | tar -C "$dest" --skip-old-files -xf -
    fi
    echo "$art: $(ls "$dest" | grep -c '\.jpg$') pictures -> ${dest#$ROOT/}"
done
