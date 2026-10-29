#!/bin/sh
# Fetch phytec-photo / phytec-capture pictures from the board into the
# project: dataset/products/<Artikelnummer>/vm017/ (raw path) or
# .../vm017_isp/ (phytec-photo --isp, "path": "isp" in the .json), and
# _background/. A file already here is replaced only by a newer one from the
# board (a .json changed in phytec-review), a newer one here stays (changed
# in tools/label_review.py). Then the pictures that are here now are deleted
# on the board (.jpg/.json/.txt -- a .jpg only when its copy here has the
# same checksum); the markers DONE, DONE_R2, SKIP, R2_START stay, so the
# photo queue does not ask for the product again, and FETCHED (the last
# picture number taken away) makes phytec-photo number on after it. A product
# still being photographed (no DONE, or an R<n>_START without DONE_R<n>)
# keeps its pictures: the queue goes on from them. KEEP=1: nothing deleted.
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
# rsync --update: a file here that is newer than the board's stays
fetch() { src="$1"; dest="$2"; shift 2; rsync -a --update -e "ssh -o BatchMode=yes" "$@" "root@$BOARD:$src/" "$dest/"; }
# delete on the board what arrived here: same .jpg (md5), any .json/.txt
# (a differing one here is the newer: rsync --update kept it)
clean() { src="$1"; dest="$2"
    [ -n "${KEEP:-}" ] && return 0
    case "$src" in */_background|*/_negatives|*/_scenes) ;; *)  # products: only finished ones
        $SSH "cd $src && [ -e DONE ] && for s in R*_START; do [ -e \"\$s\" ] || continue; [ -e DONE_\${s%_START} ] || exit 1; done" \
            || { echo "  kept on the board: still being photographed"; return 0; } ;;
    esac
    gone="$($SSH "cd $src && md5sum *.jpg *.json *.txt 2>/dev/null" | while read -r sum name; do
        [ -f "$dest/$name" ] || continue
        case "$name" in
            *.jpg) [ "$(md5sum < "$dest/$name" | cut -d' ' -f1)" = "$sum" ] || continue ;;
        esac
        echo "$name"
    done)"
    [ -n "$gone" ] || return 0
    last="$(echo "$gone" | sed -n 's/.*_0*\([0-9][0-9]*\)\.jpg$/\1/p' | sort -n | tail -1)"
    [ -n "$last" ] && $SSH "cd $src && f=\$(cat FETCHED 2>/dev/null || echo 0); [ $last -gt \"\$f\" ] && echo $last > FETCHED; true"
    echo "$gone" | $SSH "cd $src && xargs rm -f"
    echo "  deleted on the board: $(echo "$gone" | grep -c '\.jpg$') pictures"
}
ARTICLES="${*:-$($SSH "ls $SRC" | grep -v '^_raw_alt$\|^_verworfen$\|^VORFUEHRUNG$')}"
for art in $ARTICLES; do
    case "$art" in
        bg_*|_background) dest="$ROOT/dataset/products/_background/$CAM" ;;
        _negatives) dest="$ROOT/dataset/products/_background/negatives_isp" ;;
        _scenes) dest="$ROOT/dataset/products/_scenes" ;;
        _isp960) dest="$ROOT/dataset/products/_isp960"   # phytec-improve, one folder per article
                 mkdir -p "$dest"
                 fetch "$SRC/_isp960" "$dest"
                 echo "_isp960: $(find "$dest" -name '*.jpg' ! -name '*_ref.jpg' | wc -l) pictures -> ${dest#$ROOT/}"
                 continue ;;
        *) if $SSH "grep -qs '\"path\": \"isp\"' $SRC/$art/*.json"; then
               dest="$ROOT/dataset/products/$art/${CAM}_isp"
           else
               dest="$ROOT/dataset/products/$art/$CAM"
           fi ;;
    esac
    $SSH "ls $SRC/$art/*.jpg" >/dev/null 2>&1 || continue  # only markers there: nothing to fetch
    if [ -n "$SINCE" ]; then  # only what was taken from SINCE on
        files="$($SSH "cd $SRC/$art && touch -d '$SINCE' /tmp/.fetch_since && find . -type f -newer /tmp/.fetch_since")"
        [ -n "$files" ] || continue
        mkdir -p "$dest"
        echo "$files" | fetch "$SRC/$art" "$dest" --files-from=-
    else
        mkdir -p "$dest"
        fetch "$SRC/$art" "$dest"
    fi
    echo "$art: $(ls "$dest" | grep -c '\.jpg$') pictures -> ${dest#$ROOT/}"
    clean "$SRC/$art" "$dest"
done
