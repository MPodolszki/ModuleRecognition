#!/usr/bin/env python3
"""Every phytec-photo picture with the box it is trained with (green), in a
browser; a click takes a picture out of the training (red) or back in.

The choice is stored in the picture's .json as "skip_user" (true: out,
false: in even if tools/clean_photo_labels.py skipped it); build_products.py
reads it.  Pictures clean_photo_labels.py skipped start out red.

    uv run tools/label_review.py [--port 8765] [--products dataset/products]
"""

import argparse
import html
import json
import pathlib
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from synth_dataset import find_module, rect_box  # noqa: E402

THUMB = 220


def skipped(meta):
    """The same rule as build_products.py."""
    if meta.get("skip_user") is not None:
        return bool(meta["skip_user"])
    return bool(meta.get("skip"))


def photos(products):
    """{article: [json path, ...]} of all pictures with a .json, products.csv order."""
    order = [line.split(";")[0] for line in
             (products / "products.csv").read_text(encoding="utf-8").splitlines()[1:]]
    found = {}
    for meta in sorted(products.glob("*/*/*.json")):
        if meta.with_suffix(".jpg").exists() and not meta.parent.parent.name.startswith("_"):
            found.setdefault(meta.parent.parent.name, []).append(meta)
    return sorted(found.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 999)


def thumb(meta_path):
    meta = json.loads(meta_path.read_text())
    img = cv2.imread(str(meta_path.with_suffix(".jpg")))
    rect = meta.get("rect")
    if not rect:
        try:
            rect = find_module(img)
        except ValueError:
            rect = None
    if rect is None:
        small = cv2.resize(img, (THUMB, THUMB * img.shape[0] // img.shape[1]))
        return cv2.imencode(".jpg", small)[1].tobytes()
    x0, y0, x1, y1 = rect_box(rect, img.shape).astype(int)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 220, 0), 3)
    pad = int(0.6 * max(x1 - x0, y1 - y0))
    zx0, zy0 = max(x0 - pad, 0), max(y0 - pad, 0)
    zoom = img[zy0:y1 + pad, zx0:x1 + pad]
    return cv2.imencode(".jpg", cv2.resize(zoom, (THUMB, THUMB)), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


PAGE = """<!doctype html><html lang="de"><head><meta charset="utf-8">
<title>Trainingsbilder</title><style>
body{font-family:sans-serif;margin:16px;background:#f4f4f4;color:#222}
h2{margin:24px 0 6px;font-size:18px}h2 small{font-weight:normal;color:#666}
.grid{display:flex;flex-wrap:wrap;gap:6px}
.t{position:relative;cursor:pointer;border:4px solid #2a2;line-height:0}
.t img{width:%(t)dpx;height:%(t)dpx}
.t.off{border-color:#d22}.t.off img{opacity:.35}
.t span{position:absolute;left:0;top:0;background:rgba(0,0,0,.6);color:#fff;font-size:11px;line-height:14px;padding:1px 4px}
.t.off::after{content:"aus";position:absolute;right:4px;bottom:4px;background:#d22;color:#fff;font-size:12px;line-height:16px;padding:0 5px}
header{position:sticky;top:0;background:#f4f4f4;padding:6px 0;border-bottom:1px solid #ccc;z-index:1}
select{font-size:15px}
</style></head><body><header>Klick = Bild aus dem Training nehmen / wieder aufnehmen. Grün = wird trainiert, rot = aus.
&nbsp; Produkt: <select onchange="location.hash=this.value">%(options)s</select> &nbsp; <b id="n"></b></header>
%(body)s
<script>
function count(){document.getElementById('n').textContent=document.querySelectorAll('.t.off').length+' von '+document.querySelectorAll('.t').length+' aus';}
document.querySelectorAll('.t').forEach(t=>t.onclick=async()=>{
 const r=await fetch('/toggle?p='+encodeURIComponent(t.dataset.p),{method:'POST'});
 if(r.ok){t.classList.toggle('off',(await r.json()).off);count();}});
count();
</script></body></html>"""


def page(products):
    body, options = [], []
    for art, metas in photos(products):
        tiles = []
        for m in metas:
            meta = json.loads(m.read_text())
            rel = str(m.relative_to(products))
            label = f"{m.parent.name.replace('_isp', '')} {m.stem[-3:]}"
            if meta.get("kind") == "lifted":
                label += " Hand"
            tip = html.escape(meta.get("label_fix", ""))
            tiles.append(f'<div class="t{" off" if skipped(meta) else ""}" data-p="{html.escape(rel)}" '
                         f'title="{tip}"><img loading="lazy" src="/thumb?p={urllib.parse.quote(rel)}">'
                         f'<span>{html.escape(label)}</span></div>')
        options.append(f'<option value="{html.escape(art)}">{html.escape(art)}</option>')
        body.append(f'<h2 id="{html.escape(art)}">{html.escape(art)} <small>{len(metas)} Bilder</small></h2>'
                    f'<div class="grid">{"".join(tiles)}</div>')
    return PAGE % {"t": THUMB, "options": "".join(options), "body": "".join(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--products", default=str(ROOT / "dataset/products"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    products = pathlib.Path(args.products).resolve()

    def resolve(query):
        rel = urllib.parse.parse_qs(query).get("p", [""])[0]
        path = (products / rel).resolve()
        if products not in path.parents or path.suffix != ".json" or not path.exists():
            return None
        return path

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, data, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            if url.path == "/":
                self._send(200, page(products).encode(), "text/html; charset=utf-8")
            elif url.path == "/thumb" and (path := resolve(url.query)):
                self._send(200, thumb(path), "image/jpeg")
            else:
                self._send(404, b"", "text/plain")

        def do_POST(self):
            url = urllib.parse.urlparse(self.path)
            path = resolve(url.query) if url.path == "/toggle" else None
            if path is None:
                self._send(404, b"", "text/plain")
                return
            meta = json.loads(path.read_text())
            meta["skip_user"] = not skipped(meta)
            path.write_text(json.dumps(meta, indent=1))
            print(f"{'aus' if meta['skip_user'] else 'an '} {path.relative_to(products)}", flush=True)
            self._send(200, json.dumps({"off": meta["skip_user"]}).encode(), "application/json")

        def log_message(self, *_):
            pass

    print(f"http://localhost:{args.port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
