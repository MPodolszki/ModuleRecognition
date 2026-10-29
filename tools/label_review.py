#!/usr/bin/env python3
"""Every phytec-photo picture with the box it is trained with (green), in a
browser; a click takes a picture out of the training (red) or back in, the
pen opens the whole picture to draw the box by hand.

The choice is stored in the picture's .json as "skip_user" (true: out,
false: in even if tools/clean_photo_labels.py skipped it); build_products.py
reads it.  A drawn box replaces "rect" (the first one is kept as "rect_orig")
and is marked "box_user", which clean_photo_labels.py leaves alone.  Pictures clean_photo_labels.py skipped start out red.

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


def current_rect(meta, img):
    rect = meta.get("rect")
    if not rect:
        try:
            rect = find_module(img)
        except ValueError:
            rect = None
    return rect


def thumb(meta_path):
    meta = json.loads(meta_path.read_text())
    img = cv2.imread(str(meta_path.with_suffix(".jpg")))
    rect = current_rect(meta, img)
    if rect is None:
        small = cv2.resize(img, (THUMB, THUMB * img.shape[0] // img.shape[1]))
        return cv2.imencode(".jpg", small)[1].tobytes()
    x0, y0, x1, y1 = rect_box(rect, img.shape).astype(int)
    pts = cv2.boxPoints((tuple(rect[0]), tuple(rect[1]), rect[2])).astype(np.int32)
    cv2.rectangle(img, (x0, y0), (x1, y1), (170, 170, 170), 1)  # what YOLO is trained with
    cv2.polylines(img, [pts], True, (0, 220, 0), 3)
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
.t b{position:absolute;right:0;top:0;background:rgba(0,0,0,.6);color:#fff;font-size:15px;line-height:20px;padding:2px 7px;font-weight:normal}
.t b:hover{background:#06c}
#ed{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:9;align-items:center;justify-content:center;flex-direction:column;gap:10px}
#ed.on{display:flex}#ed .w{position:relative;overflow:hidden;width:min(94vw,84vh);height:min(94vw,84vh);background:#000}
#ed img{position:absolute;cursor:crosshair;user-select:none;max-width:none}
#ed svg{position:absolute;pointer-events:none}
#ed polygon{fill:none;stroke-width:2;vector-effect:non-scaling-stroke}#eo{stroke:#2c2}#en{stroke:#29f}
#ed .bar{color:#fff;font-size:15px}#ed button{font-size:16px;margin-left:8px;padding:4px 14px}
.t.off::after{content:"aus";position:absolute;right:4px;bottom:4px;background:#d22;color:#fff;font-size:12px;line-height:16px;padding:0 5px}
header{position:sticky;top:0;background:#f4f4f4;padding:6px 0;border-bottom:1px solid #ccc;z-index:1}
select{font-size:15px}
</style></head><body><header>Klick = Bild aus dem Training nehmen / wieder aufnehmen. Grün = wird trainiert, rot = aus.
&nbsp; Produkt: <select onchange="location.hash=this.value">%(options)s</select> &nbsp; <b id="n"></b></header>
%(body)s
<div id="ed"><div class="bar"><span id="et"></span> &nbsp; 1. Maus an einer Kante des Produkts entlang ziehen, 2. loslassen und zur Gegenkante fahren, 3. klicken.
<button id="es">Speichern</button><button id="ec">Abbrechen</button></div>
<div class="w"><img id="ei" draggable="false"><svg id="sv"><polygon id="eo"/><polygon id="en"/></svg></div></div>
<script>
function count(){document.getElementById('n').textContent=document.querySelectorAll('.t.off').length+' von '+document.querySelectorAll('.t').length+' aus';}
let cur=null,meta=null,phase=0,p0=null,p1=null,box=null;
const ed=document.getElementById('ed'),ei=document.getElementById('ei'),eo=document.getElementById('eo'),en=document.getElementById('en');
// zoom: the product (its box, 1.6x) fills the window; without a box the whole picture
function frame(){const W=ei.parentElement.clientWidth,sv=document.getElementById('sv');let x0=0,y0=0,x1=meta.w,y1=meta.h;
 if(meta.pts){const xs=meta.pts.map(q=>q[0]),ys=meta.pts.map(q=>q[1]),cx=(Math.min(...xs)+Math.max(...xs))/2,cy=(Math.min(...ys)+Math.max(...ys))/2,
  half=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys),120)*1.6;x0=cx-half;y0=cy-half;x1=cx+half;y1=cy+half;}
 const k=W/Math.max(x1-x0,y1-y0);
 for(const el of [ei,sv]){el.style.left=-x0*k+'px';el.style.top=-y0*k+'px';el.style.width=meta.w*k+'px';el.style.height=meta.h*k+'px';}
 sv.setAttribute('viewBox','0 0 '+meta.w+' '+meta.h);}
function place(el,pts){el.setAttribute('points',pts?pts.map(q=>q[0]+','+q[1]).join(' '):'');}
async function edit(t){cur=t;meta=await (await fetch('/meta?p='+encodeURIComponent(t.dataset.p))).json();box=null;phase=0;
 document.getElementById('et').textContent=t.querySelector('span').textContent;
 ei.onload=()=>{frame();place(eo,meta.pts);place(en,null);};ei.src='/full?p='+encodeURIComponent(t.dataset.p);ed.classList.add('on');}
function pt(e){const r=ei.getBoundingClientRect(),k=meta.w/r.width;
 return [Math.max(0,Math.min(meta.w,(e.clientX-r.left)*k)),Math.max(0,Math.min(meta.h,(e.clientY-r.top)*k))];}
// edge p0-p1, then the distance of the mouse from that edge is the other side
function rect(q){const dx=p1[0]-p0[0],dy=p1[1]-p0[1],L=Math.hypot(dx,dy)||1,nx=-dy/L,ny=dx/L;
 const h=(q[0]-p0[0])*nx+(q[1]-p0[1])*ny;
 return {pts:[p0,p1,[p1[0]+nx*h,p1[1]+ny*h],[p0[0]+nx*h,p0[1]+ny*h]],
  cx:(p0[0]+p1[0])/2+nx*h/2,cy:(p0[1]+p1[1])/2+ny*h/2,w:L,h:Math.abs(h),a:Math.atan2(dy,dx)*180/Math.PI};}
ei.onmousedown=e=>{e.preventDefault();if(phase===2){phase=3;return;}p0=pt(e);p1=p0;phase=1;box=null;};
window.onmousemove=e=>{if(phase===1){p1=pt(e);place(en,[p0,p1,p1,p0]);}
 else if(phase===2){box=rect(pt(e));place(en,box.pts);}};
window.onmouseup=e=>{if(phase===1){p1=pt(e);phase=Math.hypot(p1[0]-p0[0],p1[1]-p0[1])>5?2:0;}};
document.getElementById('ec').onclick=()=>ed.classList.remove('on');
document.addEventListener('keydown',e=>{if(e.key==='Escape')ed.classList.remove('on');});
document.getElementById('es').onclick=async()=>{
 if(phase!==3||!box||box.w<5||box.h<5){alert('Erst den Rahmen fertig zeichnen (Kante ziehen, Gegenkante klicken)');return;}
 const r=await fetch('/box?p='+encodeURIComponent(cur.dataset.p)+'&b='+[box.cx,box.cy,box.w,box.h,box.a].map(v=>v.toFixed(1)).join(','),{method:'POST'});
 if(r.ok){const img=cur.querySelector('img');img.src=img.src.split('&v=')[0]+'&v='+Date.now();cur.classList.remove('off');count();ed.classList.remove('on');}};
document.querySelectorAll('.t b').forEach(b=>b.onclick=e=>{e.stopPropagation();edit(b.parentElement);});
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
            if meta.get("box_user"):
                label += " \u270e"
            tip = html.escape(meta.get("label_fix", ""))
            tiles.append(f'<div class="t{" off" if skipped(meta) else ""}" data-p="{html.escape(rel)}" '
                         f'title="{tip}"><img loading="lazy" src="/thumb?p={urllib.parse.quote(rel)}">'
                         f'<span>{html.escape(label)}</span><b title="Rahmen zeichnen">&#9998;</b></div>')
        options.append(f'<option value="{html.escape(art)}">{html.escape(art)}</option>')
        body.append(f'<h2 id="{html.escape(art)}">{html.escape(art)} <small>{len(metas)} Bilder</small></h2>'
                    f'<div class="grid">{"".join(tiles)}</div>')
    return PAGE % {"t": THUMB, "options": "".join(options), "body": "".join(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--products", default=str(ROOT / "dataset/products"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0: reachable from other computers (no login!)")
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
            elif url.path == "/full" and (path := resolve(url.query)):
                self._send(200, path.with_suffix(".jpg").read_bytes(), "image/jpeg")
            elif url.path == "/meta" and (path := resolve(url.query)):
                img = cv2.imread(str(path.with_suffix(".jpg")))
                rect = current_rect(json.loads(path.read_text()), img)
                pts = None if rect is None else cv2.boxPoints(tuple(map(tuple, rect[:2])) + (rect[2],)).tolist()
                data = {"w": img.shape[1], "h": img.shape[0], "pts": pts}
                self._send(200, json.dumps(data).encode(), "application/json")
            else:
                self._send(404, b"", "text/plain")

        def do_POST(self):
            url = urllib.parse.urlparse(self.path)
            path = resolve(url.query) if url.path in ("/toggle", "/box") else None
            if path is None:
                self._send(404, b"", "text/plain")
                return
            meta = json.loads(path.read_text())
            if url.path == "/box":
                # centre, size along the drawn edge and across it, angle of the edge
                cx, cy, w, h, angle = map(float, urllib.parse.parse_qs(url.query)["b"][0].split(","))
                meta.setdefault("rect_orig", meta.get("rect"))
                meta["rect"] = [[cx, cy], [w, h], angle]
                meta["box_user"] = True
                meta["label_fix"] = "box drawn by hand"
                meta["skip_user"] = False
                path.write_text(json.dumps(meta, indent=1))
                print(f"box {path.relative_to(products)}: {cx:.0f},{cy:.0f} {w:.0f}x{h:.0f} {angle:.0f} deg", flush=True)
                self._send(200, b"{}", "application/json")
                return
            meta["skip_user"] = not skipped(meta)
            path.write_text(json.dumps(meta, indent=1))
            print(f"{'aus' if meta['skip_user'] else 'an '} {path.relative_to(products)}", flush=True)
            self._send(200, json.dumps({"off": meta["skip_user"]}).encode(), "application/json")

        def log_message(self, *_):
            pass

    print(f"http://{args.host}:{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
