"""The pictures phytec-photo took on this board, with the box each one is
trained with (green), as a web page for any browser in the network: a click
takes a picture out of the training (red) or back in, the pen opens it to
draw the box by hand (an edge, then the opposite side: a turned box).

    phytec-review                       # http://<board>:8765/
    phytec-review --port 8080 --capture /var/lib/phytec-detect/capture

Runs all the time (phytec-review.service). New pictures appear by themselves,
newest first, while phytec-photo / the photo queue takes them -- and those of
the trade-fair show (phytec-trial --show-pictures, capture/VORFUEHRUNG/, on
top; not training data, the next pass replaces them), and several people can work
at once: each page asks every 2 s for the changes since its last look (yellow
flash). Everything is stored in the picture's .json, which
tools/fetch_captures.sh brings to the project and build_products.py reads:
"skip_user" (true: out, false: in), a drawn box replaces "rect" (the first
one is kept as "rect_orig") and is marked "box_user".
"""

import argparse
import html
import json
import pathlib
import socket
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

CAPTURE = "/var/lib/phytec-detect/capture"
PRODUCTS = "/var/lib/phytec-detect/products.csv"
THUMB = 220
SCAN_S = 1.0
SHOW = "VORFUEHRUNG"   # phytec-trial's article: the show, not training data


def read_products(path):
    """products.csv (Artikelnummer;Bezeichnung) -> [(article, name)]."""
    rows = []
    try:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(";")]
                if parts[0] and parts[0].lower() != "artikelnummer":
                    rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
    except OSError:
        pass
    return rows


def skipped(meta):
    """The same rule as build_products.py."""
    if meta.get("skip_user") is not None:
        return bool(meta["skip_user"])
    return bool(meta.get("skip"))


def load(meta_path):
    """The .json, None while it is being written (or broken)."""
    try:
        return json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None


def pictures(capture):
    """{article: [json path, ...]}: every picture with a .json, one folder per article."""
    found = {}
    for meta in sorted(capture.glob("*/*.json")):
        art = meta.parent.name
        if art.startswith("_") or art.startswith("bg_") or not meta.with_suffix(".jpg").exists():
            continue
        found.setdefault(art, []).append(meta)
    return found


def taken(meta_path, meta):
    """When the picture was taken (sort key, newest first)."""
    return (meta or {}).get("time") or time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(meta_path.with_suffix(".jpg").stat().st_mtime))


def box_points(rect):
    return cv2.boxPoints((tuple(rect[0]), tuple(rect[1]), rect[2]))


def thumb(meta_path):
    meta = load(meta_path) or {}
    img = cv2.imread(str(meta_path.with_suffix(".jpg")))
    rect = meta.get("rect")
    if not rect:
        small = cv2.resize(img, (THUMB, THUMB * img.shape[0] // img.shape[1]))
        return cv2.imencode(".jpg", small)[1].tobytes()
    pts = box_points(rect)
    h, w = img.shape[:2]
    x0, y0 = np.clip(pts.min(0), 0, [w, h]).astype(int)
    x1, y1 = np.clip(pts.max(0), 0, [w, h]).astype(int)
    cv2.rectangle(img, (x0, y0), (x1, y1), (170, 170, 170), 1)  # what YOLO is trained with
    cv2.polylines(img, [pts.astype(np.int32)], True, (0, 220, 0), 3)
    pad = int(0.6 * max(x1 - x0, y1 - y0))
    zoom = img[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]
    return cv2.imencode(".jpg", cv2.resize(zoom, (THUMB, THUMB)), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


PAGE = """<!doctype html><html lang="de"><head><meta charset="utf-8">
<title>Trainingsbilder %(host)s</title><style>
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
.t.flash{outline:4px solid #fc0}
.t.off::after{content:"aus";position:absolute;right:4px;bottom:4px;background:#d22;color:#fff;font-size:12px;line-height:16px;padding:0 5px}
header{position:sticky;top:0;background:#f4f4f4;padding:6px 0;border-bottom:1px solid #ccc;z-index:1}
select{font-size:15px}
</style></head><body><header><b>%(host)s</b> &nbsp; Neue Fotos erscheinen von selbst. Klick = Bild aus dem Training nehmen / wieder aufnehmen. Grün = wird trainiert, rot = aus.
&nbsp; Produkt: <select onchange="location.hash=this.value">%(options)s</select> &nbsp; <b id="n"></b></header>
%(body)s
<div id="ed"><div class="bar"><span id="et"></span> <b id="ew" style="color:#fc0"></b> &nbsp; 1. Maus an einer Kante des Produkts entlang ziehen, 2. loslassen und zur Gegenkante fahren, 3. klicken.
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
async function edit(t){cur=t;document.getElementById('ew').textContent='';meta=await (await fetch('/meta?p='+encodeURIComponent(t.dataset.p))).json();box=null;phase=0;
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
 if(r.ok){mine.add((await r.json()).v);reload(cur);cur.classList.remove('off');count();ed.classList.remove('on');}};
function wire(t){t.querySelector('b').onclick=e=>{e.stopPropagation();edit(t);};
 t.onclick=async()=>{
 const off=!t.classList.contains('off');
 const r=await fetch('/toggle?p='+encodeURIComponent(t.dataset.p)+'&off='+(off?1:0),{method:'POST'});
 if(r.ok){const j=await r.json();mine.add(j.v);t.classList.toggle('off',j.off);count();}};}
document.querySelectorAll('.t').forEach(wire);
count();
// the others' changes: poll the server's change list
let since=%(v)d;const mine=new Set();
function reload(t){const img=t.querySelector('img');img.src=img.src.split('&v=')[0]+'&v='+Date.now();}
async function poll(){try{const j=await (await fetch('/changes?since='+since)).json();
 if(j.reload){location.reload();return;}
 for(const c of j.items){since=Math.max(since,c.v);if(mine.delete(c.v))continue;
  if(c.gone){const t=document.querySelector('.t[data-p="'+CSS.escape(c.p)+'"]');if(t)t.remove();continue;}
  if(c.tile){if(document.querySelector('.t[data-p="'+CSS.escape(c.p)+'"]'))continue;
   const g=document.getElementById('g-'+c.art);if(!g){location.reload();return;}
   g.insertAdjacentHTML('afterbegin',c.tile);const t=g.firstElementChild;wire(t);
   t.classList.add('flash');setTimeout(()=>t.classList.remove('flash'),5000);continue;}
  const t=document.querySelector('.t[data-p="'+CSS.escape(c.p)+'"]');if(!t)continue;
  t.classList.toggle('off',c.off);if(c.box)reload(t);
  t.classList.add('flash');setTimeout(()=>t.classList.remove('flash'),3000);
  if(cur===t&&ed.classList.contains('on'))document.getElementById('ew').textContent='Jemand anderes hat dieses Bild gerade geändert';}
 since=Math.max(since,j.v);count();}catch(e){}setTimeout(poll,2000);}
setTimeout(poll,2000);
</script></body></html>"""


def tile(capture, meta_path, meta):
    rel = str(meta_path.relative_to(capture))
    label = meta_path.stem.rsplit("_", 1)[-1]
    if meta.get("kind") == "lifted":
        label += " Hand"
    if meta.get("box_user"):
        label += " \u270e"
    return (f'<div class="t{" off" if skipped(meta) else ""}" data-p="{html.escape(rel)}" '
            f'title="{html.escape(meta.get("label_fix", ""))}"><img loading="lazy" '
            f'src="/thumb?p={urllib.parse.quote(rel)}"><span>{html.escape(label)}</span>'
            f'<b title="Rahmen zeichnen">&#9998;</b></div>')


def page(capture, products_csv, version):
    names = dict(read_products(products_csv))
    order = list(names)
    found = pictures(capture)
    body, options = [], []
    def rank(art):
        return -1 if art == SHOW else order.index(art) if art in order else len(order)
    for art in sorted(found, key=rank):
        metas = [(m, meta) for m in found[art] if (meta := load(m)) is not None]
        metas.sort(key=lambda mm: taken(*mm), reverse=True)
        tiles = [tile(capture, m, meta) for m, meta in metas]
        title = "Messe-Training (Vorf&uuml;hrung, wird nicht trainiert)" if art == SHOW else \
            html.escape(art) + (f" &middot; {html.escape(names[art])}" if names.get(art) else "")
        options.append(f'<option value="{html.escape(art)}">{html.escape(art)}</option>')
        body.append(f'<h2 id="{html.escape(art)}">{title} <small>{len(tiles)} Bilder</small></h2>'
                    f'<div class="grid" id="g-{html.escape(art)}">{"".join(tiles)}</div>')
    if not body:
        body.append('<p id="none">Noch keine Fotos.</p>')
    return PAGE % {"t": THUMB, "options": "".join(options), "body": "".join(body),
                   "v": version, "host": html.escape(socket.gethostname())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", default=CAPTURE)
    parser.add_argument("--csv", default=PRODUCTS, help="products.csv, for order and names")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0", help="127.0.0.1: this board only")
    args = parser.parse_args(argv)
    capture = pathlib.Path(args.capture).resolve()

    def resolve(query):
        rel = urllib.parse.parse_qs(query).get("p", [""])[0]
        path = (capture / rel).resolve()
        if capture not in path.parents or path.suffix != ".json" or not path.exists():
            return None
        return path

    lock = threading.Lock()   # one change at a time
    changes = []              # {"v", "p", "off", "box"} or a new picture {"v", "p", "art", "tile"}

    def changed(entry):
        entry["v"] = len(changes) + 1
        changes.append(entry)
        return entry["v"]

    def watch():
        """New and removed pictures (phytec-photo, the show) into the change list."""
        known = {m for metas in pictures(capture).values() for m in metas}
        while True:
            time.sleep(SCAN_S)
            now = {}
            for art, metas in pictures(capture).items():
                for m in metas:
                    now[m] = art
            for m in sorted(known - set(now)):
                known.discard(m)
                with lock:
                    changed({"p": str(m.relative_to(capture)), "gone": True})
            for m, art in sorted(now.items()):
                if m in known or (meta := load(m)) is None:
                    continue
                known.add(m)
                with lock:
                    changed({"p": str(m.relative_to(capture)), "art": art,
                             "tile": tile(capture, m, meta)})

    thumbs = {}  # (path, mtime of jpg and json) -> jpeg

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
                with lock:
                    version = len(changes)
                self._send(200, page(capture, args.csv, version).encode(), "text/html; charset=utf-8")
            elif url.path == "/changes":
                since = int(urllib.parse.parse_qs(url.query).get("since", ["0"])[0])
                with lock:
                    data = {"reload": True} if since > len(changes) else \
                        {"v": len(changes), "items": changes[since:]}
                self._send(200, json.dumps(data).encode(), "application/json")
            elif url.path == "/thumb" and (path := resolve(url.query)):
                key = (path, path.stat().st_mtime, path.with_suffix(".jpg").stat().st_mtime)
                if key not in thumbs:
                    thumbs[key] = thumb(path)
                self._send(200, thumbs[key], "image/jpeg")
            elif url.path == "/full" and (path := resolve(url.query)):
                self._send(200, path.with_suffix(".jpg").read_bytes(), "image/jpeg")
            elif url.path == "/meta" and (path := resolve(url.query)):
                img = cv2.imread(str(path.with_suffix(".jpg")))
                rect = (load(path) or {}).get("rect")
                data = {"w": img.shape[1], "h": img.shape[0],
                        "pts": box_points(rect).tolist() if rect else None}
                self._send(200, json.dumps(data).encode(), "application/json")
            else:
                self._send(404, b"", "text/plain")

        def do_POST(self):
            url = urllib.parse.urlparse(self.path)
            path = resolve(url.query) if url.path in ("/toggle", "/box") else None
            if path is None or load(path) is None:
                self._send(404, b"", "text/plain")
                return
            with lock:
                data = self._change(url, path)
            self._send(200, json.dumps(data).encode(), "application/json")

        def _change(self, url, path):
            meta = load(path)
            query = urllib.parse.parse_qs(url.query)
            rel = str(path.relative_to(capture))
            if url.path == "/box":
                # centre, size along the drawn edge and across it, angle of the edge
                cx, cy, w, h, angle = map(float, query["b"][0].split(","))
                meta.setdefault("rect_orig", meta.get("rect"))
                meta["rect"] = [[cx, cy], [w, h], angle]
                meta["box_user"] = True
                meta["label_fix"] = "box drawn by hand"
                meta["skip_user"] = False
                path.write_text(json.dumps(meta, indent=1))
                print(f"box {rel}: {cx:.0f},{cy:.0f} {w:.0f}x{h:.0f} {angle:.0f} deg", flush=True)
                return {"off": False, "v": changed({"p": rel, "off": False, "box": True})}
            # the state the clicker saw, flipped -- not a blind toggle, so two
            # people clicking the same picture do not undo each other
            meta["skip_user"] = query.get("off", ["1"])[0] == "1"
            path.write_text(json.dumps(meta, indent=1))
            print(f"{'aus' if meta['skip_user'] else 'an '} {rel}", flush=True)
            return {"off": meta["skip_user"],
                    "v": changed({"p": rel, "off": meta["skip_user"], "box": False})}

        def log_message(self, *_):
            pass

    threading.Thread(target=watch, daemon=True).start()
    print(f"http://{socket.gethostname()}:{args.port}/  ({capture})", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
