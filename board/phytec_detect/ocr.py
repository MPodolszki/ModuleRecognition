"""Read a business card: who is it? PaddleOCR (PP-OCRv5 mobile) without
PaddleOCR -- only onnxruntime or TFLite and OpenCV, which the board has.

    python3 -m phytec_detect.ocr card.png [--models /opt/phytec-detect/models/ocr] [--cpu]

Two networks, each either an ONNX model (CPU, onnxruntime) or a full-integer
TFLite model (export/ocr), the latter on the i.MX8MP NPU (VX delegate) or
the CPU. det runs on the NPU (~100 ms; the first run compiles the graph,
~15 s); rec does not -- the VX delegate fails on its SVTR part -- and INT8
misread names ("Khalld"), so it runs FP32 ONNX on the CPU (~0.5 s a line),
and the tallest lines, where the name is, first:

  det  finds the text lines on the straightened card (DB: a probability
       map, thresholded; every blob is a line, enlarged a little)
  rec  reads one line, 48 pixels high (CTC: best character per column,
       repeats and blanks dropped)

Fixed input sizes, as the NPU wants them: the card is scaled into
DET_H x DET_W, every line into REC_H x REC_W.

name_from() picks the person from the lines: two or three words that look
like a name, preferably the one the e-mail address on the card spells
(m.podolszki@ -> "Martin Podolszki"), never the company, street or title.
"""

import argparse
import os
import re
import sys
import time

import cv2
import numpy as np

MODEL_DIR = "/opt/phytec-detect/models/ocr"
DET_H, DET_W = 320, 512           # card, long side horizontal (85 x 55 mm ~ 1.55)
REC_H, REC_W = 48, 320            # one text line
DET_UP = 1.75                     # smallest det scale (small print)
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
VX_DELEGATE = "/usr/lib/libvx_delegate.so"


class _Net:
    """One network on onnxruntime (*.onnx) or TFLite (*_int8.tflite)."""

    def __init__(self, path, npu):
        self.path = path
        self.lock = None
        if path.endswith(".onnx"):
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2  # the demo needs the other cores
            self._sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
            self._name = self._sess.get_inputs()[0].name
            self.meta = self._sess.get_modelmeta().custom_metadata_map
            self.kind = "onnx"
            return
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        except ImportError:
            from ai_edge_litert.interpreter import Interpreter, load_delegate
        delegates = []
        if npu and os.path.exists(VX_DELEGATE):
            # no NBG cache file: with one, the delegate failed ("Create node
            # NBG fail") -- the graph is compiled at every start (~15 s, in
            # the background)
            delegates = [load_delegate(VX_DELEGATE)]
        self._interp = Interpreter(model_path=path, experimental_delegates=delegates,
                                   num_threads=2)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        self.meta = {}
        self.kind = "npu" if delegates else "tflite"

    def run(self, x):
        """x: float32 NCHW, normalised to -1..1 -> float32 output (as ONNX)."""
        if self.kind == "onnx":
            return self._sess.run(None, {self._name: x})[0]
        nhwc = x.transpose(0, 2, 3, 1)
        scale, zero = self._in["quantization"]
        if scale:
            nhwc = np.clip(np.round(nhwc / scale + zero),
                           np.iinfo(self._in["dtype"]).min, np.iinfo(self._in["dtype"]).max)
        outs = []
        for item in nhwc:  # batch 1 on the NPU
            self._interp.set_tensor(self._in["index"], item[None].astype(self._in["dtype"]))
            self._interp.invoke()
            y = self._interp.get_tensor(self._out["index"])
            s, z = self._out["quantization"]
            outs.append((y.astype(np.float32) - z) * s if s else y.astype(np.float32))
        y = np.concatenate(outs)
        return y.transpose(0, 3, 1, 2) if y.ndim == 4 else y  # det map back to NCHW


def _find(model_dir, stem, cpu):
    """<stem>_int8.tflite (NPU) unless cpu, else <stem>.onnx."""
    for name in ([] if cpu else [f"{stem}_int8.tflite"]) + [f"{stem}.onnx"]:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{model_dir}/{stem}.onnx")


def _starts(length, size, overlap=96):
    """Tile starts covering `length` with tiles of `size`, overlapping."""
    if length <= size:
        return [0]
    n = int(np.ceil((length - overlap) / (size - overlap)))
    return [round(i * (length - size) / (n - 1)) for i in range(n)]


def _same_line(a, b):
    """Two boxes of one line, seen by two tiles."""
    (ax, ay), (aw, ah), _ = a
    (bx, by), (bw, bh), _ = b
    return abs(ay - by) < min(min(aw, ah), min(bw, bh)) * 0.6 and \
        abs(ax - bx) < (max(aw, ah) + max(bw, bh)) / 4


class CardReader:
    def __init__(self, model_dir=MODEL_DIR, cpu=False, rec_int8=False):
        self.det = _Net(_find(model_dir, "det", cpu), npu=not cpu)
        self.rec = _Net(_find(model_dir, "rec", cpu or not rec_int8), npu=False)
        chars = self.rec.meta.get("character")
        if chars is None:
            with open(os.path.join(model_dir, "rec_chars.txt"), encoding="utf-8") as f:
                chars = f.read()
        # CTC classes: blank, the dictionary, space (PaddleOCR's order)
        self.chars = ["\0"] + chars.splitlines() + [" "]
        self.times = {}

    # -- det -------------------------------------------------------------------
    @staticmethod
    def _norm(bgr):
        return ((bgr.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)

    def lines(self, card):
        """Text line boxes on the card (BGR or grey): a list of rotated rects
        ((cx, cy), (w, h), angle) in card pixels. A picture larger than the
        det input is not scaled down (small print would get lost) but read
        in overlapping DET_H x DET_W tiles; smaller ones are scaled up."""
        # local contrast (CLAHE): thin, pale print (light grey on white) is
        # found far more often; the PHYTEC cards read the same either way
        gray = card if card.ndim == 2 else cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        bgr = cv2.cvtColor(_CLAHE.apply(gray), cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]
        s = min(DET_W / w, DET_H / h)
        if s >= DET_UP:
            return self._det_tile(bgr, 0, 0, s)
        # print ~10 px high is too small for det: scale up first, as
        # PaddleOCR does (shorter side >= 736), then tiles
        up = DET_UP
        bgr = cv2.resize(bgr, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        h, w = bgr.shape[:2]
        rects = []
        t = time.perf_counter()
        for y0 in _starts(h, DET_H):
            for x0 in _starts(w, DET_W):
                tile = bgr[y0:y0 + DET_H, x0:x0 + DET_W]
                for r in self._det_tile(tile, x0, y0, 1.0):
                    if not any(_same_line(r, q) for q in rects):
                        rects.append(r)
        self.times["det"] = time.perf_counter() - t
        rects = [((cx / up, cy / up), (rw / up, rh / up), a) for (cx, cy), (rw, rh), a in rects]
        rects.sort(key=lambda r: (round(r[0][1] / 8), r[0][0]))
        return rects

    def _det_tile(self, bgr, ox, oy, s):
        h, w = bgr.shape[:2]
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        canvas = np.full((DET_H, DET_W, 3), 255, np.uint8)  # white margin
        canvas[:nh, :nw] = bgr if s == 1 else cv2.resize(
            bgr, (nw, nh), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        t = time.perf_counter()
        prob = self.det.run(self._norm(canvas)[None])[0, 0]
        self.times["det"] = time.perf_counter() - t
        mask = cv2.dilate((prob > 0.3).astype(np.uint8), np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for c in contours:
            (cx, cy), (bw, bh), angle = cv2.minAreaRect(c)
            if min(bw, bh) < 3:
                continue
            m = np.zeros_like(mask)
            cv2.drawContours(m, [c], -1, 1, -1)
            if prob[m > 0].mean() < 0.5:  # box_thresh
                continue
            # a line cut by the tile's edge: the neighbour tile has it whole
            x, y, ww, hh = cv2.boundingRect(c)
            if s == 1 and ((x <= 1 and ox > 0) or (y <= 1 and oy > 0)
                           or (x + ww >= DET_W - 1 and w == DET_W)
                           or (y + hh >= DET_H - 1 and h == DET_H)):
                continue
            # unclip (PaddleOCR: distance = area * 1.6 / perimeter)
            d = bw * bh * 1.6 / (2 * (bw + bh))
            rects.append(((cx / s + ox, cy / s + oy), ((bw + 2 * d) / s, (bh + 2 * d) / s), angle))
        rects.sort(key=lambda r: (round(r[0][1] / 8), r[0][0]))  # top to bottom
        return rects

    # -- rec -------------------------------------------------------------------
    @staticmethod
    def _line_image(bgr, rect):
        """The line, straightened and read left to right: the long side's
        direction is taken pointing right (minAreaRect's angle alone left
        it upside down for slightly turned cards)."""
        (cx, cy), (w, h), angle = rect
        p = cv2.boxPoints(rect)
        e1, e2 = p[1] - p[0], p[2] - p[1]
        u = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
        u = u / max(np.hypot(*u), 1e-6)
        if u[0] < 0:
            u = -u
        v = np.array([-u[1], u[0]])  # down the line's height
        length, height = max(w, h), min(w, h)
        # INT8 det on the NPU drew some lines too short ("Martin Po"):
        # half a line height more on each side, blank space reads as nothing
        half_l, half_h = (length + height) / 2, height / 2
        c = np.array([cx, cy])
        src = np.float32([c - u * half_l - v * half_h, c + u * half_l - v * half_h,
                          c + u * half_l + v * half_h, c - u * half_l + v * half_h])
        ow, oh = max(1, int(round(2 * half_l))), max(1, int(round(height)))
        dst = np.float32([[0, 0], [ow, 0], [ow, oh], [0, oh]])
        return cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(src, dst), (ow, oh),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def read(self, card, rects):
        """Text of each line box: [(text, confidence)]."""
        gray = card if card.ndim == 2 else cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        bgr = cv2.cvtColor(_CLAHE.apply(gray), cv2.COLOR_GRAY2BGR)
        batch = np.zeros((len(rects), 3, REC_H, REC_W), np.float32)  # 0 = mid grey
        for i, rect in enumerate(rects):
            line = self._line_image(bgr, rect)
            lw = min(REC_W, max(1, int(np.ceil(REC_H * line.shape[1] / line.shape[0]))))
            batch[i, :, :, :lw] = self._norm(cv2.resize(line, (lw, REC_H)))
        if not len(rects):
            return []
        probs = self.rec.run(batch)
        out = []
        for p in probs:
            idx, conf = p.argmax(1), p.max(1)
            keep = (idx != 0) & np.r_[True, idx[1:] != idx[:-1]]
            text = "".join(self.chars[i] for i in idx[keep] if i < len(self.chars))
            out.append((text.strip(), float(conf[keep].mean()) if keep.any() else 0.0))
        return out

    def __call__(self, card, first=3):
        """Lines of the card: the `first` tallest lines (where a name is)
        are read first; the rest only if they hold no name."""
        rects = self.lines(card)
        order = sorted(range(len(rects)), key=lambda i: -min(rects[i][1]))
        texts = [("", 0.0)] * len(rects)
        t = time.perf_counter()
        for part in (order[:first], order[first:]):
            for i, line in zip(part, self.read(card, [rects[i] for i in part])):
                texts[i] = line
            if name_from(texts, rects):
                break
        self.times["rec"] = time.perf_counter() - t
        return texts, rects  # aligned: texts[i] is rects[i] ("" = not read)


# -- who is it? ----------------------------------------------------------------

NOT_A_NAME = re.compile(
    r"(gmbh|ag\b|kg\b|ltd|inc\b|corp|co\.|ug\b|e\.v|messtechnik|technik|systems|solutions|group|"
    r"str\.|straße|strasse|allee|weg\b|platz|ring\b|germany|deutschland|mainz|tel|fax|mobil|"
    r"phone|www|http|@|engineer|manager|leitung|leiter|head|director|sales|support|vertrieb|"
    r"geschäftsführ|ceo|cto|cfo|vp\b|president|consultant|berater|ingenieur|entwickl|"
    r"developer|assistant|team|department|abteilung|marketing|account|application)", re.I)
NAME_WORD = re.compile(r"^(?:[A-ZÄÖÜÀ-Þ][a-zäöüßà-ÿ'\-]+|[A-ZÄÖÜ]\.|von|van|de|der|den|zu|di|da|le)$")
TITLES = re.compile(r"^(dr\.?|prof\.?|dipl\.?-?\w*\.?|ing\.?|mba|msc|bsc|m\.sc\.?|b\.sc\.?)$", re.I)


def _simple(text):
    return (text.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
            .replace("ß", "ss"))


def _stacked(a, b):
    """Line b right below line a on the card (first name over surname):
    parallel, about one line apart, starting near the same place."""
    (ax, ay), (aw, ah), _ = a
    (bx, by), (bw, bh), _ = b
    pa, pb = cv2.boxPoints(a), cv2.boxPoints(b)
    ea, fa = pa[1] - pa[0], pa[2] - pa[1]
    u = ea if np.hypot(*ea) >= np.hypot(*fa) else fa
    u = u / max(np.hypot(*u), 1e-6)
    v = np.array([-u[1], u[0]])
    h = max(min(aw, ah), min(bw, bh))
    d = np.array([bx - ax, by - ay])
    along, across = abs(d @ u), abs(d @ v)
    starts = abs((pa @ u).min() - (pb @ u).min())
    return 0.5 * h < across < 2.6 * h and (along < max(aw, ah, bw, bh) or starts < 2 * h)


def _words(text):
    return [w for w in text.replace(",", " ").split() if not TITLES.match(w)]


def name_from(lines, rects=None):
    """(first name, surname) or None, from [(text, confidence)] of a card --
    a line of two to four name words, or (with the line boxes `rects`) a
    first name with the surname right below it in a line of its own."""
    emails = [m.group(1) for t, _ in lines
              for m in [re.search(r"([\w.\-]+)@[\w.\-]+\.\w+", t.replace(" ", ""))] if m]
    candidates = []  # (words, confidence)
    for text, conf in lines:
        words = _words(text)
        if 2 <= len(words) <= 4 and not NOT_A_NAME.search(text) and \
                not any(c.isdigit() for c in text) and all(NAME_WORD.match(w) for w in words):
            candidates.append((words, conf))
    if rects is not None and len(rects) == len(lines):
        single = [i for i, (t, c) in enumerate(lines)
                  if len(_words(t)) == 1 and NAME_WORD.match(_words(t)[0])
                  and len(_words(t)[0]) >= 3 and not NOT_A_NAME.search(t)]
        for i in single:
            for j in single:
                hi, hj = min(rects[i][1]), min(rects[j][1])
                if i < j and _stacked(rects[i], rects[j]) and 0.7 < hi / max(hj, 1e-6) < 1.4 \
                        and min(lines[i][1], lines[j][1]) >= 0.9:
                    # same font size, read with confidence: not a name over small print
                    # i above j: order along the card's height
                    a, b = (i, j) if _below(rects[i], rects[j]) else (j, i)
                    words = [_words(lines[a][0])[0], _words(lines[b][0])[0]]
                    candidates.append((words, (lines[a][1] + lines[b][1]) / 2 - 0.05))
    best, best_score = None, 0.0
    for words, conf in candidates:
        score = conf
        last = _simple(words[-1])
        if any(last in _simple(local) for local in emails):  # m.podolszki@
            score += 1.0
        elif min(len(w) for w in words) < 3:
            continue  # "Martin Po": a cut-off line, not a name
        if score > best_score:
            best, best_score = (" ".join(words[:-1]), words[-1]), score
    return best


def _below(a, b):
    """True when b lies below a across the lines (the card may be turned)."""
    p = cv2.boxPoints(a)
    e, f = p[1] - p[0], p[2] - p[1]
    u = e if np.hypot(*e) >= np.hypot(*f) else f
    u = u / max(np.hypot(*u), 1e-6)
    if u[0] < 0:
        u = -u
    v = np.array([-u[1], u[0]])  # down the text
    return float((np.array(b[0]) - np.array(a[0])) @ v) > 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="a straightened card (or any picture with text)")
    parser.add_argument("--models", default=MODEL_DIR)
    parser.add_argument("--cpu", action="store_true", help="ONNX on the CPU, not the NPU")
    args = parser.parse_args(argv)
    reader = CardReader(args.models, cpu=args.cpu)
    card = cv2.imread(args.image)
    reader(card)  # first run: NPU graph compile / cache
    t = time.perf_counter()
    lines, _ = reader(card)
    dt = time.perf_counter() - t
    for text, conf in lines:
        if text:
            print(f"  {conf:.2f}  {text}")
    print(f"name: {name_from(lines, _)}   ({reader.det.kind}/{reader.rec.kind}: "
          f"det {reader.times.get('det', 0) * 1000:.0f} ms, rec {reader.times.get('rec', 0) * 1000:.0f} ms, "
          f"total {dt * 1000:.0f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
