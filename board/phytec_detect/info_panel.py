"""Product info that slides in from the right when products are lifted.

  nothing lifted     no panel -- the table just shows boxes
  one lifted         the overlay: name, article number, short description,
                     key figures, highlights (the Merkmale NPU ... Flash as
                     "2,3 TOPS NPU", then the Alleinstellungsmerkmale), links
  (either)           slides out by itself after 30 s, until the lifted
                     products change
  two lifted         the comparison: a column per product, the same fixed
                     rows (Kerne, NPU ... Flash) for both, "–" where empty

Layout and fields: marketing's "Datensatz pro Produkt: Overlay und
Vergleich" (tools/product_info.py builds products.json from it). The overlay
covers `single` of the screen width (960 px of 1920), the comparison
`compare` (1240 px, the camera stays visible left); it eases in/out over
`slide_s`. Its picture is rendered only when its content changes; every
screen frame just blends the cached picture in at the current offset.
"""

import functools
import os
import re
import time

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, YELLOW, CYAN = (255, 255, 255, 255), (170, 170, 170, 255), (0, 230, 255, 255), \
    (255, 220, 120, 255)


FEATURES = ("NPU", "GPU", "ISP", "VPU", "CAN-FD", "RAM", "Flash")
TICKED = ("NPU", "GPU", "ISP", "VPU", "CAN-FD")  # a tick in the comparison
YES, NO = ("ja", "j", "yes", "x", "\u2713"), ("nein", "no", "n")


def feature_line(key, value):
    """A Merkmal as a highlight line of the overlay: "ja" -> "GPU", a
    number -> "2× CAN-FD", text -> "2,3 TOPS NPU"; None if empty or no."""
    value = (value or "").strip()
    if not value or value.lower() in NO:
        return None
    if value.lower() in YES:
        return key
    if value.isdigit():
        return f"{value}\u00d7 {key}"
    return f"{value} {key}"


def feature_cell(key, value):
    """A Merkmal in the comparison: (tick, text, muted) -- "ja" a tick
    alone, a number "2×", "nein" as it is, empty "–"."""
    value = (value or "").strip()
    if not value:
        return False, "\u2013", True
    if value.lower() in NO:
        return False, "nein", True
    if value.lower() in YES:
        return True, "", False
    return key in TICKED, f"{value}\u00d7" if value.isdigit() else value, False


def _option_lines(info):
    """(label, value, field) rows for one product: the Merkmale and the
    Alleinstellungsmerkmale if there are any (products.json from the
    data set), older files' highlights, else the option export."""
    if "features" in info or info.get("usp"):
        rows = [(line, "", key) for key, value in info.get("features", [])
                if (line := feature_line(key, value))]
        return rows + [(u, "", "Alleinstellungsmerkmal") for u in info.get("usp", [])]
    if info.get("highlights"):
        return [(h, "", "Highlight") for h in info["highlights"]]
    rows = []
    for o in info.get("options", []):
        name = " ".join(o.get("Option", "").split())
        desc = " ".join(o.get("Optionsbeschreibung", "").split())
        head = " ".join(o.get("Optionskopf Beschreibung", "").split())
        if not name or name == "-":
            continue
        rows.append((name, desc or head, "Option"))
    if not rows:
        rows = [(line, "", "Option") for line in info.get("summary", [])]
    return rows


def _draw_qr(img, link, size=260, margin=28):
    """QR code for `link` in the bottom right corner (white, with its quiet
    zone), 'Mehr Infos' above it; returns (top edge of the block, QR size)."""
    try:
        code = cv2.QRCodeEncoder.create().encode(link)
    except cv2.error:
        return img.shape[0], 0
    n = code.shape[0]
    cell = max(1, size // (n + 8))  # 4 modules quiet zone each side
    qr = cv2.resize(code, (n * cell, n * cell), interpolation=cv2.INTER_NEAREST)
    qr = cv2.copyMakeBorder(qr, 4 * cell, 4 * cell, 4 * cell, 4 * cell,
                            cv2.BORDER_CONSTANT, value=255)
    h, w = img.shape[:2]
    q = qr.shape[0]
    x0, y0 = w - margin - q, h - margin - q
    img[y0:y0 + q, x0:x0 + q, :3] = qr[..., None]
    img[y0:y0 + q, x0:x0 + q, 3] = 255
    (tw, _), _ = cv2.getTextSize("Mehr Infos", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    _put(img, "Mehr Infos", x0 + (q - tw) // 2, y0 - 14, 0.8, WHITE, 2)
    return y0 - 40, q


def _draw_chart(img, path, qr_size, height=300, margin=28, gap=24):
    """Product chart (e.g. the comparison spider) at the bottom, left of
    the QR code; returns its top edge (the panel's height if unreadable)."""
    chart = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    if chart is None:
        return h
    if chart.ndim == 2:
        chart = cv2.cvtColor(chart, cv2.COLOR_GRAY2BGRA)
    elif chart.shape[2] == 3:
        chart = cv2.cvtColor(chart, cv2.COLOR_BGR2BGRA)
    right = w - margin - (qr_size + gap if qr_size else 0)
    max_w = right - margin
    scale = min(height / chart.shape[0], max_w / chart.shape[1])
    chart = cv2.resize(chart, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    ch, cw = chart.shape[:2]
    x0, y0 = right - cw, h - margin - ch
    alpha = chart[..., 3:4].astype(np.float32) / 255
    region = img[y0:y0 + ch, x0:x0 + cw]
    region[..., :3] = (chart[..., :3] * alpha + region[..., :3] * (1 - alpha)).astype(np.uint8)
    region[..., 3] = 255
    return y0


def _wrap(text, width_px, scale, thickness=1):
    words, lines, line = text.split(), [], ""
    for w in words:
        test = f"{line} {w}".strip()
        if cv2.getTextSize(test, FONT, scale, thickness)[0][0] > width_px and line:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    return lines


def _put(img, text, x, y, scale, color, thickness=1):
    cv2.putText(img, text, (int(x), int(y)), FONT, scale, color, thickness, cv2.LINE_AA)


class InfoPanel:
    def __init__(self, products, labels, single=0.5, compare=1240 / 1920, slide_s=0.35,
                 show_s=30.0):
        self.products, self.labels = products, labels
        self.single, self.compare = single, compare
        self.slide_s, self.show_s = slide_s, show_s
        self._first, self._seen = {}, {}  # class -> first / last time seen lifted
        self._live = {}           # class -> BGR crop of the product in the drop zone
        self._live_dirty = False
        self._boxes = {}          # class -> picture box (x0, y0, x1, y1) in self._image
        self._key = ()            # classes shown (or being slid out)
        self._image = None        # rendered panel for _key
        self._pos = 0.0           # 0 = hidden, 1 = fully in
        self._version = 0         # bumped whenever the panel's picture changes
        self._restore = set()     # classes whose photo replaces a dropped camera picture
        self._copied = None       # width of the panel as last copied into the canvas
        self._target = 0.0
        self._last = time.monotonic()

    @property
    def fraction(self):
        """Share of the screen width: the comparison is wider."""
        return self.compare if len(self._key) > 1 else self.single

    def state(self):
        """What the panel shows right now (position, products, pictures) --
        equal states draw equal pixels."""
        return (round(self._pos, 4), self._key, self._version) if self._pos > 0 else None

    def moving(self):
        """Still sliding in or out (the screen must be redrawn)."""
        return self._pos != self._target

    def set_live(self, cls, image, tag="LIVE"):
        """Camera picture of class `cls` (BGR, None to drop it): shown
        instead of its product photo with `tag` in the corner -- LIVE from
        the drop zone, AUFGENOMMEN for the last still picture of a product
        picked up. Refreshed in place without a new render."""
        if image is None:
            if self._live.pop(cls, None) is not None:
                self._restore.add(cls)  # back to the photo, in place
                self._live_dirty = True
                self._version += 1
            return
        self._restore.discard(cls)
        self._live[cls] = (image, tag)
        self._live_dirty = True
        self._version += 1

    def shown(self):
        """Classes the panel shows now (also while sliding out)."""
        return self._key if self._pos > 0 else ()

    def picture_of(self, cls):
        entry = self._live.get(cls)
        return entry[0] if entry else None

    def update(self, lifted_classes, zone_classes=()):
        """Classes of the lifted products, the one lifted first first.
        Each product is shown for `show_s` seconds from when it was lifted
        (a product taken away for good would keep the panel open), counted
        per product: the list flickering -- two look-alikes swapping class,
        a second product lifted -- does not start the time again. A product
        missing from the list for 2 s and lifted again counts as new."""
        now = time.monotonic()
        lifted = list(dict.fromkeys(lifted_classes))
        for c in lifted:
            self._first.setdefault(c, now)
            self._seen[c] = now
        for c in [c for c, t in self._seen.items() if now - t > 2.0]:
            del self._first[c], self._seen[c]
        key = tuple(c for c in lifted if now - self._first[c] < self.show_s)
        # a product in the drop zone: shown at once and as long as it lies there
        key = tuple(dict.fromkeys(list(zone_classes) + list(key)))[:2]
        if key:
            self._target = 1.0
            if key != self._key:
                self._key, self._image = key, None
        else:
            self._target = 0.0    # keep the old content while sliding out

    def static_from(self, width):
        """x from where the canvas holds the fully shown, unchanged panel --
        the caller then clears and draws only left of it and passes
        intact=True; None while it slides, changes or is hidden."""
        pw = int(width * self.fraction)
        if self._pos >= 1 and self._target >= 1 and self._key and self._image is not None \
                and self._copied == pw and self._image.shape[1] == pw:
            return width - pw
        return None

    def compose(self, canvas, intact=False):
        """Blend the panel into `canvas` (BGRx) at its current slide position.
        intact: the canvas still holds the panel as last composed (only the
        pictures that changed are copied then)."""
        now = time.monotonic()
        step = (now - self._last) / self.slide_s
        self._last = now
        self._pos = min(self._target, self._pos + step) if self._target > self._pos \
            else max(self._target, self._pos - step)
        if self._pos <= 0 or not self._key:
            return canvas
        h, w = canvas.shape[:2]
        pw = int(w * self.fraction)
        if self._image is None or self._image.shape[:2] != (h, pw):
            self._image = self._render(h, pw)
            self._live_dirty = True
            self._version += 1
            intact = False
        patched = []
        if self._live_dirty:
            self._live_dirty = False
            for cls, box in self._boxes.items():
                if cls in self._live:
                    picture, tag = self._live[cls]
                    _fit_picture(self._image, picture, box, tag=tag)
                    patched.append(box)
                elif cls in self._restore:
                    _fit_picture(self._image, _load_picture(self._info(cls)[1].get("image")), box)
                    patched.append(box)
            self._restore.clear()
        eased = 1 - (1 - self._pos) ** 3
        shown = int(pw * eased)
        if shown <= 0:
            self._copied = None
            return canvas
        x = w - shown
        if intact and shown == pw and self._copied == pw:
            for box in patched:  # only the pictures that changed
                x0, y0, x1, y1 = (int(v) for v in box)
                canvas[y0:y1, x + x0:x + x1] = self._image[y0:y1, x0:x1]
        else:  # opaque like a web page (the rendered panel's alpha is 255)
            canvas[:, x:] = self._image[:, :shown]
        self._copied = shown
        cv2.line(canvas, (x, 0), (x, h), (115, 110, 0, 255), 4)  # petrol
        return canvas

    # -- rendering -------------------------------------------------------

    def _info(self, cls):
        art = self.labels[cls] if cls < len(self.labels) else str(cls)
        return art, self.products.get(art, {})

    def _render(self, h, w):
        page = Page(w, h)
        if len(self._key) == 1:
            _render_single(page, *self._info(self._key[0]))
        else:
            _render_compare(page, self._info(self._key[0]), self._info(self._key[1]))
        image = page.image()
        # picture boxes in key order (single: one, compare: two)
        self._boxes = dict(zip(self._key, page.picture_boxes))
        return image


# -- PHYTEC web look: white page, Roboto, petrol accents ---------------------

FONTS = os.path.join(os.path.dirname(__file__), "fonts")
PIC_W, PIC_H = 300, 170      # product picture, top right of the single panel
THUMB_W, THUMB_H = 150, 104  # ... and per product in the comparison
PETROL = (0, 110, 115)          # RGB, phytec.de breadcrumb / buttons
PETROL_TINT = (229, 241, 241)
INK = (20, 20, 20)
MUTED = (112, 112, 112)
RULE = (221, 226, 226)
BAND = (243, 246, 246)
ORANGE = (255, 151, 8)          # phytec.de "NEW" tag
ORANGE_TINT = (255, 241, 222)


class Page:
    """A panel drawn like a phytec.de product page (PIL, Roboto)."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self._fonts = {}
        self.pastes = []  # (numpy BGR(A) image, x, y): QR code, chart
        self.marks = []   # (field, box): where each field landed, for tools/render_panel.py
        self.overflow = []  # (field, text) that did not fit
        self.picture_boxes = []  # product picture boxes, in drawing order
        self.pictures = []       # (path, box) drawn into the image at the end
        try:
            from PIL import Image, ImageDraw
        except ImportError:  # no Pillow on the board: OpenCV's own font
            self.img, self.draw = None, _CvDraw(np.full((h, w, 3), 255, np.uint8))
            return
        self.img = Image.new("RGB", (w, h), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.img)

    def font(self, size, weight="Regular"):
        if self.img is None:
            return (size, weight)
        from PIL import ImageFont
        key = (size, weight)
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(os.path.join(FONTS, f"Roboto-{weight}.ttf"), size)
        return self._fonts[key]

    def text(self, x, y, text, size, weight="Regular", fill=INK):
        """Draw with the top of the text at y; returns the line height."""
        self.draw.text((x, y), text, font=self.font(size, weight), fill=fill)
        return int(size * 1.25)

    def width(self, text, size, weight="Regular"):
        return self.draw.textlength(text, font=self.font(size, weight))

    def wrap(self, text, width, size, weight="Regular"):
        words, lines, line = text.split(), [], ""
        for word in words:
            test = f"{line} {word}".strip()
            if self.width(test, size, weight) > width and line:
                lines.append(line)
                line = word
            else:
                line = test
        return lines + ([line] if line else [])

    def mark(self, field, box):
        self.marks.append((field, tuple(int(v) for v in box)))

    def rect(self, box, fill=None, outline=None, width=1, radius=0):
        self.draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    def check(self, x, y, s=18, color=PETROL):
        """phytec.de list bullet: petrol box with a white tick."""
        self.rect((x, y, x + s, y + s), fill=color, radius=3)
        self.draw.line([(x + 4, y + s * 0.52), (x + s * 0.42, y + s - 5), (x + s - 4, y + 5)],
                       fill=(255, 255, 255), width=3)

    def image(self):
        rgb = np.asarray(self.img) if self.img is not None else self.draw.rgb
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
        for path, box in self.pictures:
            _fit_picture(out, _load_picture(path), box)
        for picture, x, y in self.pastes:
            ph, pw = picture.shape[:2]
            if picture.ndim == 2:
                picture = cv2.cvtColor(picture, cv2.COLOR_GRAY2BGRA)
            elif picture.shape[2] == 3:
                picture = cv2.cvtColor(picture, cv2.COLOR_BGR2BGRA)
            a = picture[..., 3:4].astype(np.float32) / 255
            region = out[y:y + ph, x:x + pw]
            region[..., :3] = (picture[..., :3] * a + region[..., :3] * (1 - a)).astype(np.uint8)
        return out


class _CvDraw:
    """The few ImageDraw calls Page uses, with OpenCV (Hershey font)."""

    def __init__(self, rgb):
        self.rgb = rgb

    def _scale(self, font):
        size, weight = font
        return size / 30, 3 if weight == "Bold" else 2 if weight == "Medium" else 1

    def text(self, xy, text, font, fill):
        scale, thick = self._scale(font)
        (_, th), _ = cv2.getTextSize("Hg", cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(self.rgb, text, (int(xy[0]), int(xy[1] + th)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, fill, thick, cv2.LINE_AA)

    def textlength(self, text, font):
        scale, thick = self._scale(font)
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]

    def rounded_rectangle(self, box, radius=0, fill=None, outline=None, width=1):
        p0, p1 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        if fill is not None:
            cv2.rectangle(self.rgb, p0, p1, fill, -1)
        if outline is not None:
            cv2.rectangle(self.rgb, p0, p1, outline, width)

    def line(self, points, fill, width=1):
        pts = np.array([[int(x), int(y)] for x, y in points], np.int32)
        cv2.polylines(self.rgb, [pts], False, fill, width, cv2.LINE_AA)


def _header(page, crumb):
    """Petrol bar with a breadcrumb, as on top of every phytec.de page."""
    page.rect((0, 0, page.w, 58), fill=PETROL)
    x = 32
    for i, part in enumerate(crumb):
        if i:
            page.text(x, 18, ">", 18, fill=(160, 205, 207))
            x += 22
        page.text(x, 18, part, 18, "Medium" if i == len(crumb) - 1 else "Regular",
                  fill=(255, 255, 255))
        x += page.width(part, 18, "Medium" if i == len(crumb) - 1 else "Regular") + 14


def _family(name):
    n = (name or "").lower()
    return "System on Modules" if n.startswith(("phycore", "phyflex")) else "Produkte"


def _qr(link, size):
    try:
        code = cv2.QRCodeEncoder.create().encode(link)
    except cv2.error:
        return None
    n = code.shape[0]
    cell = max(1, size // (n + 8))
    qr = cv2.resize(code, (n * cell, n * cell), interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(qr, 4 * cell, 4 * cell, 4 * cell, 4 * cell,
                              cv2.BORDER_CONSTANT, value=255)


def _bottom_band(page, info, margin=32):
    """Grey band at the bottom: chart, then one QR code per link (product
    page, development kit), each under a petrol label like phytec.de's
    buttons. Returns the band's top edge."""
    links = info.get("links") or ([["Mehr Infos", info["link"]]] if info.get("link") else [])
    chart_path = info.get("chart")
    if not links and not chart_path:
        return page.h
    size = 200  # same band height with one or two codes
    codes = [(label, _qr(url, size)) for label, url in links]
    codes = [(label, q) for label, q in codes if q is not None]
    block = max([q.shape[0] for _, q in codes], default=236)
    codes = [(label, cv2.copyMakeBorder(q, (block - q.shape[0]) // 2, (block - q.shape[0] + 1) // 2,
                                        (block - q.shape[0]) // 2, (block - q.shape[0] + 1) // 2,
                                        cv2.BORDER_CONSTANT, value=255))
             for label, q in codes]  # same white square for each, longer links = finer code
    top = page.h - margin - block - 70  # room for the labels above
    page.rect((0, top - 24, page.w, page.h), fill=BAND)
    page.draw.line([(0, top - 24), (page.w, top - 24)], fill=RULE, width=1)
    y_img = page.h - margin - block
    x = page.w - margin
    for label, q in reversed(codes):  # right to left
        x -= q.shape[0]
        page.pastes.append((q, x, page.h - margin - q.shape[0]))
        bw, bh = q.shape[0], 46
        page.mark("Link Produktseite" if label == "Zum Produkt" else "Link Kit",
                  (x, y_img - bh - 16, x + bw, page.h - margin))
        page.rect((x, y_img - bh - 16, x + bw, y_img - 16), fill=PETROL, radius=4)
        fs = 19
        while page.width(label + "  \u203a", fs, "Medium") > bw - 16 and fs > 12:
            fs -= 1
        text = label + "  \u203a"
        page.text(x + (bw - page.width(text, fs, "Medium")) / 2, y_img - bh - 16 + (bh - fs) / 2 - 2,
                  text, fs, "Medium", fill=(255, 255, 255))
        x -= 20
    if chart_path:
        chart = cv2.imread(chart_path, cv2.IMREAD_UNCHANGED)
        if chart is not None:
            right = x + 20 - 24 if codes else page.w - margin
            scale = min(block / chart.shape[0], (right - margin) / chart.shape[1])
            chart = cv2.resize(chart, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            ch, cw = chart.shape[:2]
            page.pastes.append((chart, right - cw, page.h - margin - ch))
            page.text(right - cw, top - 6, "Im Vergleich", 20, "Medium")
            page.mark("Diagramm-Datei", (right - cw, top - 6, right, page.h - margin))
    return top - 24


@functools.lru_cache(maxsize=64)
def _load_picture(path):
    return cv2.imread(path) if path and os.path.exists(path) else None


def _fit_picture(canvas, picture, box, tag=None):
    """`picture` (BGR) fitted into `box` of a BGR(A) canvas, centred on the
    band colour; `tag` (LIVE, AUFGENOMMEN) in the corner for a camera picture."""
    x0, y0, x1, y1 = (int(v) for v in box)
    bw, bh = x1 - x0, y1 - y0
    region = canvas[y0:y1, x0:x1]
    region[..., :3] = BAND[::-1]
    if picture is None:
        return
    if picture.ndim == 2:
        picture = cv2.cvtColor(picture, cv2.COLOR_GRAY2BGR)
    scale = min(bw / picture.shape[1], bh / picture.shape[0])
    pw, ph = max(1, int(picture.shape[1] * scale)), max(1, int(picture.shape[0] * scale))
    small = cv2.resize(picture[..., :3], (pw, ph),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    ox, oy = (bw - pw) // 2, (bh - ph) // 2
    region[oy:oy + ph, ox:ox + pw, :3] = small
    if tag:
        from . import style
        patch = style.text_patch(tag, 16, "Bold", bg=style.ORANGE, pad=(8, 3))
        style.blit(region, patch, 8, 8)


def _picture(page, info, box):
    """Product picture box (filled in Page.image(), live pictures by
    InfoPanel.compose)."""
    page.rect(box, fill=BAND)
    page.picture_boxes.append(box)
    page.pictures.append((info.get("image"), box))
    page.mark("Produktbild", box)


def _render_single(page, art, info):
    name = info.get("name") or art
    category = info.get("category", "")
    section = category + "s" if category.endswith(("Module", "Kit")) else category or \
        (_family(name) if _family(name) != "Produkte" else "")
    _header(page, [part for part in ("Produkte", section, name) if part])
    x, w = 32, page.w
    bottom = _bottom_band(page, info) - 16
    pic = (w - 32 - PIC_W, 84, w - 32, 84 + PIC_H)
    _picture(page, info, pic)
    tw = pic[0] - 24 - x  # text width left of the picture
    y = 92
    kicker = "PHYTEC " + (info.get("category") or
                          ("System on Module" if _family(name) != "Produkte" else "Produkt"))
    page.text(x, y, kicker, 18, fill=MUTED)
    page.mark("Kategorie", (x, y - 2, x + page.width(kicker, 18), y + 24))
    if info.get("new"):  # orange "NEU" tag as on phytec.de
        tx = x + page.width(kicker, 18) + 14
        page.rect((tx, y - 2, tx + 52, y + 24), outline=ORANGE, width=2, radius=3)
        page.text(tx + 9, y, "NEU", 17, "Medium", fill=ORANGE)
        page.mark("Neu", (tx, y - 2, tx + 52, y + 24))
    y += 30
    y_title = y
    size = next((s for s in (46, 40, 34) if page.width(name, s, "Bold") <= tw), 34)
    title = page.wrap(name, tw, size, "Bold")
    for line in title:
        page.text(x, y, line, size, "Bold")
        y += int(size * 1.26)
    page.mark("Bezeichnung", (x, y_title, x + max(page.width(l, size, "Bold") for l in title),
                              y - 4))
    if re.match(r"^[A-Z]{2,5}-[A-Z0-9]", art):  # not a placeholder folder name
        page.text(x, y + 2, f"Artikelnummer {art}", 20, fill=MUTED)
        page.mark("Artikelnummer", (x, y, x + page.width(f"Artikelnummer {art}", 20), y + 28))
    y += 40
    y_desc = y
    # next to the picture narrow, below it the full width
    words, line, lines = info.get("description", "").split(), "", []
    for word in words:
        width = tw if y_desc + 26 * len(lines) < pic[3] else w - 2 * x
        if line and page.width(f"{line} {word}", 19) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines += [line] if line else []
    if len(lines) > 3:
        page.overflow.append(("Kurzbeschreibung", f"{len(lines)} Zeilen, nur 3 passen"))
    for line in lines[:3]:
        page.text(x, y, line, 19, fill=INK)
        y += 26
    if y > y_desc:
        page.mark("Kurzbeschreibung", (x, y_desc - 2, w - x, y))
    y = max(y + 6, pic[3] + 12)
    facts = info.get("facts") or []
    if facts:  # key figures row, as under the title on phytec.de
        cw = (w - 2 * x) / len(facts)
        page.draw.line([(x, y), (w - x, y)], fill=RULE, width=1)
        for i, (label, value) in enumerate(facts):
            fx = x + i * cw
            if i:
                page.draw.line([(fx - 10, y + 12), (fx - 10, y + 70)], fill=RULE, width=1)
            if page.width(value, 22, "Medium") > cw - 20:
                page.overflow.append((label, f"zu breit: {value}"))
            page.text(fx, y + 14, label, 16, fill=MUTED)
            page.text(fx, y + 38, value, 22, "Medium", fill=PETROL)
            page.mark(label, (fx, y + 10, fx + cw - 20, y + 70))
        y += 84
        page.draw.line([(x, y), (w - x, y)], fill=RULE, width=1)
        y += 22
    else:
        y += 8
    page.text(x, y, "Highlights", 28, "Bold")
    y += 48
    rows = _option_lines(info)
    if not rows:
        page.text(x, y, "Produktdaten folgen.", 20, fill=MUTED)
    # Merkmale + Alleinstellungsmerkmale can be 11 lines: closer together
    # (down to 30 px) before one is left out
    pitch = 36 if any(v for _, v, _ in rows) else max(30, min(36, (bottom - y) // max(1, len(rows))))
    size = 21 if pitch > 32 else 20
    for k, (label, value, field) in enumerate(rows):
        lines = page.wrap(value, w - 2 * x - 34, 17) if value else []
        need = pitch - 8 + 24 * min(len(lines), 1)
        if y + need > bottom:
            page.overflow.append(("Highlights", f"nur {k} von {len(rows)} passen"))
            break
        if field != "Option" and page.width(label, size, "Medium") > w - 2 * x - 34:
            page.overflow.append((field, f"zu breit: {label}"))
        page.check(x, y + 3)
        page.text(x + 34, y, label, size, "Medium")
        page.mark(field, (x, y - 2, x + 34 + page.width(label, size, "Medium"), y + 26))
        y += pitch - 8
        for line in lines[:1]:
            page.text(x + 34, y, line, 17, fill=MUTED)
            y += 24
        y += 8


# the comparison's columns: first product petrol, second orange-brown
COLUMN_COLORS = (PETROL, (186, 88, 18))


def _render_compare(page, a, b):
    """Two columns, the same fixed rows for both (Kerne, NPU ... Flash; a
    row neither product has a key for is left out), a "Zum Produkt" button
    and QR code per product at the bottom."""
    _header(page, ["Produkte", "Vergleich"])
    w, h, m = page.w, page.h, 32
    label_w = 150
    col = (w - 2 * m - label_w - 24) // 2
    xs = (m + label_w, m + label_w + col + 24)
    infos = (a, b)
    # column heads: colour bar, category + NEU, name, article number, picture
    y0, ends = 84, []
    for x, (art, info), color in zip(xs, infos, COLUMN_COLORS):
        page.rect((x, y0, x + col, y0 + 4), fill=color)
        y = y0 + 18
        box = (x + col - THUMB_W, y, x + col, y + THUMB_H)
        _picture(page, info, box)
        tw = col - THUMB_W - 14
        category = info.get("category", "")
        if category:
            page.text(x, y, category, 16, fill=MUTED)
            page.mark("Kategorie", (x, y - 2, x + page.width(category, 16), y + 22))
        if info.get("new"):
            tx = x + (page.width(category, 16) + 10 if category else 0)
            page.rect((tx, y - 1, tx + 44, y + 21), outline=ORANGE, width=2, radius=3)
            page.text(tx + 8, y + 1, "NEU", 14, "Medium", fill=ORANGE)
            page.mark("Neu", (tx, y - 1, tx + 44, y + 21))
        y += 26
        name = info.get("name") or art
        size = next((s for s in (30, 26, 22) if page.width(name, s, "Bold") <= tw), 22)
        y_name = y
        for line in page.wrap(name, tw, size, "Bold")[:2]:
            page.text(x, y, line, size, "Bold")
            y += int(size * 1.25)
        page.mark("Bezeichnung", (x, y_name, x + tw, y - 2))
        if re.match(r"^[A-Z]{2,5}-[A-Z0-9]", art):
            page.text(x, y + 2, art, 16, fill=MUTED)
            page.mark("Artikelnummer", (x, y, x + page.width(art, 16), y + 22))
            y += 24
        ends.append(max(y + 14, box[3] + 14))
    y = max(ends)
    # the rows
    cells = []
    if any("cores" in info for _, info in infos):
        cells.append(("Kerne", [(False, info["cores"] or "\u2013", not info["cores"], True)
                                if "cores" in info else (False, "\u2013", True, True)
                                for _, info in infos]))
    feats = [dict(info.get("features", [])) for _, info in infos]
    for key in FEATURES:
        if not any(key in f for f in feats):
            continue  # applies to neither (category)
        row = []
        for f in feats:
            tick, text, muted = feature_cell(key, f.get(key, ""))
            row.append((tick, text, muted, key in ("RAM", "Flash")))
        cells.append((key, row))
    bottom = _compare_band(page, infos, xs, col)
    rh = min(47, (bottom - 16 - y) // max(1, len(cells)))
    page.draw.line([(m, y), (w - m, y)], fill=RULE, width=1)
    for key, row in cells:
        page.text(m, y + (rh - 18) / 2 - 2, key, 18, fill=MUTED)
        for x, (tick, text, muted, colored), color in zip(xs, row, COLUMN_COLORS):
            tx = x
            if tick:
                page.check(x, y + (rh - 22) / 2, 22, color)
                tx += 34
            if text:
                page.text(tx, y + (rh - 22) / 2 - 3, text, 22 if not muted else 20,
                          "Bold" if not muted else "Regular",
                          fill=MUTED if muted else color if colored or key == "Kerne" else INK)
                if not muted and page.width(text, 22, "Bold") > col - (tx - x):
                    page.overflow.append((key, f"zu breit: {text}"))
        page.mark(key, (m, y + 2, w - m, y + rh - 2))
        y += rh
        page.draw.line([(m, y), (w - m, y)], fill=RULE, width=1)


def _compare_band(page, infos, xs, col, margin=32):
    """Grey band at the bottom: per product a "Zum Produkt" button in its
    column's colour with the QR code under it. Returns the band's top."""
    codes = []
    for (art, info), x, color in zip(infos, xs, COLUMN_COLORS):
        url = next((u for label, u in info.get("links", []) if label == "Zum Produkt"), None)
        q = _qr(url, 150) if url else None
        if q is not None:
            codes.append((x, color, q))
    if not codes:
        return page.h
    block = max(q.shape[0] for _, _, q in codes)
    top = page.h - margin - block - 64
    page.rect((0, top - 24, page.w, page.h), fill=BAND)
    page.draw.line([(0, top - 24), (page.w, top - 24)], fill=RULE, width=1)
    for x, color, q in codes:
        y = page.h - margin - q.shape[0]
        page.pastes.append((q, x, y))
        text = "Zum Produkt  \u203a"
        bw = page.width(text, 18, "Medium") + 40
        page.rect((x, top, x + bw, top + 44), fill=color, radius=4)
        page.text(x + 20, top + 12, text, 18, "Medium", fill=(255, 255, 255))
        page.mark("Link Produktseite", (x, top, x + max(bw, q.shape[0]), page.h - margin))
    return top - 24
