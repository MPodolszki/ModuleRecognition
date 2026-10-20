"""Product info that slides in from the right when products are lifted.

  nothing lifted     no panel -- the table just shows boxes
  one lifted         name, article number and its options (products.json,
                     from info.tsv / summary.txt)
  (either)           slides out by itself after 30 s, until the lifted
                     products change
  two lifted         both side by side; if they share the option layout
                     (same family, same number of option rows) the rows are
                     put next to each other and differences are highlighted

The panel covers `fraction` of the screen width and eases in/out over
`slide_s`. Its picture is rendered only when its content changes; every
screen frame just blends the cached picture in at the current offset.
"""

import os
import re
import time

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, YELLOW, CYAN = (255, 255, 255, 255), (170, 170, 170, 255), (0, 230, 255, 255), \
    (255, 220, 120, 255)


def _option_lines(info):
    """(label, value) rows for one product: the hand-written highlights
    (produktinfos.csv) if there are any, else the option export."""
    if info.get("highlights"):
        return [(h, "") for h in info["highlights"]]
    rows = []
    for o in info.get("options", []):
        name = " ".join(o.get("Option", "").split())
        desc = " ".join(o.get("Optionsbeschreibung", "").split())
        head = " ".join(o.get("Optionskopf Beschreibung", "").split())
        if not name or name == "-":
            continue
        rows.append((name, desc or head))
    if not rows:
        rows = [(line, "") for line in info.get("summary", [])]
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
    def __init__(self, products, labels, fraction=0.5, slide_s=0.35, show_s=30.0):
        self.products, self.labels = products, labels
        self.fraction, self.slide_s, self.show_s = fraction, slide_s, show_s
        self._first, self._seen = {}, {}  # class -> first / last time seen lifted
        self._key = ()            # classes shown (or being slid out)
        self._image = None        # rendered panel for _key
        self._pos = 0.0           # 0 = hidden, 1 = fully in
        self._target = 0.0
        self._last = time.monotonic()

    def moving(self):
        """Still sliding in or out (the screen must be redrawn)."""
        return self._pos != self._target

    def update(self, lifted_classes):
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
        key = tuple(c for c in lifted if now - self._first[c] < self.show_s)[:2]
        if key:
            self._target = 1.0
            if key != self._key:
                self._key, self._image = key, None
        else:
            self._target = 0.0    # keep the old content while sliding out

    def compose(self, canvas):
        """Blend the panel into `canvas` (BGRx) at its current slide position."""
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
        eased = 1 - (1 - self._pos) ** 3
        shown = int(pw * eased)
        if shown <= 0:
            return canvas
        region = canvas[:, w - shown:]
        region[:] = self._image[:, :shown]  # opaque, like a web page
        region[..., 3] = 255  # opaque also on the transparent overlay layer
        cv2.line(canvas, (w - shown, 0), (w - shown, h), (115, 110, 0, 255), 4)  # petrol
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
        return page.image()


# -- PHYTEC web look: white page, Roboto, petrol accents ---------------------

FONTS = os.path.join(os.path.dirname(__file__), "fonts")
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

    def check(self, x, y, s=18):
        """phytec.de list bullet: petrol box with a white tick."""
        self.rect((x, y, x + s, y + s), fill=PETROL, radius=3)
        self.draw.line([(x + 4, y + s * 0.52), (x + s * 0.42, y + s - 5), (x + s - 4, y + 5)],
                       fill=(255, 255, 255), width=3)

    def image(self):
        rgb = np.asarray(self.img) if self.img is not None else self.draw.rgb
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
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


def _render_single(page, art, info):
    name = info.get("name") or art
    category = info.get("category", "")
    section = category + "s" if category.endswith(("Module", "Kit")) else category or \
        (_family(name) if _family(name) != "Produkte" else "")
    _header(page, [part for part in ("Produkte", section, name) if part])
    x, w = 32, page.w
    bottom = _bottom_band(page, info) - 16
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
    for line in page.wrap(name, w - 2 * x, 46, "Bold"):
        page.text(x, y, line, 46, "Bold")
        y += 58
    page.mark("Bezeichnung", (x, y_title, x + max(page.width(l, 46, "Bold") for l in
                                                   page.wrap(name, w - 2 * x, 46, "Bold")), y - 4))
    if re.match(r"^[A-Z]{2,5}-[A-Z0-9]", art):  # not a placeholder folder name
        page.text(x, y + 2, f"Artikelnummer {art}", 20, fill=MUTED)
        page.mark("Artikelnummer", (x, y, x + page.width(f"Artikelnummer {art}", 20), y + 28))
    y += 40
    y_desc = y
    desc = page.wrap(info.get("description", ""), w - 2 * x, 19)
    if len(desc) > 3:
        page.overflow.append(("Kurzbeschreibung", f"{len(desc)} Zeilen, nur 3 passen"))
    for line in desc[:3]:
        page.text(x, y, line, 19, fill=INK)
        y += 26
    if y > y_desc:
        page.mark("Kurzbeschreibung", (x, y_desc - 2, w - x, y))
    y += 6
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
    y_high = y
    if not _option_lines(info):
        page.text(x, y, "Produktdaten folgen.", 20, fill=MUTED)
    for k, (label, value) in enumerate(_option_lines(info)):
        lines = page.wrap(value, w - 2 * x - 34, 17) if value else []
        need = 28 + 24 * min(len(lines), 1)
        if y + need > bottom:
            if info.get("highlights"):
                page.overflow.append(("Highlight", f"nur {k} von {len(info['highlights'])} passen"))
            break
        if info.get("highlights") and page.width(label, 21, "Medium") > w - 2 * x - 34:
            page.overflow.append(("Highlight", f"zu breit: {label}"))
        page.check(x, y + 3)
        page.text(x + 34, y, label, 21, "Medium")
        y += 28
        for line in lines[:1]:
            page.text(x + 34, y, line, 17, fill=MUTED)
            y += 24
        y += 10
    if info.get("highlights"):
        page.mark("Highlight 1-8", (x, y_high - 4, w - x, y - 8))


def _render_compare(page, a, b):
    (art_a, info_a), (art_b, info_b) = a, b
    _header(page, ["Produkte", "Vergleich"])
    w, h = page.w, page.h
    col = (w - 3 * 32) // 2
    xs = (32, 64 + col)
    y0 = 92
    ends = []
    for x, art, info in ((xs[0], art_a, info_a), (xs[1], art_b, info_b)):
        y = y0
        for line in page.wrap(info.get("name") or art, col, 32, "Bold")[:2]:
            page.text(x, y, line, 32, "Bold")
            y += 40
        page.text(x, y, art, 17, fill=MUTED)
        ends.append(y + 34)
    y = max(ends)
    page.draw.line([(32, y), (w - 32, y)], fill=RULE, width=2)
    y += 20
    rows_a, rows_b = _option_lines(info_a), _option_lines(info_b)
    if len(rows_a) == len(rows_b):  # same option layout: row by row, differences marked
        for (na, va), (nb, vb) in zip(rows_a, rows_b):
            if y > h - 50:
                break
            differ = (na, va) != (nb, vb)
            if differ:
                page.rect((20, y - 6, w - 20, y + 34), fill=ORANGE_TINT, radius=4)
                page.rect((20, y - 6, 26, y + 34), fill=ORANGE)
            for x, text in ((xs[0], na), (xs[1], nb)):
                shown = page.wrap(text, col - 30, 19)[0] if text else "-"
                page.check(x, y + 3, 16)
                page.text(x + 28, y, shown, 19, "Medium" if differ else "Regular")
            y += 44
    else:  # different families: each its own list
        for x, rows in ((xs[0], rows_a), (xs[1], rows_b)):
            yy = y
            if not rows:
                page.text(x, yy, "Produktdaten folgen.", 19, fill=MUTED)
            for label, _ in rows:
                if yy > h - 50:
                    break
                page.check(x, yy + 3, 16)
                page.text(x + 28, yy, page.wrap(label, col - 30, 19)[0], 19)
                yy += 40
