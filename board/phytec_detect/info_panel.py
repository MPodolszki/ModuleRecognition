"""Product info that slides in from the right when products are lifted.

  nothing lifted     no panel -- the table just shows boxes
  one lifted         name, article number and its options (products.json,
                     from info.tsv / summary.txt)
  two lifted         both side by side; if they share the option layout
                     (same family, same number of option rows) the rows are
                     put next to each other and differences are highlighted

The panel covers `fraction` of the screen width and eases in/out over
`slide_s`. Its picture is rendered only when its content changes; every
screen frame just blends the cached picture in at the current offset.
"""

import time

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, YELLOW, CYAN = (255, 255, 255, 255), (170, 170, 170, 255), (0, 230, 255, 255), \
    (255, 220, 120, 255)


def _option_lines(info):
    """(label, value) rows for one product."""
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
    def __init__(self, products, labels, fraction=0.5, slide_s=0.35):
        self.products, self.labels = products, labels
        self.fraction, self.slide_s = fraction, slide_s
        self._key = ()            # classes shown (or being slid out)
        self._image = None        # rendered panel for _key
        self._pos = 0.0           # 0 = hidden, 1 = fully in
        self._target = 0.0
        self._last = time.monotonic()

    def update(self, lifted_classes):
        """Classes of the lifted products, the one lifted first first."""
        key = tuple(dict.fromkeys(lifted_classes))[:2]  # unique, max two
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
        cv2.addWeighted(self._image[:, :shown], 0.93, region, 0.07, 0, dst=region)
        cv2.line(canvas, (w - shown, 0), (w - shown, h), YELLOW, 3)
        return canvas

    # -- rendering -------------------------------------------------------

    def _info(self, cls):
        art = self.labels[cls] if cls < len(self.labels) else str(cls)
        return art, self.products.get(art, {})

    def _render(self, h, w):
        img = np.zeros((h, w, 4), np.uint8)
        img[:] = (32, 28, 24, 255)
        if len(self._key) == 1:
            self._render_single(img, *self._info(self._key[0]))
        else:
            self._render_compare(img, self._info(self._key[0]), self._info(self._key[1]))
        return img

    def _render_single(self, img, art, info):
        h, w = img.shape[:2]
        x, y = 28, 60
        for line in _wrap(info.get("name") or art, w - 2 * x, 1.3, 3):
            _put(img, line, x, y, 1.3, YELLOW, 3)
            y += 48
        _put(img, art, x, y, 0.8, GREY, 2)
        y += 30
        cv2.line(img, (x, y), (w - x, y), GREY, 1)
        y += 40
        for name, value in _option_lines(info):
            if y > h - 20:
                break
            _put(img, name, x, y, 0.75, WHITE, 2)
            y += 30
            for line in _wrap(value, w - 2 * x - 20, 0.6)[:2]:
                if y > h - 20:
                    break
                _put(img, line, x + 20, y, 0.6, GREY, 1)
                y += 26
            y += 8

    def _render_compare(self, img, a, b):
        (art_a, info_a), (art_b, info_b) = a, b
        h, w = img.shape[:2]
        col = (w - 3 * 20) // 2
        xs = (20, 40 + col)
        _put(img, "Vergleich", 20, 46, 1.1, WHITE, 2)
        y0 = 96
        heights = []
        for x, art, info in ((xs[0], art_a, info_a), (xs[1], art_b, info_b)):
            y = y0
            for line in _wrap(info.get("name") or art, col, 0.9, 2)[:2]:
                _put(img, line, x, y, 0.9, YELLOW, 2)
                y += 34
            _put(img, art, x, y, 0.6, GREY, 1)
            heights.append(y + 20)
        y = max(heights)
        cv2.line(img, (20, y), (w - 20, y), GREY, 1)
        y += 34
        rows_a, rows_b = _option_lines(info_a), _option_lines(info_b)
        aligned = len(rows_a) == len(rows_b)
        if aligned:  # same option layout: row by row, differences highlighted
            for (na, va), (nb, vb) in zip(rows_a, rows_b):
                if y > h - 20:
                    break
                differ = (na, va) != (nb, vb)
                color = CYAN if differ else WHITE
                if differ:
                    cv2.rectangle(img, (14, y - 24), (w - 14, y + 8), (70, 60, 40, 255), -1)
                for x, text in ((xs[0], na), (xs[1], nb)):
                    _put(img, _wrap(text, col, 0.65)[0] if text else "-", x, y, 0.65, color,
                         2 if differ else 1)
                y += 34
        else:        # different families: each its own list
            for x, rows in ((xs[0], rows_a), (xs[1], rows_b)):
                yy = y
                for name, _ in rows:
                    if yy > h - 20:
                        break
                    _put(img, _wrap(name, col, 0.65)[0], x, yy, 0.65, WHITE, 1)
                    yy += 30
