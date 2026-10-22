"""The live screen in the phytec.de look (as the info panel): petrol header
bar, Roboto, petrol name tags, orange for lifted products.

Text is rendered with Pillow into small BGRA patches that are cached by
text, so a redraw only copies pixels -- the screen layer is redrawn up to
25 times a second. Without Pillow the OpenCV font is used.
"""

import functools
import os

import cv2
import numpy as np

FONTS = os.path.join(os.path.dirname(__file__), "fonts")
# BGR(A), from the info panel's RGB colours
PETROL = (115, 110, 0)
PETROL_LIGHT = (207, 205, 160)   # breadcrumb separators on phytec.de
ORANGE = (8, 151, 255)
WHITE = (255, 255, 255)
GREEN = (60, 160, 40)
BLUE = (200, 120, 0)             # card zone (#0078C8)
HEADER_H = 58                    # same as the info panel's breadcrumb bar
TITLE = ("PHYTEC", "KI-Produkterkennung")


@functools.lru_cache(maxsize=16)
def _font(size, weight):
    from PIL import ImageFont
    return ImageFont.truetype(os.path.join(FONTS, f"Roboto-{weight}.ttf"), size)


@functools.lru_cache(maxsize=512)
def text_patch(text, size, weight="Regular", fg=WHITE, bg=PETROL, pad=(0, 0)):
    """BGRA picture of `text` on an opaque `bg`, `pad` = (x, y) margin."""
    px, py = pad
    try:
        font = _font(size, weight)
        from PIL import Image, ImageDraw
    except (ImportError, OSError):
        scale, thick = size / 30, 2 if weight != "Regular" else 1
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        patch = np.empty((th + base + 2 * py, tw + 2 * px, 4), np.uint8)
        patch[:] = bg + (255,)
        cv2.putText(patch, text, (px, py + th), cv2.FONT_HERSHEY_SIMPLEX, scale, fg + (255,),
                    thick, cv2.LINE_AA)
        return patch
    ascent, descent = font.getmetrics()
    w = int(np.ceil(font.getlength(text)))
    img = Image.new("RGB", (w + 2 * px, ascent + descent + 2 * py), bg[::-1])
    ImageDraw.Draw(img).text((px, py), text, font=font, fill=fg[::-1])
    rgb = np.asarray(img)
    patch = np.empty(rgb.shape[:2] + (4,), np.uint8)
    patch[..., :3] = rgb[..., ::-1]
    patch[..., 3] = 255
    return patch


def blit(canvas, patch, x, y):
    """Copy a patch into a BGR(A) canvas, clipped to it."""
    h, w = canvas.shape[:2]
    x, y = int(x), int(y)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + patch.shape[1], w), min(y + patch.shape[0], h)
    if x1 <= x0 or y1 <= y0:
        return
    canvas[y0:y1, x0:x1] = patch[y0 - y:y1 - y, x0 - x:x1 - x, :canvas.shape[2]]


def header(canvas, info="", title=TITLE, accent=None):
    """Petrol bar over the full width: "PHYTEC | KI-Produkterkennung" left,
    `info` (detector speed, search ...) right. `accent` (BGR) draws a strip
    under the bar, e.g. orange while a search is shown."""
    alpha = (255,) if canvas.shape[2] == 4 else ()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], HEADER_H), PETROL + alpha, -1)
    x = 24
    brand = text_patch(title[0], 24, "Bold")
    blit(canvas, brand, x, (HEADER_H - brand.shape[0]) // 2)
    x += brand.shape[1] + 14
    cv2.line(canvas, (x, 16), (x, HEADER_H - 16), PETROL_LIGHT + alpha, 2)
    x += 14
    sub = text_patch(title[1], 20, "Medium")
    blit(canvas, sub, x, (HEADER_H - sub.shape[0]) // 2)
    if info:
        right = text_patch(info, 18, "Regular", fg=PETROL_LIGHT if accent is None else WHITE)
        blit(canvas, right, canvas.shape[1] - 24 - right.shape[1],
             (HEADER_H - right.shape[0]) // 2)
    if accent is not None:
        cv2.rectangle(canvas, (0, HEADER_H), (canvas.shape[1], HEADER_H + 5), accent + alpha, -1)
    return canvas


def tag(canvas, x, y, text, bg=PETROL, value=None):
    """Name tag (petrol, white text) with its bottom left corner at (x, y),
    kept below the header bar. `value` (e.g. "98 %") is drawn as a patch of
    its own: the name's patch stays cached while the score changes."""
    patch = text_patch(text, 20, "Medium", bg=bg, pad=(10, 4))
    if value is not None:
        patch = np.concatenate([patch[:, :-4], text_patch(value, 20, "Medium", bg=bg,
                                                          pad=(6, 4))], 1)
    y = max(int(y), HEADER_H + 6 + patch.shape[0])
    blit(canvas, patch, x, y - patch.shape[0])
