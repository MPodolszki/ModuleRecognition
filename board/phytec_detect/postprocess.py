"""Pre/post-processing shared by every backend.

All backends run the same graph: YOLO26 cut at its six one2one head convs
(see export/cut_onnx.py), i.e. per stride 8/16/32 one box map (4 channels,
left/top/right/bottom distances in grid cells) and one class map (nc
logits). Decoding is therefore identical for Hailo-8, NPU and CPU, and only
the accelerator differs -- which is what the comparison is meant to show.

The decode reproduces ultralytics' end2end Detect head exactly
(anchors at cell centres, dist2bbox xyxy, sigmoid scores, top-k without
NMS); export/cut_onnx.py checks it against the full graph.
"""

import cv2
import numpy as np


def get_labels(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def letterbox(bgr, size=640):
    """Scale `bgr` (BGR, BGRx or 2-D gray) into a size x size RGB square,
    grey (114) padding centred.

    Matches ultralytics' LetterBox (center=True, linear interpolation), so
    the model sees the same picture it was validated on. Returns the padded
    RGB uint8 image and (scale, pad_x, pad_y) to map boxes back.
    """
    img_h, img_w = bgr.shape[:2]
    if bgr.ndim == 2 and img_h <= size and img_w <= size and (img_h == size or img_w == size):
        # A tile that needs no scaling: pad and expand in C, not numpy (the
        # numpy path cost ~20 ms per 640x640 tile on the Cortex-A53).
        pad_x, pad_y = (size - img_w) / 2, (size - img_h) / 2
        top, left = round(pad_y - 0.1), round(pad_x - 0.1)
        padded = cv2.copyMakeBorder(bgr, top, size - img_h - top, left, size - img_w - left,
                                    cv2.BORDER_CONSTANT, value=114)
        return cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB), (1.0, left, top)
    scale = min(size / img_h, size / img_w)
    new_w, new_h = round(img_w * scale), round(img_h * scale)
    if (new_w, new_h) != (img_w, img_h):
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    if bgr.ndim == 2:  # monochrome camera: same value in R, G and B
        out[top:top + new_h, left:left + new_w] = bgr[..., None]
    else:  # BGR(A) -> RGB on the small picture, not the full camera frame.
        out[top:top + new_h, left:left + new_w] = bgr[..., 2::-1]
    return out, (scale, left, top)


def _sort_outputs(outputs, nc, input_size):
    """Pair up the six HxWxC head maps by stride, whatever order a runtime
    returns them in. Box maps have 4 channels, class maps nc."""
    if nc == 4:
        raise ValueError("nc == 4 makes box and class maps indistinguishable by shape")
    levels = {}
    for out in outputs:
        out = np.asarray(out)
        if out.ndim == 4:
            out = out[0]
        h, w, c = out.shape
        stride = input_size // h
        kind = "box" if c == 4 else "cls"
        if kind == "cls" and c != nc:
            raise ValueError(f"head map {out.shape} matches neither 4 nor nc={nc} channels")
        levels.setdefault(stride, {})[kind] = out
    return [(s, levels[s]["box"], levels[s]["cls"]) for s in sorted(levels)]


_GRIDS = {}


def _grid(h, w):
    """Anchor centres (cell + 0.5) of an h x w map, flattened; cached."""
    if (h, w) not in _GRIDS:
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
        _GRIDS[h, w] = gx.reshape(-1), gy.reshape(-1)
    return _GRIDS[h, w]


def decode(outputs, nc, input_size=640, conf=0.25, max_det=300):
    """Six NHWC head maps -> detections in letterboxed model pixels.

    Returns an (N, 6) float32 array of x1, y1, x2, y2, score, class_id,
    highest score first.

    Runs on the Cortex-A53 for every frame, so it only touches what it
    must: logits are thresholded per level before anything is
    concatenated, and boxes are decoded for the surviving anchors only.
    """
    thres = np.log(conf / (1 - conf)) if 0 < conf < 1 else -np.inf
    boxes, logits, classes = [], [], []
    for stride, box, cls in _sort_outputs(outputs, nc, input_size):
        h, w = box.shape[:2]
        flat = cls.reshape(-1)
        cand = np.flatnonzero(flat > thres)
        if not cand.size:
            continue
        anchor, class_id = np.divmod(cand, nc)
        ax, ay = _grid(h, w)
        ax, ay = ax[anchor], ay[anchor]
        ltrb = box.reshape(-1, 4)[anchor]
        boxes.append(np.stack([ax - ltrb[:, 0], ay - ltrb[:, 1],
                               ax + ltrb[:, 2], ay + ltrb[:, 3]], axis=1) * stride)
        logits.append(flat[cand])
        classes.append(class_id)
    if not boxes:
        return np.zeros((0, 6), np.float32)
    boxes = np.concatenate(boxes)
    logits = np.concatenate(logits)
    classes = np.concatenate(classes)

    if logits.size > max_det:
        top = np.argpartition(logits, -max_det)[-max_det:]
        boxes, logits, classes = boxes[top], logits[top], classes[top]
    order = np.argsort(-logits, kind="stable")
    score = 1.0 / (1.0 + np.exp(-logits[order]))
    return np.concatenate([boxes[order], score[:, None], classes[order, None]],
                          axis=1).astype(np.float32)


def to_image(dets, letterbox_info, image_shape):
    """Map decode() boxes from the letterboxed square back to image pixels."""
    scale, left, top = letterbox_info
    out = dets.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - left) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - top) / scale
    img_h, img_w = image_shape[:2]
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, img_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, img_h)
    return out


def id_to_color(class_id):
    rng = np.random.RandomState(int(class_id) + 3)
    return tuple(int(c) for c in rng.randint(64, 255, size=3))


def module_quad(gray, box, margin=0.15):
    """The module's turned outline inside a detection box: 4 corners (x, y)
    of the minimum-area rectangle around the dark board on the bright table,
    or the box corners when that does not work out (module in a hand, on
    something dark, ...)."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    w, h = x2 - x1, y2 - y1
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if w < 12 or h < 12:
        return corners
    gh, gw = gray.shape[:2]
    cx0, cy0 = max(0, int(x1 - margin * w)), max(0, int(y1 - margin * h))
    cx1, cy1 = min(gw, int(x2 + margin * w)), min(gh, int(y2 + margin * h))
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return corners
    # on <= ~100 px: a few ms less per module, and a pixel or two is plenty
    step = max(1, int(max(cx1 - cx0, cy1 - cy0) / 100))
    crop = cv2.GaussianBlur(np.ascontiguousarray(gray[cy0:cy1:step, cx0:cx1:step]), (3, 3), 0)
    _, mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return corners
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    area = rect[1][0] * rect[1][1] * step * step
    # plausible: the turned rectangle fills a good part of the box and does
    # not reach far beyond it (a hand or a shadow merged with the board)
    if not 0.35 * w * h <= area <= 1.15 * w * h:
        return corners
    if cv2.contourArea(c) * step * step < 0.75 * area:  # not rectangular enough
        return corners
    return (cv2.boxPoints(rect) * step + [cx0, cy0]).astype(np.float32)


def draw(image, dets, labels, quads=None):
    """Draw image-space detections onto a BGR or BGRx frame, in place.
    labels=None draws the frames only; quads (N, 4, 2) draws turned frames
    (module_quad) instead of the boxes."""
    for i, (x1, y1, x2, y2, score, class_id) in enumerate(dets):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        color = id_to_color(class_id) + ((255,) if image.shape[2] == 4 else ())
        if quads is not None:
            q = np.round(quads[i]).astype(np.int32)
            cv2.polylines(image, [q], True, color, 3, cv2.LINE_AA)
            p1 = (int(q[:, 0].min()), int(q[:, 1].min()))
        else:
            cv2.rectangle(image, p1, p2, color, 3)
        if labels is None:
            continue
        text = f"{labels[int(class_id)]} {score * 100:.0f}%"
        font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        (tw, th), base = cv2.getTextSize(text, font, fscale, thick)
        y_text = max(p1[1], th + base + 4)
        cv2.rectangle(image, (p1[0], y_text - th - base - 4), (p1[0] + tw + 8, y_text),
                      color, -1)
        cv2.putText(image, text, (p1[0] + 4, y_text - base - 2), font, fscale,
                    (0, 0, 0) + ((255,) if image.shape[2] == 4 else ()), thick, cv2.LINE_AA)
    return image
