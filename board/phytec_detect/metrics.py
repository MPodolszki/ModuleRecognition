"""mAP50 / mAP50-95 the way ultralytics computes them (COCO 101-point
interpolation, IoU thresholds 0.50:0.05:0.95), in plain numpy so it runs on
the board without torch. Checked against `yolo val` by
tools/check_metrics.py.
"""

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def box_iou(a, b):
    """(N, 4) x (M, 4) xyxy -> (N, M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(dets, gt_boxes, gt_classes):
    """True-positive matrix (N dets, 10 thresholds), ultralytics' matching:
    per threshold, highest IoU first, each ground truth and detection used
    once, classes must agree."""
    tp = np.zeros((len(dets), len(IOU_THRESHOLDS)), bool)
    if len(dets) == 0 or len(gt_boxes) == 0:
        return tp
    iou = box_iou(gt_boxes, dets[:, :4])
    iou = iou * (gt_classes[:, None] == dets[None, :, 5])
    for t, thres in enumerate(IOU_THRESHOLDS):
        m = np.array(np.nonzero(iou >= thres)).T  # (gt, det) pairs
        if not len(m):
            continue
        if len(m) > 1:
            # Verbatim ultralytics BaseValidator.match_predictions: no re-sort
            # between the two unique() steps, so a ground truth goes to the
            # lowest-index (= highest-score) detection, not the best IoU.
            m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
        tp[m[:, 1], t] = True
    return tp


def _ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return np.trapezoid(np.interp(x, mrec, mpre), x) if hasattr(np, "trapezoid") \
        else np.trapz(np.interp(x, mrec, mpre), x)


class MeanAP:
    """Accumulate images, then summary()."""

    def __init__(self, nc):
        self.nc = nc
        self._tp, self._conf, self._cls, self._gt = [], [], [], []

    def add(self, dets, gt_boxes, gt_classes):
        self._tp.append(match(dets, gt_boxes, gt_classes))
        self._conf.append(dets[:, 4])
        self._cls.append(dets[:, 5])
        self._gt.append(gt_classes)

    def summary(self, conf_for_pr=0.25):
        tp = np.concatenate(self._tp) if self._tp else np.zeros((0, 10), bool)
        conf = np.concatenate(self._conf) if self._conf else np.zeros(0)
        cls = np.concatenate(self._cls) if self._cls else np.zeros(0)
        gt = np.concatenate(self._gt) if self._gt else np.zeros(0)

        order = np.argsort(-conf)
        tp, conf, cls = tp[order], conf[order], cls[order]
        ap = np.zeros((self.nc, 10))
        p_at, r_at = np.zeros(self.nc), np.zeros(self.nc)
        present = np.zeros(self.nc, bool)
        for c in range(self.nc):
            n_gt = int((gt == c).sum())
            sel = cls == c
            if n_gt == 0:
                continue
            present[c] = True
            if not sel.any():
                continue
            tpc = np.cumsum(tp[sel], axis=0)
            fpc = np.cumsum(~tp[sel], axis=0)
            recall = tpc / (n_gt + 1e-16)
            precision = tpc / (tpc + fpc)
            for t in range(10):
                ap[c, t] = _ap(recall[:, t], precision[:, t])
            k = int((conf[sel] >= conf_for_pr).sum())
            if k:
                p_at[c], r_at[c] = precision[k - 1, 0], recall[k - 1, 0]
        if not present.any():
            return {"mAP50": 0.0, "mAP50-95": 0.0, "P": 0.0, "R": 0.0, "classes": 0}
        return {"mAP50": float(ap[present, 0].mean()),
                "mAP50-95": float(ap[present].mean()),
                f"P@{conf_for_pr}": float(p_at[present].mean()),
                f"R@{conf_for_pr}": float(r_at[present].mean()),
                "classes": int(present.sum())}
