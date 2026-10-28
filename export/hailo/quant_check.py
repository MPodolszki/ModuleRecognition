"""Quantisation loss of a HAR on given pictures: the quantised emulator vs
full precision, decoded with the board's decoder, matched per detection.
    (in the suite container) python quant_check.py model.har pictures.npy"""
import sys
import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext

sys.path.insert(0, "/work/board")
from phytec_detect.postprocess import decode  # noqa: E402

har, pics = sys.argv[1], np.load(sys.argv[2]).astype(np.float32)
labels = [l.strip() for l in open(har.rsplit("/", 1)[0] + "/labels.txt") if l.strip()]
nc = len(labels)
runner = ClientRunner(har=har)
out = {}
for ctx_name in ("SDK_FP_OPTIMIZED", "SDK_QUANTIZED"):
    with runner.infer_context(getattr(InferenceContext, ctx_name)) as ctx:
        out[ctx_name] = runner.infer(ctx, pics, batch_size=8)


def per_image(outs, i):
    return [o[i] for o in outs] if isinstance(outs, list) else [outs[i]]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


per_class, lost, gained = {}, 0, 0
for i in range(len(pics)):
    fp = decode(per_image(out["SDK_FP_OPTIMIZED"], i), nc, conf=0.05)
    q = decode(per_image(out["SDK_QUANTIZED"], i), nc, conf=0.05)
    for d in fp:
        if d[4] < 0.3:
            continue
        m = [e for e in q if int(e[5]) == int(d[5]) and iou(d, e) > 0.5]
        sq = max((e[4] for e in m), default=0.0)
        per_class.setdefault(labels[int(d[5])], []).append((float(d[4]), float(sq)))
        lost += d[4] >= 0.5 > sq
    for e in q:
        if e[4] >= 0.5 and not any(int(d[5]) == int(e[5]) and iou(d, e) > 0.5 and d[4] >= 0.5 for d in fp):
            gained += 1
print(f"pictures {len(pics)}: lost by quantisation (fp >= 0.5, int8 < 0.5) {lost}, "
      f"only int8 >= 0.5 {gained}")
print(f"{'class':24s} {'n':>5s} {'fp':>6s} {'int8':>6s} {'diff':>6s}")
for k, v in sorted(per_class.items(), key=lambda kv: np.mean([b - a for a, b in kv[1]])):
    a = np.array(v)
    print(f"{k:24s} {len(a):5d} {a[:, 0].mean():6.2f} {a[:, 1].mean():6.2f} {(a[:, 1] - a[:, 0]).mean():+6.2f}")
