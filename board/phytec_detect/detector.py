"""Backend + pre/post-processing, with per-stage timing.

The stages are timed separately because pre- and post-processing run on
the Cortex-A53s for every backend: only `infer` is what the accelerator
itself contributes, and that is the number to compare between Hailo-8 and
the NPU.

Several model directories can be stacked on one backend (e.g. the PHYTEC
module model on top of a COCO model): every model runs on the same
letterboxed picture, the class lists are concatenated and the detections
merged, so the module simply shows up as one more class.
"""

import json
import os
import time
from collections import deque

import numpy as np

from .backends import make_backend
from .postprocess import decode, get_labels, letterbox, to_image


class _Model:
    def __init__(self, backend, model_dir, hef):
        meta_path = os.path.join(model_dir, "model.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self.labels = get_labels(os.path.join(model_dir, "labels.txt"))
        self.imgsz = meta.get("imgsz", 640)
        self.backend = make_backend(backend, model_dir, hef=hef)


class Detector:
    def __init__(self, backend, model_dirs, conf=0.25, max_det=300, hef=None):
        if isinstance(model_dirs, str):
            model_dirs = [model_dirs]
        self.backend_name = backend
        self.conf = conf
        self.max_det = max_det

        start = time.perf_counter()
        # An explicit HEF only applies to the first (main) model.
        self.models = [_Model(backend, d, hef if i == 0 else None)
                       for i, d in enumerate(model_dirs)]
        self.load_s = time.perf_counter() - start
        sizes = {m.imgsz for m in self.models}
        if len(sizes) != 1:
            raise ValueError(f"stacked models need one input size, got {sorted(sizes)}")
        self.imgsz = sizes.pop()

        self.labels, self._offsets = [], []
        for m in self.models:
            self._offsets.append(len(self.labels))
            self.labels += m.labels
        self.nc = len(self.labels)
        self.times = {k: deque(maxlen=100) for k in ("pre", "infer", "post")}

    @property
    def paths(self):
        return [m.backend.path for m in self.models]

    def __call__(self, bgr):
        """BGR or BGRx frame -> (N, 6) x1, y1, x2, y2, score, class in frame pixels."""
        t0 = time.perf_counter()
        rgb, info = letterbox(bgr, self.imgsz)
        t1 = time.perf_counter()
        maps = [m.backend.infer(rgb) for m in self.models]
        t2 = time.perf_counter()
        dets = []
        for m, out, offset in zip(self.models, maps, self._offsets):
            d = decode(out, len(m.labels), self.imgsz, self.conf, self.max_det)
            d[:, 5] += offset
            dets.append(d)
        dets = np.concatenate(dets) if len(dets) > 1 else dets[0]
        if len(self.models) > 1:
            dets = dets[np.argsort(-dets[:, 4])][:self.max_det]
        dets = to_image(dets, info, bgr.shape)
        t3 = time.perf_counter()
        self.times["pre"].append(t1 - t0)
        self.times["infer"].append(t2 - t1)
        self.times["post"].append(t3 - t2)
        return dets

    def ms(self, stage):
        values = self.times[stage]
        return 1000 * sum(values) / len(values) if values else 0.0

    def close(self):
        for m in self.models:
            m.backend.close()
