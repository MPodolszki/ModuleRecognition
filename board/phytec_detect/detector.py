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
from .tiling import edge_cut, merge


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
        self.times = {k: deque(maxlen=100) for k in ("pre", "infer", "post", "track")}

    @property
    def paths(self):
        return [m.backend.path for m in self.models]

    def __call__(self, frame, tiler=None):
        """BGR/BGRx/2-D gray frame -> (N, 6) x1, y1, x2, y2, score, class in
        frame pixels. With a Tiler, every tile is detected and the results
        are merged (see tiling.py)."""
        return self.run(self.prepare(frame, tiler))

    def prepare(self, frame, tiler=None):
        """CPU half before the accelerator: cut tiles, letterbox them.
        Split from run() so a worker can prepare the next frame while the
        accelerator is still busy with this one."""
        t0 = time.perf_counter()
        if tiler is None:
            pieces, scale = [(frame, (0, 0))], 1.0
            frame_size = (frame.shape[1], frame.shape[0])
        else:
            pieces, scale = list(tiler.crops(frame)), tiler.scale
            frame_size = tiler.size
        boxed = [letterbox(crop, self.imgsz) for crop, _ in pieces]
        self.times["pre"].append(time.perf_counter() - t0)
        return [(crop.shape, origin) for crop, origin in pieces], boxed, scale, frame_size

    def run(self, job):
        """Accelerator + decode/merge of a prepare()d frame."""
        return self.finish(self.submit(job))

    def submit(self, job):
        """Start a prepare()d frame on the accelerator(s); finish() decodes it.
        Backends without asynchronous submission run it right here."""
        t1 = time.perf_counter()
        rgbs = [rgb for rgb, _ in job[1]]
        tokens = [(True, m.backend.submit_many(rgbs)) if hasattr(m.backend, "submit_many")
                  else (False, m.backend.infer_many(rgbs)) for m in self.models]
        return job, tokens, t1

    def finish(self, submitted):
        """Wait for a submit()ted frame and decode/merge its detections."""
        job, tokens, t1 = submitted
        pieces, boxed, scale, frame_size = job
        maps = [m.backend.collect(token) if pending else token  # [model][piece]
                for m, (pending, token) in zip(self.models, tokens)]
        t2 = time.perf_counter()
        dets, cuts = [], []
        for k, ((shape, (x0, y0)), (_, info)) in enumerate(zip(pieces, boxed)):
            for m, out, offset in zip(self.models, maps, self._offsets):
                d = decode(out[k], len(m.labels), self.imgsz, self.conf, self.max_det)
                d = to_image(d, info, shape)
                d[:, [0, 2]] += x0
                d[:, [1, 3]] += y0
                d[:, 5] += offset
                dets.append(d)
                cuts.append(edge_cut(d, (x0, y0, x0 + shape[1], y0 + shape[0]), frame_size)
                            if len(pieces) > 1 else np.zeros(len(d), bool))
        dets = np.concatenate(dets) if dets else np.zeros((0, 6), np.float32)
        cut = np.concatenate(cuts) if cuts else np.zeros(0, bool)
        if scale != 1.0:
            dets[:, :4] /= scale
        dets = merge(dets, cut)  # also one box per product across models/classes
        if len(dets) > self.max_det:
            dets = dets[np.argsort(-dets[:, 4])][:self.max_det]
        self.times["infer"].append(t2 - t1)
        self.times["post"].append(time.perf_counter() - t2)
        return dets

    def ms(self, stage):
        values = self.times[stage]
        return 1000 * sum(values) / len(values) if values else 0.0

    def close(self):
        for m in self.models:
            m.backend.close()
