"""Inference backends. Every one takes the same letterboxed RGB uint8 picture
(640x640x3) and returns the six YOLO26 head maps as float32 NHWC arrays, so
phytec_detect.postprocess.decode works on all of them unchanged.

  hailo  Hailo-8, HEF from export/hailo (DFC 3.33 / HailoRT 4.23)
  npu    i.MX8MP NPU, INT8 TFLite through /usr/lib/libvx_delegate.so
  cpu    the same INT8 TFLite on the Cortex-A53 cores (XNNPACK)
  fp32   the unquantised cut ONNX on the CPU (onnxruntime) -- the accuracy
         reference the INT8 models are measured against
"""

import glob
import os

import numpy as np

BACKENDS = ("hailo", "npu", "cpu", "fp32", "remote")
NAMES = {"hailo": "Hailo-8", "npu": "i.MX8MP NPU", "cpu": "i.MX8MP CPU (INT8)",
         "remote": "LAN (phytec_detect.remote)",
         "fp32": "CPU FP32 (reference)"}

VX_DELEGATE = "/usr/lib/libvx_delegate.so"
# The VX delegate compiles the graph for the NPU on first invoke (can take
# a while for YOLO); caching the result makes every later start fast.
NPU_CACHE_DIR = os.environ.get("PHYTEC_DETECT_NPU_CACHE", "/var/cache/phytec-detect")


class HailoDetector:
    """HEF on the Hailo-8 via the HailoRT InferModel API, float32 outputs."""

    def __init__(self, hef_path, timeout_ms=10_000, batch_size=8):
        from hailo_platform import (HEF, FormatType, HailoSchedulingAlgorithm,
                                    VDevice)

        self.path = str(hef_path)
        self._timeout_ms = timeout_ms
        hef = HEF(self.path)
        self.input_name = hef.get_input_vstream_infos()[0].name
        self.output_names = [i.name for i in hef.get_output_vstream_infos()]

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        # Same group id -> several HEFs in this process share the one Hailo-8,
        # time-sliced by HailoRT's scheduler (stacked models in Detector).
        params.group_id = "SHARED"
        self._vdevice = VDevice(params)
        self._stack = []
        try:
            self._model = self._vdevice.create_infer_model(self.path)
            # A multi-context HEF switches contexts for every batch; batching
            # tiles spreads that cost (batch 1: ~19 ms per tile).
            self._model.set_batch_size(batch_size)
            self._model.input().set_format_type(FormatType.UINT8)
            # Let HailoRT dequantise (and handle the 16-bit outputs of the
            # Model Zoo profile) -- the host gets plain float32 maps.
            for name in self.output_names:
                self._model.output(name).set_format_type(FormatType.FLOAT32)
            configured = self._model.configure()
            self._stack.append(configured)
            self._configured = configured.__enter__()
            self._pool = {}  # index -> (bindings, {output name: buffer}), made on demand
        except BaseException:
            self.close()
            raise

    def _binding(self, i):
        if i not in self._pool:
            outputs = {name: np.empty(self._model.output(name).shape, np.float32)
                       for name in self.output_names}
            bindings = self._configured.create_bindings()
            for name, buf in outputs.items():
                bindings.output(name).set_buffer(buf)
            self._pool[i] = (bindings, outputs)
        return self._pool[i]

    def infer(self, rgb):
        return self.infer_many([rgb])[0]

    def submit_many(self, rgbs):
        """Start the pictures on the Hailo-8 and return at once; collect()
        waits for them. Two submissions may be in flight (alternating sets
        of buffers): the next frame runs on the chip while the host decodes
        the previous one."""
        self._slot = 1 - getattr(self, "_slot", 1)
        used, jobs, keep = [], [], []
        for i, rgb in enumerate(rgbs):
            bindings, outputs = self._binding(self._slot * 64 + i)
            buf = np.ascontiguousarray(rgb)
            keep.append(buf)  # alive until the chip has read it
            bindings.input(self.input_name).set_buffer(buf)
            self._configured.wait_for_async_ready(timeout_ms=self._timeout_ms)
            jobs.append(self._configured.run_async([bindings]))
            used.append(outputs)
        return jobs, used, keep

    def collect(self, token):
        jobs, used, _keep = token
        for job in jobs:
            job.wait(self._timeout_ms)
        return [[outputs[name].copy() for name in self.output_names] for outputs in used]

    def infer_many(self, rgbs):
        """Several pictures in one run(): HailoRT pipelines them through the
        device instead of one round trip each (the tiled detector's case)."""
        # Submit everything first, then wait: run() with a list of bindings
        # still went frame by frame (~19 ms per tile, vs ~6 ms per frame the
        # device manages at batch 8 in `hailortcli run`).
        used, jobs = [], []
        for i, rgb in enumerate(rgbs):
            bindings, outputs = self._binding(i)
            bindings.input(self.input_name).set_buffer(np.ascontiguousarray(rgb))
            self._configured.wait_for_async_ready(timeout_ms=self._timeout_ms)
            jobs.append(self._configured.run_async([bindings]))
            used.append(outputs)
        for job in jobs:
            job.wait(self._timeout_ms)
        return [[outputs[name].copy() for name in self.output_names] for outputs in used]

    def close(self):
        while self._stack:
            try:
                self._stack.pop().__exit__(None, None, None)
            except Exception:
                pass
        if getattr(self, "_vdevice", None) is not None:
            self._vdevice.release()
            self._vdevice = None


def _tflite_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:  # host: ai-edge-litert, or full TensorFlow
        try:
            from ai_edge_litert.interpreter import Interpreter, load_delegate
        except ImportError:
            from tensorflow.lite import Interpreter
            from tensorflow.lite.experimental import load_delegate
    return Interpreter, load_delegate


class TFLiteDetector:
    """INT8 TFLite, on the NPU (VX delegate) or the CPU."""

    def __init__(self, tflite_path, npu, num_threads=4):
        Interpreter, load_delegate = _tflite_interpreter()
        self.path = str(tflite_path)
        delegates = None
        if npu:
            try:
                os.makedirs(NPU_CACHE_DIR, exist_ok=True)
                os.environ.setdefault("VIV_VX_ENABLE_CACHE_GRAPH_BINARY", "1")
                os.environ.setdefault("VIV_VX_CACHE_BINARY_GRAPH_DIR", NPU_CACHE_DIR)
            except OSError:
                pass
            delegates = [load_delegate(VX_DELEGATE)]
        self._interp = Interpreter(model_path=self.path, experimental_delegates=delegates,
                                   num_threads=None if npu else num_threads)
        self._interp.allocate_tensors()

        inp = self._interp.get_input_details()[0]
        self._in_index, self._in_dtype = inp["index"], inp["dtype"]
        scale, zero = inp["quantization"]
        # 1/255 and 0 (to_tflite.py's calibration) means: raw pixels as they are.
        self._in_q = None if (self._in_dtype == np.uint8 and abs(scale - 1 / 255) < 1e-6
                              and zero == 0) else (scale, zero)
        self._outs = [(d["index"], *d["quantization"])
                      for d in self._interp.get_output_details()]

    def infer(self, rgb):
        x = rgb
        if self._in_q:
            scale, zero = self._in_q
            info = np.iinfo(self._in_dtype)
            x = np.clip(np.round(rgb / 255.0 / scale + zero), info.min, info.max)
        self._interp.set_tensor(self._in_index, x.astype(self._in_dtype, copy=False)[None])
        self._interp.invoke()
        out = []
        for index, scale, zero in self._outs:
            y = self._interp.get_tensor(index)
            out.append((y.astype(np.float32) - zero) * scale if scale else y)
        return out

    def infer_many(self, rgbs):
        return [self.infer(rgb) for rgb in rgbs]

    def close(self):
        self._interp = None


class OnnxDetector:
    """Unquantised cut ONNX on the CPU (onnxruntime)."""

    def __init__(self, onnx_path, num_threads=4):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        self.path = str(onnx_path)
        self._sess = ort.InferenceSession(self.path, opts, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name

    def infer(self, rgb):
        blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) * (1 / 255.0)
        return [m.transpose(0, 2, 3, 1) for m in self._sess.run(None, {self._input: blob})]

    def infer_many(self, rgbs):
        return [self.infer(rgb) for rgb in rgbs]

    def close(self):
        self._sess = None


def _find(model_dir, *patterns):
    for pattern in patterns:
        hits = sorted(glob.glob(os.path.join(model_dir, pattern)))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{model_dir}/{patterns[0]}")


def make_backend(name, model_dir, hef=None):
    """Build backend *name* from the model files in *model_dir*.

    For hailo, the Model Zoo profile HEF (*_mz.hef) is preferred over the
    plain INT8 one unless *hef* picks a file explicitly.
    """
    if name == "hailo":
        return HailoDetector(hef or _find(model_dir, "*_mz.hef", "*.hef"))
    if name in ("npu", "cpu"):
        return TFLiteDetector(_find(model_dir, "*_int8.tflite"), npu=name == "npu")
    if name == "fp32":
        cut = [p for p in sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
               if not p.endswith("_full.onnx")]
        if not cut:
            raise FileNotFoundError(f"{model_dir}/<name>.onnx")
        return OnnxDetector(cut[0])
    raise ValueError(f"unknown backend {name!r}, expected one of {BACKENDS}")
