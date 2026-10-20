#!/usr/bin/env python3
"""Cut YOLO26 ONNX -> full-integer TFLite for the i.MX8MP NPU.

Runs in export/npu's own environment (TensorFlow 2.19, the board's TFLite
runtime version, so no op version the board cannot load):

    cd export/npu
    uv run to_tflite.py ../../models/phyflex_n/phyflex_n.onnx \\
                        ../../models/phyflex_n/calib_640.npy

* Calibrated on the same calib_640.npy as the Hailo HEF.
* uint8 input: the calibration range [0, 1] gives scale 1/255 and zero
  point 0, so the board feeds the letterboxed RGB pixels as they are.
* int8 outputs: dequantised on the board (six small maps), which keeps the
  whole graph on the NPU -- a float output would add a CPU Dequantize op.
* NHWC in and out (onnx2tf transposes), the layout Hailo also returns.

Afterwards the INT8 model is checked against the FP32 ONNX on the host
(CPU interpreter, no NPU) to catch a broken quantisation before it reaches
the board.
"""

import argparse
import os
import pathlib
import shutil
import sys
import tempfile

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "board"))
from phytec_detect.postprocess import decode, letterbox  # noqa: E402


def convert(onnx_path, calib_path, out_path, calib_count=256):
    import onnx
    import onnx2tf

    input_name = onnx.load(str(onnx_path)).graph.input[0].name
    calib = np.load(calib_path, mmap_mode="r")
    if len(calib) > calib_count:
        # onnx2tf calibrates single-threaded in float32: 1024 pictures took
        # over half an hour; an even spread of 256 does as well for INT8.
        calib = calib[np.linspace(0, len(calib) - 1, calib_count).astype(int)]
    calib = np.ascontiguousarray(calib)
    print(f"calibration: {calib.shape} {calib.dtype}")

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        calib_f = os.path.join(tmp, "calib.npy")
        np.save(calib_f, calib.astype(np.float32) / 255.0)
        # onnx2tf's sanity inference otherwise downloads 20 sample pictures
        # from GitHub into the cwd (and fails behind proxies); hand it ours.
        sample = np.stack([cv2.resize(img, (128, 128)) for img in calib[:20]])
        np.save(os.path.join(tmp, "calibration_image_sample_data_20x128x128x3_float32.npy"),
                sample.astype(np.float32) / 255.0)
        os.chdir(tmp)
        work = os.path.join(tmp, "saved_model")
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=work,
            output_integer_quantized_tflite=True,
            quant_type="per-channel",
            input_quant_dtype="uint8",
            output_quant_dtype="int8",
            custom_input_op_name_np_data_path=[
                [input_name, calib_f,
                 np.zeros((1, 1, 1, 3), np.float32), np.ones((1, 1, 1, 3), np.float32)]],
            # The graph is already simplified by ultralytics' export.
            not_use_onnxsim=True,
            non_verbose=True,
        )
        produced = sorted(pathlib.Path(work).glob("*_full_integer_quant.tflite"))
        if not produced:
            sys.exit("onnx2tf produced no *_full_integer_quant.tflite")
        shutil.copy(produced[0], out_path)
        fp32 = sorted(pathlib.Path(work).glob("*_float32.tflite"))
        if fp32:
            shutil.copy(fp32[0], out_path.with_name(out_path.stem.replace("_int8", "")
                                                    + "_fp32.tflite"))
        os.chdir(cwd)
    print(f"written: {out_path}")


def run_tflite(path, rgb):
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    scale, zero = inp["quantization"]
    x = rgb[None]
    if inp["dtype"] != np.uint8 or abs(scale - 1 / 255) > 1e-6 or zero != 0:
        x = np.clip(np.round(rgb / 255.0 / scale + zero), 0, 255).astype(inp["dtype"])[None]
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    outs = []
    for det in interp.get_output_details():
        y = interp.get_tensor(det["index"])
        s, z = det["quantization"]
        outs.append((y.astype(np.float32) - z) * s if s else y)
    return inp, outs


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[0]), np.maximum(a[1], b[1])
    x2, y2 = np.minimum(a[2], b[2]), np.minimum(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def check(onnx_path, tflite_path, nc, image):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rgb, _ = letterbox(cv2.imread(str(image)), sess.get_inputs()[0].shape[2])
    blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    ref = decode([m.transpose(0, 2, 3, 1) for m in sess.run(None, {sess.get_inputs()[0].name: blob})],
                 nc, conf=0.25)

    inp, outs = run_tflite(tflite_path, rgb)
    print(f"tflite input: {inp['dtype'].__name__} {list(inp['shape'])} "
          f"scale={inp['quantization'][0]:.6f} zero={inp['quantization'][1]}")
    print("tflite outputs:", [list(o.shape) for o in outs])
    got = decode(outs, nc, conf=0.25)

    matched = 0
    for r in ref:
        best = max((iou(r, g) for g in got if g[5] == r[5]), default=0.0)
        matched += best >= 0.5
    print(f"check on {pathlib.Path(image).name}: FP32 ONNX {len(ref)} / INT8 TFLite {len(got)} "
          f"detections, {matched}/{len(ref)} FP32 boxes found again (IoU >= 0.5)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx")
    parser.add_argument("calib")
    parser.add_argument("-o", "--output", help="default: <onnx dir>/<name>_int8.tflite")
    parser.add_argument("--nc", type=int, help="class count (default: from labels.txt)")
    parser.add_argument("--calib-count", type=int, default=256,
                        help="calibration pictures used, evenly spread (default 256)")
    parser.add_argument("--check-image", help="image for the FP32-vs-INT8 check")
    args = parser.parse_args()

    onnx_path = pathlib.Path(args.onnx).resolve()
    out = pathlib.Path(args.output) if args.output else \
        onnx_path.with_name(f"{onnx_path.stem}_int8.tflite")
    nc = args.nc or len((onnx_path.parent / "labels.txt").read_text().splitlines())

    convert(onnx_path, args.calib, out, args.calib_count)
    if args.check_image:
        check(onnx_path, out, nc, args.check_image)


if __name__ == "__main__":
    main()
