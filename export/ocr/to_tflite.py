#!/usr/bin/env python3
"""PP-OCRv5 mobile det / rec ONNX -> full-integer TFLite for the i.MX8MP NPU.

Runs in export/npu's environment (TensorFlow 2.19 = the board's runtime):

    cd export/npu
    uv run ../ocr/to_tflite.py det.onnx det_calib.npy 320x512 -o det_int8.tflite
    uv run ../ocr/to_tflite.py rec.onnx rec_calib.npy 48x320  -o rec_int8.tflite

Fixed input size (the NPU compiles one graph per shape), uint8 input and
int8 output like the YOLO model; phytec_detect.ocr quantises the
normalised picture ((BGR / 255 - 0.5) / 0.5) with the input's scale and
zero point and dequantises the output. Calibration: export/ocr/make_calib.py.
"""

import argparse
import os
import pathlib
import shutil
import sys
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx")
    parser.add_argument("calib", help="N x H x W x 3 uint8 BGR (make_calib.py)")
    parser.add_argument("size", help="HxW input size, e.g. 320x512")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--calib-count", type=int, default=200)
    args = parser.parse_args()

    import onnx
    import onnx2tf

    h, w = (int(v) for v in args.size.split("x"))
    onnx_path = pathlib.Path(args.onnx).resolve()
    out = pathlib.Path(args.output).resolve()
    name = onnx.load(str(onnx_path)).graph.input[0].name
    calib = np.load(args.calib)
    if len(calib) > args.calib_count:
        calib = calib[np.linspace(0, len(calib) - 1, args.calib_count).astype(int)]
    if calib.shape[1:3] != (h, w):
        sys.exit(f"calibration is {calib.shape[1:3]}, model input {h}x{w}")
    print(f"calibration: {calib.shape}")

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        calib_f = os.path.join(tmp, "calib.npy")
        np.save(calib_f, calib.astype(np.float32) / 255.0)
        np.save(os.path.join(tmp, "calibration_image_sample_data_20x128x128x3_float32.npy"),
                np.zeros((20, 128, 128, 3), np.float32))
        os.chdir(tmp)
        work = os.path.join(tmp, "saved_model")
        half = np.full((1, 1, 1, 3), 0.5, np.float32)
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=work,
            overwrite_input_shape=[f"{name}:1,3,{h},{w}"],
            output_integer_quantized_tflite=True,
            quant_type="per-channel",
            input_quant_dtype="uint8",
            output_quant_dtype="int8",
            custom_input_op_name_np_data_path=[[name, calib_f, half, half]],
            non_verbose=True,
        )
        produced = sorted(pathlib.Path(work).glob("*_full_integer_quant.tflite"))
        if not produced:
            sys.exit("onnx2tf produced no *_full_integer_quant.tflite")
        shutil.copy(produced[0], out)
        os.chdir(cwd)
    print(f"written: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
