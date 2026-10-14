#!/usr/bin/env python3
"""Cut YOLO26 ONNX -> Hailo-8 HEF (DFC 3.33, runs inside the AI SW Suite).

Started by export/hailo/run.sh, which mounts the project at /work:

    export/hailo/run.sh models/phyflex_n/phyflex_n.onnx models/phyflex_n/calib_640.npy

The ONNX is already cut at the six one2one head convs (export/cut_onnx.py),
i.e. exactly the parse range Hailo Model Zoo v2.18 uses for YOLO26
(cfg/base/yolo26.yaml), so the whole graph is parsed and the six maps come
out as the HEF's outputs. Decoding happens on the board
(phytec_detect.postprocess), same as for the NPU.

Quantisation profiles:
  mz    Hailo Model Zoo v2.18 yolo26n/s recipe: optimization level 4 with
        AdaRound, 16-bit head convs/outputs and four depthwise layers. Best
        Hailo accuracy, needs a GPU in the container to finish in
        reasonable time.
  int8  Plain INT8 everywhere, like the NPU's TFLite model -- the
        apples-to-apples setting for the accuracy comparison.

The normalisation (/255) is compiled into the HEF, so the board feeds the
letterboxed RGB uint8 picture unchanged -- like the NPU model.
"""

import argparse
import pathlib
import sys

import numpy as np

# Layer names are those the DFC assigns when parsing ultralytics' YOLO26n/s
# graph cut at the one2one head (same as the Model Zoo's yolo26n/s.alls).
# yolo26m numbers its head convs differently (see yolo26m.alls).
MZ_16BIT = {
    "n": ["conv61", "conv77", "conv91", "conv64", "conv80", "conv94"],
    "s": ["conv61", "conv77", "conv91", "conv64", "conv80", "conv94"],
    "m": ["conv101", "conv104", "conv87", "conv74", "conv71", "conv90"],
}


def model_script(profile, size, opt_level, calib_size, batch):
    lines = ["normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])",
             f"model_optimization_config(calibration, batch_size={batch}, "
             f"calibset_size={calib_size})"]
    if profile == "mz":
        level = 4 if opt_level is None else opt_level
        lines += [
            f"model_optimization_flavor(optimization_level={level}, compression_level=0)",
            f"post_quantization_optimization(adaround, policy=enabled, "
            f"batch_size={1 if size == 'm' else batch})",
            "quantization_param([dw1, dw6, dw7, dw8], precision_mode=a16_w16)",
            f"quantization_param([{', '.join(MZ_16BIT[size])}], precision_mode=a16_w16)",
            "quantization_param([output_layer1, output_layer2, output_layer3, "
            "output_layer4, output_layer5, output_layer6], precision_mode=a16_w16)",
        ]
    else:
        level = 2 if opt_level is None else opt_level
        lines.append(f"model_optimization_flavor(optimization_level={level}, "
                     f"compression_level=0)")
    return "\n".join(lines) + "\n"


def emulator_check(runner, calib, nc):
    """Quantised emulator vs. full precision on a few calibration pictures,
    decoded with the board's own decoder."""
    from hailo_sdk_client import InferenceContext

    sys.path.insert(0, "/work/board")
    from phytec_detect.postprocess import decode

    data = calib[:8].astype(np.float32)
    results = {}
    for ctx_name in ("SDK_FP_OPTIMIZED", "SDK_QUANTIZED"):
        with runner.infer_context(getattr(InferenceContext, ctx_name)) as ctx:
            results[ctx_name] = runner.infer(ctx, data)

    def per_image(outs, i):
        return [o[i] for o in outs] if isinstance(outs, list) else [outs[i]]

    for i in range(len(data)):
        fp = decode(per_image(results["SDK_FP_OPTIMIZED"], i), nc, conf=0.25)
        q = decode(per_image(results["SDK_QUANTIZED"], i), nc, conf=0.25)
        top = lambda d: f"{d[0, 4]:.2f}" if len(d) else "-"
        print(f"  calib[{i}]: fp {len(fp):3d} det (top {top(fp)})   "
              f"quantized {len(q):3d} det (top {top(q)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx")
    parser.add_argument("calib")
    parser.add_argument("--profile", choices=("mz", "int8"), default="mz")
    parser.add_argument("--size", choices=("n", "s", "m"),
                        help="YOLO26 size, for the mz layer names (default: from file name)")
    parser.add_argument("--opt-level", type=int,
                        help="override the profile's optimization level (0-4)")
    parser.add_argument("--calib-size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hw-arch", default="hailo8")
    parser.add_argument("-o", "--output", help="default: <onnx dir>/<name>_<profile>.hef")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    from hailo_sdk_client import ClientRunner

    onnx_path = pathlib.Path(args.onnx).resolve()
    name = onnx_path.stem
    size = args.size or next((s for s in "nsm" if f"26{s}" in name or name.endswith(f"_{s}")), "n")
    hef_path = pathlib.Path(args.output) if args.output else \
        onnx_path.with_name(f"{name}_{args.profile}.hef")
    labels = onnx_path.parent / "labels.txt"
    nc = len(labels.read_text().splitlines()) if labels.exists() else None

    calib = np.load(args.calib)
    calib_size = min(args.calib_size, len(calib))
    if len(calib) < args.calib_size:
        print(f"note: {len(calib)} calibration pictures, fewer than --calib-size "
              f"{args.calib_size}; using all of them")

    runner = ClientRunner(hw_arch=args.hw_arch)

    print(f"== parse {onnx_path.name} (YOLO26{size}, {args.profile})")
    runner.translate_onnx_model(str(onnx_path), name,
                                net_input_shapes={"images": [1, 3, 640, 640]})

    script = model_script(args.profile, size, args.opt_level, calib_size, args.batch)
    print("== model script\n" + script)
    runner.load_model_script(script)

    print(f"== optimize: {calib.shape} {calib.dtype}")
    runner.optimize(calib[:calib_size].astype(np.float32))
    runner.save_har(str(hef_path.with_suffix(".har")))

    if not args.no_check and nc:
        print("== emulator check (board decoder)")
        try:
            emulator_check(runner, calib, nc)
        except Exception as error:  # informative only, never fail the build on it
            print(f"  skipped: {error}")

    print("== compile")
    hef_path.write_bytes(runner.compile())
    print(f"== written: {hef_path}")


if __name__ == "__main__":
    main()
