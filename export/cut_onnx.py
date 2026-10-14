#!/usr/bin/env python3
"""best.pt -> YOLO26 ONNX cut at the six one2one head convs.

The cut graph is the single source for both accelerators:
  * Hailo-8:  export/hailo/compile_hef.py (DFC 3.33, AI SW Suite container)
  * i.MX8MP:  export/npu/to_tflite.py     (onnx2tf -> INT8 TFLite, VX delegate)
and its six outputs are decoded on the board by phytec_detect.postprocess.

Everything after these convs (anchor decode, sigmoid, top-k) is what the
Hailo Model Zoo also leaves to the host for YOLO26, and what the VX delegate
would push back to the CPU anyway -- cutting it keeps both accelerators on
exactly the same graph.

The script also checks the board decoder against the full end2end graph,
so a later ultralytics change in the head cannot silently break the demo.

    uv run export/cut_onnx.py runs/detect/phyflex_n/weights/best.pt
"""

import argparse
import json
import pathlib
import re
import sys

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.utils import ASSETS

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "board"))
from phytec_detect.postprocess import decode, letterbox  # noqa: E402

HEAD_NODES = [f"/model.23/one2one_cv{kind}.{level}/one2one_cv{kind}.{level}.2/Conv"
              for level in range(3) for kind in (2, 3)]


def export_full(weights, imgsz, out_dir):
    model = YOLO(weights)
    # nms=False is what selects the NMS-free one2one head (ultralytics sets
    # model.end2end = args.nms is False); the default exports one2many + NMS.
    path = model.export(format="onnx", imgsz=imgsz, opset=13, simplify=True,
                        nms=False, dynamic=False)
    full = out_dir / f"{pathlib.Path(weights).stem}_full.onnx"
    pathlib.Path(path).replace(full)
    return model, full


def cut(full_path, cut_path, imgsz, nc):
    graph = onnx.load(str(full_path))
    by_name = {node.name: node for node in graph.graph.node}
    missing = [n for n in HEAD_NODES if n not in by_name]
    if missing:
        raise SystemExit(f"head nodes not found (ultralytics head changed?): {missing}")

    outputs = [by_name[n].output[0] for n in HEAD_NODES]
    onnx.utils.extract_model(str(full_path), str(cut_path),
                             [graph.graph.input[0].name], outputs)

    # Readable, stable output names: box_s8, cls_s8, box_s16, ...
    model = onnx.load(str(cut_path))
    rename = {}
    for node_name, tensor in zip(HEAD_NODES, outputs):
        kind_id, level = map(int, re.search(r"one2one_cv(\d)\.(\d)/", node_name).groups())
        kind = "box" if kind_id == 2 else "cls"
        rename[tensor] = f"{kind}_s{8 << level}"
    for node in model.graph.node:
        node.output[:] = [rename.get(o, o) for o in node.output]
    for out in model.graph.output:
        out.name = rename.get(out.name, out.name)
    model.graph.output.sort(key=lambda o: (int(o.name.split("_s")[1]), o.name))
    onnx.checker.check_model(model)
    onnx.save(model, str(cut_path))

    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        channels = 4 if out.name.startswith("box") else nc
        grid = imgsz // int(out.name.split("_s")[1])
        assert shape == [1, channels, grid, grid], (out.name, shape)
        print(f"  {out.name:8s} {shape}")


def verify(full_path, cut_path, imgsz, nc, image):
    bgr = cv2.imread(str(image))
    rgb, _ = letterbox(bgr, imgsz)
    blob = (rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)

    full = ort.InferenceSession(str(full_path), providers=["CPUExecutionProvider"])
    ref = full.run(None, {full.get_inputs()[0].name: blob})[0][0]  # (300, 6)

    sess = ort.InferenceSession(str(cut_path), providers=["CPUExecutionProvider"])
    maps = sess.run(None, {sess.get_inputs()[0].name: blob})
    ours = decode([m.transpose(0, 2, 3, 1) for m in maps], nc, imgsz, conf=0.0,
                  max_det=len(ref))

    ref = ref[ref[:, 4] > 0.25]
    got = ours[ours[:, 4] > 0.25]
    if len(ref) != len(got):
        raise SystemExit(f"decoder mismatch: {len(got)} vs {len(ref)} detections > 0.25")
    if len(ref):
        err = np.abs(got[:, :5] - ref[:, :5]).max()
        same_cls = np.array_equal(got[:, 5], ref[:, 5])
        if err > 1e-2 or not same_cls:
            raise SystemExit(f"decoder mismatch: max abs err {err:.4f}, classes equal: {same_cls}")
        print(f"  decoder check: {len(ref)} detections, max abs err {err:.2e} -> OK")
    else:
        print("  decoder check: no detection > 0.25 on the check image (nothing compared)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out-dir", default=str(ROOT / "models"))
    parser.add_argument("--name", help="base name of the outputs (default: weights' run name)")
    parser.add_argument("--check-image", default=str(ASSETS / "bus.jpg"),
                        help="image the board decoder is checked on")
    args = parser.parse_args()

    weights = pathlib.Path(args.weights)
    name = args.name or (weights.parents[1].name if weights.parent.name == "weights"
                         else weights.stem)
    out_dir = pathlib.Path(args.out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== export {weights}")
    model, full = export_full(weights, args.imgsz, out_dir)
    names = [model.names[i] for i in range(len(model.names))]
    nc = len(names)

    cut_path = out_dir / f"{name}.onnx"
    print(f"== cut -> {cut_path}")
    cut(full, cut_path, args.imgsz, nc)

    print("== verify board decoder against the full end2end graph")
    verify(full, cut_path, args.imgsz, nc, args.check_image)

    (out_dir / "labels.txt").write_text("\n".join(names) + "\n")
    (out_dir / "model.json").write_text(json.dumps(
        {"name": name, "imgsz": args.imgsz, "nc": nc, "names": names,
         "outputs": ["box_s8", "cls_s8", "box_s16", "cls_s16", "box_s32", "cls_s32"],
         "input": "RGB uint8 letterboxed, /255 inside the model or by the runtime"},
        indent=2) + "\n")
    print(f"== done: {out_dir}")


if __name__ == "__main__":
    main()
