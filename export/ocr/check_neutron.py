#!/usr/bin/env python3
"""det_neutron.tflite against det_int8.tflite, on the build host.

    export/ocr/check_neutron.py models/ocr [N]

Runs N calibration cards (det_calib.npy, export/ocr/make_calib.py) through
det_int8.tflite on the TFLite CPU kernels and through the Neutron model in
NXP's neutron-runner, which executes the NeutronGraph nodes bit-exactly as
the i.MX95 NPU does. The runner needs the original subgraphs inside the
model, so the model checked is converted here, by to_neutron.sh with
--keep-graphs -- the same conversion as det_neutron.tflite, plus the graphs.
Both get the same quantised input phytec_detect.ocr feeds; compared are the
text mask (prob > 0.3) and the text lines found in it, with the line
finder's rules (dilate 2x2, sides >= 3 px, mean probability >= 0.5): a line
counts as found again when a box of the other side overlaps it by IoU > 0.5.

NEUTRON_RUNNER / NEUTRON_CONVERTER override the tools on PATH.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np


def lines(prob):
    """Axis-aligned boxes of the text lines phytec_detect.ocr would keep."""
    import cv2
    mask = cv2.dilate((prob > 0.3).astype(np.uint8), np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        _, (bw, bh), _ = cv2.minAreaRect(c)
        if min(bw, bh) < 3:
            continue
        m = np.zeros_like(mask)
        cv2.drawContours(m, [c], -1, 1, -1)
        if prob[m > 0].mean() >= 0.5:
            boxes.append(cv2.boundingRect(c))
    return boxes


def found(boxes, others):
    """How many of `boxes` an `others` box overlaps by IoU > 0.5."""
    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        h = max(0, min(ay + ah, by + bh) - max(ay, by))
        return w * h / float(aw * ah + bw * bh - w * h)
    return sum(any(iou(a, b) > 0.5 for b in others) for a in boxes)


def main():
    model_dir = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    runner = os.environ.get("NEUTRON_RUNNER", "neutron-runner")
    from ai_edge_litert.interpreter import Interpreter

    cpu = Interpreter(model_path=os.path.join(model_dir, "det_int8.tflite"))
    cpu.allocate_tensors()
    inp, out = cpu.get_input_details()[0], cpu.get_output_details()[0]
    scale, zero = inp["quantization"]
    o_scale, o_zero = out["quantization"]
    calib = np.load(os.path.join(model_dir, "det_calib.npy"), mmap_mode="r")
    picks = np.linspace(0, len(calib) - 1, count).astype(int)

    ious, n_ref, n_npu, hit_ref, hit_npu = [], 0, 0, 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        checked = os.path.join(tmp, "det_neutron_check.tflite")
        subprocess.run([os.path.join(os.path.dirname(os.path.abspath(__file__)), "to_neutron.sh"),
                        os.path.join(model_dir, "det_int8.tflite"), checked, "--keep-graphs"],
                       check=True, stdout=subprocess.DEVNULL)
        for i in picks:
            # phytec_detect.ocr: (BGR / 255 - 0.5) / 0.5, quantised with the
            # input's scale and zero point
            x = (calib[i].astype(np.float32) / 255.0 - 0.5) / 0.5
            q = np.clip(np.round(x / scale + zero), np.iinfo(inp["dtype"]).min,
                        np.iinfo(inp["dtype"]).max).astype(inp["dtype"])[None]
            cpu.set_tensor(inp["index"], q)
            cpu.invoke()
            ref = cpu.get_tensor(out["index"])
            src, res = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
            q.tofile(src)
            subprocess.run([runner, "--input", checked,
                            "--dataset", src, "--output-results", res],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            npu = np.fromfile(res, dtype=out["dtype"])[: ref.size].reshape(ref.shape)
            p_ref = (ref.astype(np.float32) - o_zero) * o_scale
            p_npu = (npu.astype(np.float32) - o_zero) * o_scale
            a, b = p_ref > 0.3, p_npu > 0.3
            ious.append(float((a & b).sum() / max((a | b).sum(), 1)))
            l_ref, l_npu = lines(p_ref[0, :, :, 0]), lines(p_npu[0, :, :, 0])
            f_ref, f_npu = found(l_ref, l_npu), found(l_npu, l_ref)
            n_ref, n_npu, hit_ref, hit_npu = (n_ref + len(l_ref), n_npu + len(l_npu),
                                              hit_ref + f_ref, hit_npu + f_npu)
            print(f"card {i:4d}: text mask IoU {ious[-1]:.4f}, lines CPU {len(l_ref)} / NPU "
                  f"{len(l_npu)}, CPU lines found by the NPU {f_ref}, NPU lines in CPU {f_npu}")
    print(f"{count} cards: text mask IoU min {min(ious):.4f}, mean {np.mean(ious):.4f}; "
          f"CPU lines found by the NPU {hit_ref}/{n_ref}, NPU lines also on the CPU "
          f"{hit_npu}/{n_npu}")


if __name__ == "__main__":
    main()
