"""Detection on another board over the LAN: its accelerator, our camera.

    python3 -m phytec_detect.remote serve -b hailo -m /opt/phytec-detect/models/products_v6_n
    phytec-detect -b remote --remote 172.25.32.142:5600 ...      (on the board with the cameras)

The board with the cameras sends every detection picture (grey, scaled to
the model's width exactly as letterbox() would scale it locally, so the
server sees the very same input); the server runs the usual pipeline --
letterbox, accelerator, decode, merge -- and answers with the detections
and its own timing. Two pictures are in flight, as on a local
accelerator: the next one travels and runs while the previous result
comes back. The client looks like a Detector to the demo (backend
"remote"), so tracking, panels and everything else work unchanged.

Protocol (TCP, little endian): a JSON hello each way, then
  frame:  b"FRM1" seq:u32 width:u16 height:u16  + width*height grey bytes
  result: b"RES1" seq:u32 n:u32 pre_ms infer_ms post_ms fps:f32  + n*6 f32
"""

import argparse
import json
import socket
import struct
import sys
import time
from collections import deque

import cv2
import numpy as np

PORT = 5600
FRAME = struct.Struct("<4sIHH")
RESULT = struct.Struct("<4sIIffff")


def _recv_exact(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError("connection closed")
        got += k
    return buf


def _send_json(sock, obj):
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack("<I", len(data)) + data)


def _recv_json(sock):
    (n,) = struct.unpack("<I", _recv_exact(sock, 4))
    return json.loads(bytes(_recv_exact(sock, n)))


# -- server: the board with the accelerator ----------------------------------

def serve(args):
    from .backends import NAMES
    from .detector import Detector
    det = Detector(args.backend, args.model_dir, conf=args.conf, hef=args.hef)
    name = NAMES.get(args.backend, args.backend)
    print(f"{name} ready: {', '.join(det.paths)} -- listening on port {args.port}", flush=True)
    srv = socket.create_server(("", args.port), reuse_port=True)
    while True:
        conn, peer = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"client {peer[0]} connected", flush=True)
        try:
            _recv_json(conn)
            _send_json(conn, {"name": name, "backend": args.backend, "labels": det.labels,
                              "imgsz": det.imgsz, "models": det.paths})
            _serve_client(conn, det)
        except (ConnectionError, OSError) as error:
            print(f"client {peer[0]} gone: {error}", flush=True)
        finally:
            conn.close()


def _serve_client(conn, det):
    pending, done = None, deque(maxlen=20)
    while True:
        # with a frame pending on the accelerator, wait briefly for the next
        # one (to start it before decoding this one), else block
        conn.settimeout(0.005 if pending else None)
        frame_msg = None
        try:
            head = _recv_exact(conn, FRAME.size)
            conn.settimeout(None)
            magic, seq, w, h = FRAME.unpack(head)
            if magic != b"FRM1":
                raise ConnectionError(f"bad frame header {magic!r}")
            gray = np.frombuffer(_recv_exact(conn, w * h), np.uint8).reshape(h, w)
            frame_msg = (seq, gray)
        except socket.timeout:
            pass
        started = None
        if frame_msg is not None:
            t0 = time.perf_counter()
            job = det.prepare(frame_msg[1])
            started = (frame_msg[0], det.submit(job), time.perf_counter() - t0)
        if pending is not None:
            seq, submitted, pre = pending
            t_inf = time.perf_counter()
            dets = det.finish(submitted).astype(np.float32)
            done.append(time.perf_counter())
            fps = (len(done) - 1) / (done[-1] - done[0]) if len(done) > 1 else 0.0
            conn.sendall(RESULT.pack(b"RES1", seq, len(dets), pre * 1000, det.ms("infer"),
                                     det.ms("post"), fps) + dets.tobytes())
        pending = started


# -- client: the board with the cameras ----------------------------------------

class RemoteDetector:
    """A Detector whose accelerator is on another board (see module doc)."""

    def __init__(self, address, conf=0.5, timeout_s=5.0):
        self.conf = conf  # what the demo shows (the server detects down to its own -c)
        host, _, port = address.partition(":")
        self.address = f"{host}:{port or PORT}"
        t0 = time.perf_counter()
        self._sock = socket.create_connection((host, int(port or PORT)), timeout=timeout_s)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _send_json(self._sock, {"hello": 1})
        hello = _recv_json(self._sock)
        self._sock.settimeout(timeout_s)
        self.load_s = time.perf_counter() - t0
        self.backend_name = "remote"
        self.display_name = f"{hello['name']} (LAN)"
        self.labels = hello["labels"]
        self.nc = len(self.labels)
        self.imgsz = hello["imgsz"]
        self.paths = [f"{self.address}: {', '.join(hello['models'])}"]
        self.times = {k: deque(maxlen=100) for k in ("pre", "infer", "post", "track", "net")}
        self.server_fps = 0.0
        self._seq = 0
        self._sent = {}  # seq -> (send time, scale)

    def __call__(self, frame, tiler=None):
        return self.run(self.prepare(frame, tiler))

    def prepare(self, frame, tiler=None):
        """Grey, scaled to the model width exactly as letterbox() would."""
        t0 = time.perf_counter()
        gray = frame if frame.ndim == 2 else cv2.cvtColor(
            frame, cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = min(self.imgsz / h, self.imgsz / w)
        new_w, new_h = round(w * scale), round(h * scale)
        if (new_w, new_h) != (w, h):
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self.times["pre"].append(time.perf_counter() - t0)
        return np.ascontiguousarray(gray), scale

    def submit(self, job):
        gray, scale = job
        self._seq += 1
        h, w = gray.shape
        self._sent[self._seq] = (time.perf_counter(), scale)
        self._sock.sendall(FRAME.pack(b"FRM1", self._seq, w, h) + gray.tobytes())
        return self._seq

    def finish(self, seq):
        while True:
            magic, rseq, n, pre_ms, infer_ms, post_ms, fps = RESULT.unpack(
                _recv_exact(self._sock, RESULT.size))
            if magic != b"RES1":
                raise ConnectionError(f"bad result header {magic!r}")
            dets = np.frombuffer(_recv_exact(self._sock, n * 24), np.float32).reshape(n, 6).copy()
            sent, scale = self._sent.pop(rseq, (None, 1.0))
            if sent is not None:
                rtt = time.perf_counter() - sent
                self.times["infer"].append(infer_ms / 1000)
                self.times["post"].append(post_ms / 1000)
                self.times["net"].append(max(0.0, rtt - (pre_ms + infer_ms + post_ms) / 1000))
            self.server_fps = fps
            if rseq == seq:
                dets[:, :4] /= scale
                return dets

    def run(self, job):
        return self.finish(self.submit(job))

    def ms(self, stage):
        values = self.times[stage]
        return 1000 * sum(values) / len(values) if values else 0.0

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phytec-detect-remote", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run detection for clients on the LAN")
    s.add_argument("-b", "--backend", default="hailo")
    s.add_argument("-m", "--model-dir", action="append", required=True)
    s.add_argument("--hef")
    s.add_argument("-c", "--conf", type=float, default=0.2)
    s.add_argument("--port", type=int, default=PORT)
    t = sub.add_parser("bench", help="send a picture N times, print the timing")
    t.add_argument("address")
    t.add_argument("image")
    t.add_argument("-n", type=int, default=100)
    args = parser.parse_args(argv)
    if args.cmd == "serve":
        serve(args)
    else:
        det = RemoteDetector(args.address)
        img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        pending, t0, n = None, time.perf_counter(), 0
        for _ in range(args.n):  # two in flight, as the demo does
            seq = det.submit(det.prepare(img))
            if pending is not None:
                dets = det.finish(pending)
                n += 1
            pending = seq
        dets = det.finish(pending)
        n += 1
        dt = time.perf_counter() - t0
        print(f"{det.display_name}: {n / dt:.1f} fps end to end, {len(dets)} detections; "
              f"server: infer {det.ms('infer'):.0f} ms, post {det.ms('post'):.0f} ms, "
              f"{det.server_fps:.1f} fps; network {det.ms('net'):.1f} ms per frame")
        det.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
