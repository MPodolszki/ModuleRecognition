"""Tracking (which products are lifted, their turned outlines, the still
pictures) in a process of its own.

With everything in one Python process, tracking, drawing and the
accelerator threads share one interpreter lock; on the i.MX8MP's four
A53 cores that cost more than the work itself. The grey frame (2 MB at
1920x1080) goes over shared memory, the detections and results over a
pipe; the child runs exactly the same code as the in-process tracking
(TableTracker.update, Worker._quads, _snap_track).
"""

import multiprocessing as mp
import types
from multiprocessing import shared_memory

import numpy as np


class TrackProcess:
    def __init__(self):
        ctx = mp.get_context("spawn")  # not fork: cameras, GStreamer and HailoRT are open
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_child, args=(child,), daemon=True, name="pd-track")
        self._proc.start()
        child.close()
        self._shm, self._size = None, 0
        self.snaps = {}

    def run(self, gray, dets):
        """Track one frame; returns (lifted (N, 6), quads (N, 4, 2), compute s)."""
        h, w = gray.shape
        if self._shm is None or h * w > self._size:
            if self._shm is not None:
                self._shm.close()
                self._shm.unlink()
            self._size = h * w
            self._shm = shared_memory.SharedMemory(create=True, size=self._size)
            self._conn.send(("shm", self._shm.name))
        np.ndarray((h, w), np.uint8, buffer=self._shm.buf)[:] = gray
        self._conn.send(("track", h, w, dets))
        kind, *payload = self._conn.recv()
        if kind == "error":
            raise RuntimeError(f"tracking process: {payload[0]}")
        lifted, quads, snaps, compute_s = payload
        if snaps is not None:
            self.snaps = snaps
        return lifted, quads, compute_s

    def reset(self):
        self._conn.send(("reset",))

    def close(self):
        try:
            self._conn.send(("stop",))
        except (OSError, BrokenPipeError):
            pass
        self._proc.join(2)
        if self._proc.is_alive():
            self._proc.kill()
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
            self._shm = None


def _child(conn):
    import ctypes
    import signal
    import time
    try:
        ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
        ctypes.CDLL(None).prctl(15, b"pd-track", 0, 0, 0)    # PR_SET_NAME
    except (OSError, AttributeError):
        pass
    from .live_cli import Worker, _snap_track
    from .tracking import TableTracker

    state = types.SimpleNamespace(tracker=TableTracker(), snaps={}, _quad_cache=[])
    shm, sent_snaps = None, None
    while True:
        msg = conn.recv()
        kind = msg[0]
        if kind == "stop":
            break
        if kind == "reset":
            state.tracker = TableTracker()
            continue
        if kind == "shm":
            if shm is not None:
                shm.close()
            try:  # the parent owns (and unlinks) it
                shm = shared_memory.SharedMemory(name=msg[1], track=False)
            except TypeError:  # Python < 3.13
                shm = shared_memory.SharedMemory(name=msg[1])
                from multiprocessing import resource_tracker
                resource_tracker.unregister(shm._name, "shared_memory")
            continue
        _, h, w, dets = msg
        try:
            t0 = time.perf_counter()
            gray = np.ndarray((h, w), np.uint8, buffer=shm.buf)
            state.tracker.update(gray, dets)
            lifted = np.array([[*t.box, t.score, t.cls] for t in state.tracker.picked()],
                              np.float32).reshape(-1, 6)
            quads = Worker._quads(state, gray, dets)
            _snap_track(state, gray, dets, quads)
            # still pictures only when they changed (they are ~180 kB each)
            key = {c: id(p) for c, p in state.snaps.items()}
            snaps = None if key == sent_snaps else dict(state.snaps)
            sent_snaps = key
            conn.send(("ok", lifted, quads, snaps, time.perf_counter() - t0))
        except Exception as error:  # noqa: BLE001 -- reported to the demo
            conn.send(("error", f"{type(error).__name__}: {error}"))
    if shm is not None:
        shm.close()
