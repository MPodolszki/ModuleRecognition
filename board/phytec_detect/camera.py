"""phyCAM on CSI1 in, fullscreen Wayland out.

Both are taken over from meta-hailo-examples-phytec's object_detect
(camera_cli.py), which already sorted out the board's quirks:

* The ISI's own Bayer->RGB conversion delivers black frames on this
  board/kernel, so the sensor is read as raw GRBG and debayered here.
* The sensor's horizontal blanking is reset to its 60 fps value, because
  the facematch demo leaves a wider one behind that caps the camera at
  ~24 fps.
* The appsink keeps only the newest frame, so a slow backend never works
  on stale pictures.
* The software debayer has no white balance: raw AR0144 frames come out
  dark and strongly green. WhiteBalance below fixes that (gray world plus a
  brightness gain) for the screen and the detector alike -- and the same
  function prepares training pictures on the PC, so both see one look.
"""

import subprocess
import sys

import cv2
import numpy as np

CSI1_DEVICE = "/dev/video-isi-csi1"
CSI1_SENSOR = "/dev/cam-csi1"
CSI1_WIDTH, CSI1_HEIGHT = 1280, 800
CSI1_HBLANK = 208


class WhiteBalance:
    """Gray-world white balance + brightness, as per-channel gains.

    Gains are measured on a small copy every `every` frames and smoothed,
    so the per-frame cost is one saturating multiply of the full frame.
    """

    def __init__(self, every=15, target=115.0, smooth=0.3, max_gain=4.0):
        self.every, self.target, self.smooth, self.max_gain = every, target, smooth, max_gain
        self.gains = None
        self._n = 0

    def measure(self, frame):
        small = cv2.resize(frame[..., :3], (64, 40), interpolation=cv2.INTER_AREA)
        means = small.reshape(-1, 3).mean(0) + 1e-3          # B, G, R
        gains = means.mean() / means                           # gray world
        gains *= self.target / max(means.mean(), 1.0)          # brightness
        return np.clip(gains, 0.5, self.max_gain)

    def __call__(self, frame):
        if self._n % self.every == 0:
            g = self.measure(frame)
            self.gains = g if self.gains is None else \
                (1 - self.smooth) * self.gains + self.smooth * g
        self._n += 1
        b, g, r = (float(v) for v in self.gains)
        scale = (b, g, r, 1.0) if frame.shape[2] == 4 else (b, g, r)
        return cv2.multiply(frame, scale)


def white_balance(bgr):
    """One-shot WhiteBalance for a single (training) picture."""
    return WhiteBalance(every=1, smooth=1.0)(bgr)


class Camera:
    """CSI1 phyCAM, or a video/image-sequence file for testing off-camera."""

    def __init__(self, source="csi1", white_balance=True):
        self.is_camera = source == "csi1"
        self._wb = WhiteBalance() if (white_balance and self.is_camera) else None
        if self.is_camera:
            size = f"{CSI1_WIDTH}x{CSI1_HEIGHT}"
            subprocess.run(f"setup-pipeline-csi1 -s {size} -c {size}",
                           shell=True, check=True)
            subprocess.run(["v4l2-ctl", "-d", CSI1_SENSOR, "-c",
                            f"horizontal_blanking={CSI1_HBLANK}"], check=False)
            pipeline = (
                f"v4l2src device={CSI1_DEVICE} ! "
                f"video/x-bayer,format=grbg,width={CSI1_WIDTH},height={CSI1_HEIGHT} ! "
                f"appsink drop=true max-buffers=1"
            )
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open {source!r}")

    def read(self):
        """Newest frame as BGRx (4 channels), ready for the screen."""
        ok, raw = self._cap.read()
        if not ok:
            return False, None
        if self.is_camera:
            # GRBG on the wire is what OpenCV calls GB (its names are offset
            # by one pixel) -- same conversion as object_detect.
            frame = cv2.cvtColor(raw, cv2.COLOR_BAYER_GB2BGRA)
            return True, self._wb(frame) if self._wb else frame
        return True, cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)

    def release(self):
        self._cap.release()


def screen_size():
    """Current mode of the connected output, from Weston's log."""
    size = None
    try:
        for line in open("/run/user/1000/weston.log"):
            if ", current," in line:
                size = line.split()[0].split("@")[0]
        width, height = size.split("x")
        return int(width), int(height)
    except (OSError, AttributeError, ValueError):
        return 1920, 1080


class Screen:
    """BGRx frames -> G2D scale to the screen's width -> fullscreen Wayland."""

    def __init__(self, width, height):
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        self._Gst = Gst
        screen_w, _ = screen_size()
        out_h = (screen_w * height // width) & ~1
        self._pipeline = Gst.parse_launch(
            f"appsrc name=src is-live=true do-timestamp=true format=time "
            f"max-buffers=2 leaky-type=downstream "
            f"caps=video/x-raw,format=BGRx,width={width},height={height},"
            f"framerate=0/1 ! queue max-size-buffers=2 leaky=downstream ! "
            f"imxvideoconvert_g2d ! video/x-raw,width={screen_w},height={out_h} ! "
            f"waylandsink fullscreen=true sync=false")
        self._src = self._pipeline.get_by_name("src")
        self._pipeline.set_state(Gst.State.PLAYING)

    def show(self, bgrx):
        self._src.emit("push-buffer", self._Gst.Buffer.new_wrapped(bgrx.tobytes()))

    def close(self):
        self._src.emit("end-of-stream")
        self._pipeline.set_state(self._Gst.State.NULL)


def label_frame(frame, text, color=(0, 255, 255)):
    """Solid bar with large text at the top, readable from a few metres."""
    font, scale, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3, 12
    (_, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    bar_height = text_h + baseline + 2 * margin
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_height), (0, 0, 0), -1)
    cv2.putText(frame, text, (margin, margin + text_h), font, scale, color,
                thickness, cv2.LINE_AA)
    return frame


def die(message, code=2):
    print(message, file=sys.stderr)
    sys.exit(code)
