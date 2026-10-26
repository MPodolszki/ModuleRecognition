"""phyCAMs in, fullscreen Wayland out.

Two cameras, found by their sensor driver:

  AR0521 (VM-017, monochrome)  detection camera, full 2592x1944 GRAY8,
                               behind the FPD-Link deserializer on CSI1
  AR0521 (VM-017, colour)      live picture, raw Bayer binned to 1296x972,
                               behind the FPD-Link deserializer on CSI2
  AR0144 (VM-016, colour)      live picture, 1280x800 raw Bayer

Monochrome and colour AR0521 share a driver name; the media bus codes the
sensor offers (Y8 vs. Bayer) tell them apart.

The AR0144 path is taken over from meta-hailo-examples-phytec's
object_detect, which already sorted out the board's quirks:

* The ISI's own Bayer->RGB conversion delivers black frames on this
  board/kernel, so the sensor is read as raw GRBG and debayered here.
* The sensor's horizontal blanking is reset to its 60 fps value, because
  the facematch demo leaves a wider one behind that caps the camera at
  ~24 fps.
* The software debayer has no white balance: raw AR0144 frames come out
  dark and strongly green. WhiteBalance below fixes that (gray world plus a
  brightness gain) for the screen and the detector alike -- and the same
  function prepares training pictures on the PC, so both see one look.

The AR0521 driver has no auto exposure (fixed 1817 lines, far too bright
for a lit table), so AutoExposure drives its `exposure` control.

Every appsink keeps only the newest frame, so a slow consumer never works
on stale pictures.
"""

import functools
import glob
import os
import re
import subprocess
import sys

import cv2
import numpy as np

SENSORS = {
    # (name prefix, monochrome?): how to run it
    ("ar0521", True): {"mbus": "Y8_1X8", "size": (2592, 1944),
                       "caps": "video/x-raw,format=GRAY8"},
    # colour AR0521 (VM-017 colour): live picture, 2x2-binned to 1296x972
    ("ar0521", False): {"mbus": "SGRBG8_1X8", "size": (2592, 1944), "out": (1296, 972),
                        "caps": "video/x-bayer,format=grbg"},
    ("ar0144", False): {"mbus": None, "size": (1280, 800),
                        "caps": "video/x-bayer,format=grbg", "hblank": 208},
}


@functools.lru_cache(maxsize=1)
def soc():
    """'imx95', 'imx8mp' or '' -- from the device tree's compatible list."""
    try:
        compatible = open("/proc/device-tree/compatible", "rb").read().split(b"\0")
    except OSError:
        return ""
    for name in ("imx95", "imx8mp"):
        if f"fsl,{name}".encode() in compatible:
            return name
    return ""


# The i.MX95 has no imx8-phycam-isp service with modes: its Neo ISP runs
# behind libcamera, which scales to whatever size is asked for. 1920x1080 is
# what the i.MX8MP demo runs in (ISP mode 1), so the model sees the same
# picture size; PHYTEC_ISP_SIZE=WxH overrides it.
IMX95_ISP_SIZE = (1920, 1080)


def isp_size(csi):
    """Output size of the ISP on CSI1/CSI2. i.MX8MP: the mode imx8-phycam-isp
    runs, from its SensorN_Entry.cfg (mode = N, [mode.N] xml = ..._WxH.xml);
    i.MX95: IMX95_ISP_SIZE or PHYTEC_ISP_SIZE."""
    if soc() == "imx95":
        try:
            width, height = os.environ["PHYTEC_ISP_SIZE"].split("x")
            return int(width), int(height)
        except (KeyError, ValueError):
            return IMX95_ISP_SIZE
    try:
        cfg = open(f"/opt/imx8-isp/bin/Sensor{csi - 1}_Entry.cfg").read()
        mode = cfg.split("mode=")[1].split()[0]
        xml = cfg.split(f"[mode.{mode}]")[1].split(".xml")[0]
        width, height = xml.rsplit("_", 1)[1].split("x")
        return int(width), int(height)
    except (OSError, IndexError, ValueError):
        return None


def is_mono(subdev):
    """A monochrome sensor offers Y8/Y10/Y12, a colour one Bayer codes."""
    try:
        codes = subprocess.run(["v4l2-ctl", "-d", subdev, "--list-subdev-mbus-codes"],
                               capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "MEDIA_BUS_FMT_Y8" in codes and "MEDIA_BUS_FMT_S" not in codes


def find_sensor(csi):
    """(subdev path, sensor name, fpd-link port or None) on CSI `csi`, or None."""
    for path in [f"/dev/cam-csi{csi}"] + sorted(glob.glob(f"/dev/cam-csi{csi}-port*")):
        if not os.path.exists(path):
            continue
        subdev = os.path.basename(os.path.realpath(path))
        try:
            name = open(f"/sys/class/video4linux/{subdev}/name").read().strip()
        except OSError:
            continue
        port = int(path.rsplit("port", 1)[1]) if "-port" in path else None
        return path, name, port
    return None


def libcamera_id(csi):
    """libcamera's name for the sensor on CSI `csi`: its device tree path,
    e.g. /base/soc/bus@42000000/i2c@42530000/camera@36 -- or None."""
    found = find_sensor(csi)
    if not found:
        return None
    subdev = os.path.basename(os.path.realpath(found[0]))
    node = os.path.realpath(f"/sys/class/video4linux/{subdev}/device/of_node")
    base = "/sys/firmware/devicetree"
    return node[len(base):] if node.startswith(base + "/base/") else None


def isp_source(csi, caps):
    """GStreamer source for the ISP's picture on CSI `csi`, ending in `caps`.

    i.MX8MP: imx8-phycam-isp's V4L2 node. i.MX95: libcamerasrc on the Neo ISP
    (the imx95-phyflex-libra-rdk-neoisp overlay), which also scales -- there
    is no mode to set up beforehand."""
    if soc() == "imx95":
        name = libcamera_id(csi)
        if not name:
            raise RuntimeError(f"no camera on CSI{csi}")
        # The ISI pipeline handler claims the sensors too and then fails
        # ("Cannot find a supported YUV/RGB format"): the Neo ISP's only.
        os.environ.setdefault("LIBCAMERA_PIPELINES_MATCH_LIST", "nxp/neo")
        # "Image0 frame loss!" several times a second when the demo takes
        # fewer frames than the sensor delivers: warnings only, not errors
        os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:ERROR")
        # libcamerasrc only negotiates the sensor's full size (2592x1944 for
        # the AR0521). The i.MX8MP's ISP mode 1 is a 1:1 1920x1080 window from
        # the sensor's centre -- the picture scale the model was trained on
        # -- so the same window is cut out here, not scaled.
        size = re.search(r"width=(\d+),height=(\d+)", caps)
        sw, sh = (int(v) for v in os.environ.get("PHYTEC_SENSOR_SIZE", "2592x1944").split("x"))
        if size and (int(size.group(1)), int(size.group(2))) != (sw, sh):
            w, h = int(size.group(1)), int(size.group(2))
            fmt = re.search(r"format=(\w+)", caps)
            full = f"video/x-raw,width={sw},height={sh}" + (f",format={fmt.group(1)}" if fmt else "")
            left, top = (sw - w) // 2 // 2 * 2, (sh - h) // 2 // 2 * 2
            return (f'libcamerasrc camera-name="{name}" ! {full} ! videocrop left={left} top={top} '
                    f'right={sw - w - left} bottom={sh - h - top} ! {caps}')
        return f'libcamerasrc camera-name="{name}" ! {caps}'
    return f"v4l2src device=/dev/video-isp-csi{csi} ! {caps}"


class WhiteBalance:
    """Gray-world white balance + brightness, as per-channel gains.

    Gains are measured on a small copy every `every` frames and smoothed;
    applying them is one cv2.LUT with a per-channel table, which the A53s
    manage at camera rate (a float cv2.multiply of the 1280x800 frame
    dropped the camera loop from 59 to 11 fps).
    """

    def __init__(self, every=15, target=115.0, smooth=0.3, max_gain=4.0):
        self.every, self.target, self.smooth, self.max_gain = every, target, smooth, max_gain
        self.gains = None
        self._lut = None
        self._n = 0

    def measure(self, frame):
        small = cv2.resize(frame[..., :3], (64, 40), interpolation=cv2.INTER_AREA)
        means = small.reshape(-1, 3).mean(0) + 1e-3          # B, G, R
        gains = means.mean() / means                           # gray world
        gains *= self.target / max(means.mean(), 1.0)          # brightness
        return np.clip(gains, 0.5, self.max_gain)

    def _table(self, channels):
        ramp = np.arange(256, dtype=np.float32)[:, None]
        gains = np.append(self.gains, 1.0)[:channels]          # alpha untouched
        return np.clip(ramp * gains, 0, 255).astype(np.uint8).reshape(1, 256, channels)

    def __call__(self, frame):
        if self._n % self.every == 0:
            g = self.measure(frame)
            self.gains = g if self.gains is None else \
                (1 - self.smooth) * self.gains + self.smooth * g
            self._lut = self._table(frame.shape[2])
        self._n += 1
        return cv2.LUT(frame, self._lut)


class AutoLevel:
    """Brightness for a monochrome sensor behind the i.MX95's Neo ISP.

    The ISP's auto exposure does not act on the b/w AR0521 (exposure value,
    analogue and digital gain are all ignored), and its picture comes out
    about half as bright as the i.MX8MP's -- the table near 125 instead of
    ~200. The model was trained on the i.MX8MP's pictures, so this gain
    brings the bright end (the 99th percentile: the table) to `target`,
    measured every `every` frames on every 4th pixel, smoothed; applied as
    one cv2.LUT (~1 ms at 1920x1080).
    """

    def __init__(self, every=15, target=205.0, smooth=0.3, max_gain=4.0):
        self.every, self.target, self.smooth, self.max_gain = every, target, smooth, max_gain
        self.gain, self._lut, self._n = None, None, 0

    def __call__(self, gray):
        if self._n % self.every == 0:
            top = float(np.percentile(gray[::4, ::4], 99)) + 1e-3
            g = float(np.clip(self.target / top, 1.0, self.max_gain))
            self.gain = g if self.gain is None else (1 - self.smooth) * self.gain + self.smooth * g
            self._lut = np.clip(np.arange(256, dtype=np.float32) * self.gain, 0, 255).astype(np.uint8)
        self._n += 1
        return cv2.LUT(gray, self._lut)


def white_balance(bgr):
    """One-shot WhiteBalance for a single (training) picture."""
    return WhiteBalance(every=1, smooth=1.0)(bgr)


class AutoExposure:
    """Software AE for sensors without one: steer `exposure` (lines) so the
    middle of the picture averages `target`.

    Measured on the central half only -- the fisheye's black border and the
    table's edges should not pull the exposure around. The step is damped
    (power 0.6) and v4l2-ctl runs detached, so the camera loop never waits.
    """

    def __init__(self, subdev, start=600, target=110.0, every=8, lo=4, hi=4000):
        self.subdev, self.target, self.every, self.lo, self.hi = subdev, target, every, lo, hi
        self.exposure = float(start)
        self._n = 0
        self._set(start)

    def _set(self, value):
        subprocess.Popen(["v4l2-ctl", "-d", self.subdev, "-c", f"exposure={int(value)}"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def __call__(self, gray):
        self._n += 1
        if self._n % self.every:
            return
        h, w = gray.shape[:2]
        centre = gray[h // 4:3 * h // 4:8, w // 4:3 * w // 4:8]
        mean = float(centre.mean()) + 1.0
        new = float(np.clip(self.exposure * (self.target / mean) ** 0.6, self.lo, self.hi))
        if abs(new - self.exposure) / self.exposure > 0.04:
            self.exposure = new
            self._set(new)


class Camera:
    """A phyCAM on CSI1/CSI2 ("csi1", "csi2"), or a video file for testing.

    read() returns the newest frame in its native layout: 2-D GRAY8 for the
    monochrome AR0521, BGRx (4 channels) for the colour AR0144 and files.
    """

    def __init__(self, source="csi1", white_balance=True, auto_exposure=True, size=None,
                 aspect=None, crop_y=0.5, crop=None, max_width=1920, isp=False, isp_width=None,
                 isp_fps=None):
        """aspect: e.g. 16/9 -- read only a full-width band of that aspect
        from the sensor (fills a 16:9 screen, fewer pixels to detect on);
        crop_y places the band: 0 top, 0.5 centre, 1 bottom.
        crop: (x, y, w, h) sensor window instead, e.g. the table area of a
        wide-angle live camera (phytec-calibrate computes it); read at full
        resolution up to max_width, 2x2-binned above.
        isp: read through the i.MX8MP ISP (imx8-phycam-isp service) in the
        size of its current mode instead of raw through the ISI -- the ISP does exposure, denoise and sharpening
        itself; aspect/crop are not applied then.
        isp_width: let the ISP's scaler shrink the mono picture to this width
        (height to a multiple of 8, e.g. 960x544): it comes as NV12 -- the
        GRAY8 output crops instead of scaling -- and its Y plane is the
        picture, without any work on the CPU."""
        self.crop = None  # sensor window actually used, (x, y, w, h)
        self.scale = 1.0  # output pixels per sensor pixel
        self.is_camera = source in ("csi1", "csi2")
        self.mono = False
        self.sensor = None
        self._wb = self._ae = None
        self._level = None
        self._yuy2 = False
        self._nv12 = None  # (w, h) when the ISP scales the mono picture
        if not self.is_camera:
            self._cap = cv2.VideoCapture(source)
        else:
            csi = int(source[-1])
            found = find_sensor(csi)
            if not found:
                raise RuntimeError(f"no camera on CSI{csi}")
            subdev, name, port = found
            mono = is_mono(subdev)
            kind = next((k for k in SENSORS if name.startswith(k[0]) and k[1] == mono), None)
            if kind is None:
                raise RuntimeError(f"unsupported sensor {name!r} on CSI{csi}")
            cfg = SENSORS[kind]
            self.sensor, self.mono = name, mono
            sensor_w, sensor_h = cfg["size"]
            if isp:
                # the ISP's mode decides the size: 2592x1944 is the whole
                # sensor, 1920x1080 a 1:1 window from its centre. On the
                # i.MX95 libcamera picks the sensor window itself; whether it
                # is that same centre window (and so the same framing the
                # model was trained on) is still to be checked on the board.
                sensor_w, sensor_h = isp_size(csi) or (sensor_w, sensor_h)
                self.crop = ((cfg["size"][0] - sensor_w) // 2, (cfg["size"][1] - sensor_h) // 2,
                             sensor_w, sensor_h)
                # The ISP outputs YUV or grey only. Colour comes as YUY2 and is
                # converted here (~15 ms at 1920x1080): converting on the G2D
                # while the screen uses it too gave ghost images in the demo.
                fmt = "GRAY8" if mono else "YUY2"
                out_w, out_h = sensor_w, sensor_h
                if mono and isp_width and isp_width < sensor_w:
                    fmt = "NV12"
                    out_w = isp_width & ~15
                    out_h = int(round(sensor_h * out_w / sensor_w / 8)) * 8
                    self._nv12 = (out_w, out_h)
                    # x and y scale differ a little (544 / 1080 vs 960 / 1920)
                    self.scale = np.array([out_w / sensor_w, out_h / sensor_h])
                self._cap = cv2.VideoCapture(
                    isp_source(csi, f"video/x-raw,width={out_w},height={out_h},format={fmt}"
                               + (f",framerate={isp_fps}/1" if isp_fps else ""))
                    + " ! appsink drop=true max-buffers=1", cv2.CAP_GSTREAMER)
                if self._nv12:
                    self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                if not mono:
                    self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                    self._yuy2 = True
                if not self._cap.isOpened():
                    hint = ("libcamera, neoisp overlay loaded?" if soc() == "imx95"
                            else "imx8-phycam-isp running, mode 2?")
                    raise RuntimeError(f"could not open the ISP on CSI{csi} ({hint})")
                self.is_camera = False  # frames come out finished
                if mono and soc() == "imx95":
                    self._level = AutoLevel()
                return
            crop_w, crop_h = sensor_w, sensor_h
            offset = None
            if aspect and name.startswith("ar0521"):
                crop_h = min(sensor_h, int(round(sensor_w / aspect / 2)) * 2)
                # +4: the AR0521's active area starts at (4, 4), as in
                # setup-pipeline-csi1's own default offset
                offset = (4, 4 + (int((sensor_h - crop_h) * min(max(crop_y, 0.0), 1.0)) & ~1))
            if crop is not None and name.startswith("ar0521"):
                cx, cy, cw, ch = (int(v) & ~1 for v in crop)
                cx, cy = min(max(cx, 0), sensor_w - 64), min(max(cy, 0), sensor_h - 64)
                crop_w, crop_h = min(cw, sensor_w - cx), min(ch, sensor_h - cy)
                offset = (4 + cx, 4 + cy)
                width, height = crop_w, crop_h
                if width > max_width:  # 2x2 binning
                    width, height = (crop_w // 2) & ~1, (crop_h // 2) & ~1
            elif size:
                width, height = size
            elif "out" in cfg:  # binned output, same aspect as the crop
                width = cfg["out"][0]
                height = int(round(crop_h * width / crop_w / 2)) * 2
            else:
                width, height = crop_w, crop_h
            self.crop = (offset[0] - 4 if offset else 0, offset[1] - 4 if offset else 0,
                         crop_w, crop_h)
            self.scale = width / crop_w
            cmd = [f"setup-pipeline-csi{csi}", "-s", f"{width}x{height}",
                   "-c", f"{crop_w}x{crop_h}"]
            if offset:
                cmd += ["-o", f"({offset[0]},{offset[1]})"]
            if cfg["mbus"]:
                cmd += ["-f", cfg["mbus"]]
            if port is not None:
                cmd += ["-p", str(port)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            if cfg.get("hblank"):
                subprocess.run(["v4l2-ctl", "-d", subdev, "-c",
                                f"horizontal_blanking={cfg['hblank']}"], check=False)
            if name.startswith("ar0521") and auto_exposure:  # no AE in the driver
                self._ae = AutoExposure(subdev)
            if not self.mono and white_balance:
                self._wb = WhiteBalance()
            pipeline = (f"v4l2src device=/dev/video-isi-csi{csi} ! "
                        f"{cfg['caps']},width={width},height={height} ! "
                        f"appsink drop=true max-buffers=1")
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open {source!r}")

    def read(self):
        ok, raw = self._cap.read()
        if not ok:
            return False, None
        if self._nv12:  # Y plane of the ISP-scaled NV12 picture
            w, h = self._nv12
            raw = raw.reshape(-1)[:w * h].reshape(h, w)
            return True, raw
        if self.mono:
            if raw.ndim == 3:  # some OpenCV builds hand GRAY8 out as 3 channels
                raw = raw[..., 0]
            if self._ae:
                self._ae(raw)
            if self._level:
                raw = self._level(raw)
            return True, raw
        if self.is_camera:
            if raw.ndim == 3:
                raw = raw[..., 0]
            if self._ae:
                self._ae(raw)
            # GRBG on the wire is what OpenCV calls GB (its names are offset
            # by one pixel) -- same conversion as object_detect.
            frame = cv2.cvtColor(raw, cv2.COLOR_BAYER_GB2BGRA)
            return True, self._wb(frame) if self._wb else frame
        if self._yuy2:  # colour from the ISP
            return True, cv2.cvtColor(raw, cv2.COLOR_YUV2BGRA_YUY2)
        return True, cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)

    def release(self):
        self._cap.release()


def to_bgrx(frame):
    """Any read() frame -> BGRx for the screen."""
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGRA) if frame.ndim == 2 else frame


@functools.lru_cache(maxsize=1)
def screen_size():
    """Current mode of the connected output, from Weston's log (read once:
    the live view asks for it every frame)."""
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
        screen_w, screen_h = screen_size()
        # Fit inside the screen keeping the aspect ratio; Weston centres it.
        scale = min(screen_w / width, screen_h / height)
        out_w, out_h = int(width * scale) & ~1, int(height * scale) & ~1
        self._pipeline = Gst.parse_launch(
            f"appsrc name=src is-live=true do-timestamp=true format=time "
            f"max-buffers=2 leaky-type=downstream "
            f"caps=video/x-raw,format=BGRx,width={width},height={height},"
            f"framerate=0/1 ! queue max-size-buffers=2 leaky=downstream ! "
            f"imxvideoconvert_g2d ! video/x-raw,width={out_w},height={out_h} ! "
            f"waylandsink fullscreen=true sync=false")
        self._src = self._pipeline.get_by_name("src")
        self._pipeline.set_state(Gst.State.PLAYING)

    def show(self, bgrx):
        self._src.emit("push-buffer", self._Gst.Buffer.new_wrapped(bgrx.tobytes()))

    def close(self):
        self._src.emit("end-of-stream")
        self._pipeline.set_state(self._Gst.State.NULL)


class OverlayScreen:
    """Live colour picture straight from the ISP to the screen -- camera,
    ISP, G2D compositor, Wayland, at the camera's 30 fps without passing
    through Python -- with a transparent BGRA layer on top for frames,
    labels and panels. show(layer) replaces the layer whenever it changes;
    the compositor keeps the last one in between."""

    def __init__(self, csi, size, fps=30):
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        self._Gst = Gst
        w, h = size
        sw, sh = screen_size()
        leaky = "queue max-size-buffers=2 leaky=downstream"
        self._pipeline = Gst.parse_launch(
            f"imxcompositor_g2d name=c sink_0::width={sw} sink_0::height={sh} "
            f"sink_1::width={sw} sink_1::height={sh} sink_1::zorder=1 ! "
            f"waylandsink fullscreen=true sync=false "
            + isp_source(csi, f"video/x-raw,format=YUY2,width={w},height={h},framerate={fps}/1")
            + f" ! {leaky} ! c.sink_0 "
            f"appsrc name=layer is-live=true do-timestamp=true format=time "
            f"caps=video/x-raw,format=BGRA,width={sw},height={sh},framerate={fps}/1 ! "
            # The G2D only blends DMA buffers: a wrapped Python buffer is
            # dropped without a word. videoconvert writes the layer into a
            # buffer from the compositor's own pool (the BGRA -> RGBA swap is
            # just the price of that copy).
            f"videoconvert ! video/x-raw,format=RGBA ! {leaky} ! c.sink_1")
        self._src = self._pipeline.get_by_name("layer")
        self.size = (sw, sh)
        self.frames, self.error = 0, None  # composed frames; first pipeline error
        self._last = None

        self.grab_to = None  # path: save the next composed frame there (snapshot)

        def count(pad, info):
            self.frames += 1
            if self.grab_to:
                try:
                    self._grab(pad, info)
                except Exception as error:  # noqa: BLE001 -- a snapshot is not worth a crash
                    print(f"overlay screen: snapshot failed: {error!r}", flush=True)
                self.grab_to = None
            return Gst.PadProbeReturn.OK

        self._pipeline.get_by_name("c").get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, count)
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.enable_sync_message_emission()

        def on_message(_bus, msg):
            if msg.type in (Gst.MessageType.ERROR, Gst.MessageType.WARNING):
                err, dbg = msg.parse_error() if msg.type == Gst.MessageType.ERROR \
                    else msg.parse_warning()
                text = f"{msg.src.get_name()}: {err.message} ({dbg})"
                print(f"overlay screen: {text}", flush=True)
                if msg.type == Gst.MessageType.ERROR and self.error is None:
                    self.error = text
        bus.connect("sync-message", on_message)
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self.error = "pipeline does not start"

    def _grab(self, pad, info):
        Gst = self._Gst
        caps = pad.get_current_caps().get_structure(0)
        cw, ch = caps.get_value("width"), caps.get_value("height")
        rgb = caps.get_value("format").startswith("RGB")
        buf = info.get_buffer()
        ok, m = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("cannot map the composed frame")
        try:
            a = np.frombuffer(m.data, np.uint8)
            stride = cw * 4 if a.size % (cw * 4) == 0 else a.size // ch
            img = a[:stride * ch].reshape(ch, stride)[:, :cw * 4].reshape(ch, cw, 4)
            cv2.imwrite(self.grab_to, img[..., 2::-1] if rgb else img[..., :3])
        finally:
            buf.unmap(m)

    def show(self, bgra):
        """Replace the layer. The caller draws only when something changed
        (comparing 8 MB with the last layer cost ~23 ms on the A53, keeping
        a copy another ~9 ms -- more than the drawing itself)."""
        self._src.emit("push-buffer", self._Gst.Buffer.new_wrapped(bgra.tobytes()))

    def close(self):
        self._src.emit("end-of-stream")
        self._pipeline.set_state(self._Gst.State.NULL)


def label_frame(frame, text, color=(0, 255, 255)):
    """Solid bar with large text at the top, readable from a few metres."""
    font, scale, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3, 12
    (_, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    bar_height = text_h + baseline + 2 * margin
    alpha = (255,) if frame.ndim == 3 and frame.shape[2] == 4 else ()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_height), (0, 0, 0) + alpha, -1)
    cv2.putText(frame, text, (margin, margin + text_h), font, scale,
                color + ((255,) if frame.ndim == 3 and frame.shape[2] == 4 else ()),
                thickness, cv2.LINE_AA)
    return frame


def die(message, code=2):
    print(message, file=sys.stderr)
    sys.exit(code)
