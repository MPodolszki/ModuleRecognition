"""Calibrate detection camera -> live camera on the table plane.

Lay the printed ArUco target (tools/make_aruco_board.py, docs/*.pdf) flat
on the table, in view of both cameras, then:

    phytec-calibrate                 # add this placement's points and refit
    phytec-calibrate --reset         # start over
    phytec-calibrate --isp --features
                                     # cameras through the ISP (phytec-detect
                                     # --isp), no target: matches what both
                                     # cameras see on the table (SIFT)

With normal lenses one placement is enough (the homography holds on the
whole table); moving the sheet and running it again adds points and
refits, which helps if the detection lens has visible distortion. Every
run keeps the point pairs of the earlier ones and writes

    /etc/phytec-detect/plane_map.json      (used by phytec-detect)
    /tmp/phytec-calibrate.jpg              (check picture: live view with
                                            mapped marker corners in red)
"""

import argparse
import os
import sys

import cv2
import numpy as np

from .calib import PlaneMap, frame_to_sensor, view_crop_for
from .camera import Camera, die

CONF_DIR = "/etc/phytec-detect"
MAP_PATH = os.path.join(CONF_DIR, "plane_map.json")
POINTS_PATH = os.path.join(CONF_DIR, "plane_points.npz")
# The ISP dewarps and crops differently from the raw path: its own mapping.
ISP_MAP_PATH = os.path.join(CONF_DIR, "plane_map_isp.json")
ISP_POINTS_PATH = os.path.join(CONF_DIR, "plane_points_isp.npz")


def isp_map_path(isp_width=0):
    """The ISP mapping for a detection width (0 = the ISP's full size)."""
    return ISP_MAP_PATH.replace(".json", f"_w{isp_width}.json") if isp_width else ISP_MAP_PATH


def grab(cam, n=12):
    """Let auto exposure / white balance settle, return the last frame."""
    frame = None
    for _ in range(n):
        ok, frame = cam.read()
        if not ok:
            die("could not read from camera")
    return frame


def markers(gray):
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
        cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def feature_pairs(det, view, max_pairs=400):
    """Point pairs from what both cameras see (SIFT, ratio test, homography
    RANSAC): no target needed, but the table needs something on it -- the
    modules, a sheet of print. Returns (det points, view points)."""
    sift = cv2.SIFT_create(nfeatures=4000)
    ka, da = sift.detectAndCompute(det, None)
    kb, db = sift.detectAndCompute(view, None)
    if da is None or db is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    src = np.float64([ka[m.queryIdx].pt for m in good])
    dst = np.float64([kb[m.trainIdx].pt for m in good])
    _, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    keep = inl.ravel().astype(bool) if inl is not None else np.zeros(len(src), bool)
    src, dst = src[keep], dst[keep]
    if len(src) > max_pairs:
        idx = np.linspace(0, len(src) - 1, max_pairs).astype(int)
        src, dst = src[idx], dst[idx]
    return src, dst


def _aspect(text):
    """'16:9' -> 1.777..., None -> None."""
    if not text:
        return None
    w, _, h = text.partition(":")
    return float(w) / float(h or 1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phytec-calibrate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detect-cam", default="csi1")
    parser.add_argument("--aspect", default=None,
                        help="read a full-width band of this aspect from the detection "
                             "sensor, e.g. 16:9 to fill the screen (default: whole sensor)")
    parser.add_argument("--crop-y", type=float, default=0.5,
                        help="where that band sits: 0 top, 0.5 centre, 1 bottom "
                             "(use the same values as phytec-detect)")
    parser.add_argument("--view-cam", default="csi2")
    parser.add_argument("--reset", action="store_true", help="forget earlier placements")
    parser.add_argument("--isp", action="store_true",
                        help="both cameras through the ISP (for phytec-detect --isp)")
    parser.add_argument("--isp-width", type=int, default=0,
                        help="with --isp: detection picture scaled by the ISP to this width, "
                             "as phytec-detect --isp-width (0 = full size); its own map file, "
                             "the ISP's scaled output frames a slightly different window")
    parser.add_argument("--features", action="store_true",
                        help="no ArUco target: match what both cameras see on the table")
    parser.add_argument("--model", choices=("auto", "homography", "poly3"), default="auto",
                        help="mapping model (auto: homography unless the cubic fits clearly better)")
    args = parser.parse_args(argv)

    os.makedirs(CONF_DIR, exist_ok=True)
    # Both cameras at their full field of view: the target is easier to find,
    # and the mapping is stored in sensor pixels, valid for any crop later.
    width = args.isp_width if args.isp else 0
    det_cam = Camera(args.detect_cam, isp=args.isp, isp_width=width or None)
    view_cam = Camera(args.view_cam, isp=args.isp)
    map_path, points_path = (isp_map_path(width), ISP_POINTS_PATH.replace(
        ".npz", f"_w{width}.npz" if width else ".npz")) if args.isp else (MAP_PATH, POINTS_PATH)

    def gray(img):
        return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    for _ in range(3):  # several tries: AE needs a few frames on a white sheet
        det = grab(det_cam, 20)
        view = grab(view_cam, 20)
        if args.features:
            p_det, p_view = feature_pairs(gray(det), gray(view))
            print(f"features: {len(p_det)} matching points")
            if len(p_det) >= 8:
                break
        else:
            m_det, m_view = markers(gray(det)), markers(gray(view))
            common = sorted(set(m_det) & set(m_view))
            print(f"markers: detection cam {len(m_det)}, live cam {len(m_view)}, "
                  f"both {len(common)}")
            if common:
                p_det = np.concatenate([m_det[i] for i in common])
                p_view = np.concatenate([m_view[i] for i in common])
                break
            p_det = []
    det_cam.release()
    view_cam.release()
    if len(p_det) < 8:
        die("too few points seen by both cameras -- "
            + ("put modules or printed paper on the table" if args.features
               else "is the target in both views and in focus?"))

    src = frame_to_sensor(p_det, det_cam)
    dst = frame_to_sensor(p_view, view_cam)
    if os.path.exists(points_path) and not args.reset and not args.features:
        old = np.load(points_path)
        src, dst = np.concatenate([old["src"], src]), np.concatenate([old["dst"], dst])
    np.savez(points_path, src=src, dst=dst)

    try:
        plane, dropped = PlaneMap.fit(src, dst, (2604, 2604), (2604, 2604), args.model)
    except ValueError as error:
        die(f"{error} -- move the target and run phytec-calibrate again")
    # The live camera's window = what the detection camera sees in the demo
    # (its --aspect/--crop-y band), widened to 16:9.
    asp = _aspect(args.aspect)
    det_h = min(1944, int(round(2592 / asp / 2)) * 2) if asp else 1944
    det_y = int((1944 - det_h) * min(max(args.crop_y, 0.0), 1.0)) & ~1
    plane.view_crop = tuple(view_crop_for(plane, (0, det_y, 2592, det_h), (2592, 1944)))
    plane.save(map_path)
    print(f"live camera window (sensor px): {plane.view_crop}")
    print(f"{plane.kind}: {len(src)} point pairs ({dropped} rejected), rms {plane.rms:.1f} px "
          f"in the live picture -> {map_path}")

    from .calib import sensor_to_frame
    check = view.copy()
    for p in sensor_to_frame(plane(src), view_cam).astype(int):
        cv2.circle(check, tuple(int(v) for v in p), 4, (0, 0, 255, 255), -1)
    hull = sensor_to_frame(plane(plane.hull), view_cam).astype(np.int32)
    cv2.polylines(check, [hull], True, (0, 255, 0, 255), 2)
    vx, vy, vw, vh = plane.view_crop
    a = sensor_to_frame([[vx, vy], [vx + vw, vy + vh]], view_cam).astype(int)
    cv2.rectangle(check, tuple(a[0]), tuple(a[1]), (255, 255, 0, 255), 3)
    cv2.imwrite("/tmp/phytec-calibrate.jpg", check[..., :3])
    print("check picture: /tmp/phytec-calibrate.jpg (red: mapped marker corners, "
          "green: area the markers covered, cyan: live camera window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
