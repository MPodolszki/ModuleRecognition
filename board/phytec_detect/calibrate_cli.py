"""Calibrate detection camera -> live camera on the table plane.

Lay the printed ArUco target (tools/make_aruco_board.py, docs/*.pdf) flat
on the table, in view of both cameras, then:

    phytec-calibrate                 # add this placement's points and refit
    phytec-calibrate --reset         # start over

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
    parser.add_argument("--model", choices=("auto", "homography", "poly3"), default="auto",
                        help="mapping model (auto: homography unless the cubic fits clearly better)")
    args = parser.parse_args(argv)

    os.makedirs(CONF_DIR, exist_ok=True)
    # Both cameras at their full field of view: the target is easier to find,
    # and the mapping is stored in sensor pixels, valid for any crop later.
    det_cam, view_cam = Camera(args.detect_cam), Camera(args.view_cam)
    for _ in range(3):  # several tries: AE needs a few frames on a white sheet
        det = grab(det_cam, 20)
        view = grab(view_cam)
        m_det = markers(det if det.ndim == 2 else cv2.cvtColor(det, cv2.COLOR_BGRA2GRAY))
        m_view = markers(cv2.cvtColor(view, cv2.COLOR_BGRA2GRAY))
        common = sorted(set(m_det) & set(m_view))
        if common:
            break
    det_cam.release()
    view_cam.release()
    print(f"markers: detection cam {len(m_det)}, live cam {len(m_view)}, both {len(common)}")
    if not common:
        die("no marker seen by both cameras -- is the target in both views and in focus?")

    src = frame_to_sensor(np.concatenate([m_det[i] for i in common]), det_cam)
    dst = frame_to_sensor(np.concatenate([m_view[i] for i in common]), view_cam)
    if os.path.exists(POINTS_PATH) and not args.reset:
        old = np.load(POINTS_PATH)
        src, dst = np.concatenate([old["src"], src]), np.concatenate([old["dst"], dst])
    np.savez(POINTS_PATH, src=src, dst=dst)

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
    plane.save(MAP_PATH)
    print(f"live camera window (sensor px): {plane.view_crop}")
    print(f"{plane.kind}: {len(src)} point pairs ({dropped} rejected), rms {plane.rms:.1f} px "
          f"in the live picture -> {MAP_PATH}")

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
