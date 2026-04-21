import json
import traceback
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

from village.scripts.log import log
from village.settings import settings


class CameraCalibration:
    def __init__(self, image_dir: Path, spacing_mm: float,
                 dot_radius_mm: float = 1.5) -> None:
        self.image_dir = image_dir
        self.spacing_mm = spacing_mm
        self.dot_radius_mm = dot_radius_mm

        self.running = False
        self.error = False
        self.result: dict | None = None

    def run_in_thread(self) -> None:
        self.running = True
        self.error = False
        self.result = None
        t = Thread(target=self._run, daemon=True)
        t.start()

    def _run(self) -> None:
        try:
            self.result = _calibrate(self.image_dir, self.spacing_mm)
        except Exception:
            log.error("Camera calibration error",
                      exception=traceback.format_exc())
            self.error = True
        finally:
            self.running = False

    def save(self, out_path: Path) -> None:
        if self.result is None:
            return
        with open(out_path, "w") as f:
            json.dump(self.result, f, indent=2)


def _make_blob_detector() -> cv2.SimpleBlobDetector:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 25
    params.filterByCircularity = True
    params.minCircularity = 0.85
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minThreshold = 50
    params.maxThreshold = 150
    params.thresholdStep = 10
    return cv2.SimpleBlobDetector_create(params)


def _blobs_to_points(keypoints, spacing_mm: float):
    """Order blob keypoints into a grid and return (obj_pts, img_pts).

    Clusters blobs by Y into rows, sorts each row by X, then assigns
    world coordinates (col * spacing, row * spacing, 0).
    Returns None if blobs are too few or don't form a clean grid.
    """
    if len(keypoints) < 4:
        return None, None

    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    pts = pts[np.argsort(pts[:, 1])]  # sort by Y

    y_vals = pts[:, 1]
    gaps = np.diff(y_vals)
    if len(gaps) == 0:
        return None, None
    threshold = np.median(gaps) * 5
    row_breaks = np.where(gaps > threshold)[0] + 1
    rows = np.split(pts, row_breaks)

    # Reject if row lengths are too inconsistent (spurious blobs)
    row_lens = [len(r) for r in rows]
    expected_cols = int(np.median(row_lens))
    if expected_cols < 2 or any(abs(l - expected_cols) > 1 for l in row_lens):
        return None, None

    img_pts, obj_pts = [], []
    for row_idx, row in enumerate(rows):
        row = row[np.argsort(row[:, 0])]  # sort by X within row
        if len(row) != expected_cols:
            continue
        for col_idx, pt in enumerate(row):
            img_pts.append(pt)
            obj_pts.append([col_idx * spacing_mm, row_idx * spacing_mm, 0.0])

    if len(img_pts) < 4:
        return None, None

    return (np.array(obj_pts, dtype=np.float32),
            np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2))


def _calibrate(image_dir: Path, spacing_mm: float) -> dict:
    detector = _make_blob_detector()

    obj_points: list = []
    img_points: list = []
    image_size = None
    failed: list[str] = []

    paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            failed.append(path.name)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        keypoints = detector.detect(gray)
        obj_pts, img_pts = _blobs_to_points(keypoints, spacing_mm)
        if obj_pts is None:
            failed.append(path.name)
            continue

        obj_points.append(obj_pts)
        img_points.append(img_pts)

    n_used = len(obj_points)
    if n_used < 4:
        raise RuntimeError(f"Only {n_used} image(s) yielded a usable grid "
                           f"(need ≥ 4).")

    res = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    _, camera_matrix, dist_coeffs, rvecs, tvecs = res

    total_err = 0.0
    for i, (op, ip) in enumerate(zip(obj_points, img_points)):
        proj, _ = cv2.projectPoints(
            op, rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        total_err += cv2.norm(ip, proj, cv2.NORM_L2) / len(proj)
    mean_err = total_err / n_used

    return {"camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.flatten().tolist(),
            "reprojection_error_px": float(mean_err),
            "image_size_wh": list(image_size),
            "n_images_used": n_used,
            "n_images_failed": len(failed),
            "failed_images": failed,
            "spacing_mm": spacing_mm}


def default_capture_dir() -> Path:
    return Path(settings.get("DATA_DIRECTORY")) / "camera_calib_captures"


def default_result_path() -> Path:
    return Path(settings.get("DATA_DIRECTORY")) / "camera_calibration.json"
