import sys
import cv2
import numpy as np

from village.calibration.camera_calibration import (
    _make_blob_detector, _blobs_to_points,
)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    img_path = sys.argv[1]
    spacing_mm = float(sys.argv[2])

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}")
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Image size: {gray.shape[1]}x{gray.shape[0]}")
    print(f"Pixel range: min={gray.min()}  max={gray.max()}  mean={gray.mean():.1f}")

    detector = _make_blob_detector()
    keypoints = detector.detect(gray)
    sizes = sorted(kp.size for kp in keypoints)
    print(f"\nBlobs detected: {len(keypoints)}")
    if sizes:
        print(f"Blob sizes: min={sizes[0]:.1f}  median={sizes[len(sizes)//2]:.1f}"
              f"  max={sizes[-1]:.1f}")

    obj_pts, img_pts = _blobs_to_points(keypoints, spacing_mm)
    if obj_pts is None:
        print("Grid ordering FAILED — inconsistent blob layout.")
    else:
        rows = int(obj_pts[:, 1].max() / spacing_mm) + 1
        cols = int(obj_pts[:, 0].max() / spacing_mm) + 1
        print(f"Grid ordered OK: {cols} cols x {rows} rows ({len(obj_pts)} points)")

    vis = cv2.drawKeypoints(img, keypoints, np.array([]),
                            (0, 0, 255),
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if img_pts is not None:
        for pt in img_pts.reshape(-1, 2):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

    out_path = img_path.rsplit(".", 1)[0] + "_debug.jpg"
    cv2.imwrite(out_path, vis)
    print(f"Saved: {out_path}  (red=all blobs, green=grid points)")

if __name__ == "__main__":
    main()
