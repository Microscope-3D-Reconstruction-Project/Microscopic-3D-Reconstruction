"""
microscope_scripts/check_undistort.py

Loads intrinsics from camera_calib_intrinsic.json inside a calibration folder,
undistorts a user-specified image from that folder's images/ subfolder, and
shows the original and undistorted frames side by side in an OpenCV window.

Usage:
    python microscope_scripts/check_undistort.py \\
        --calib_dir microscope-data/calibrations/20240501_123456 \\
        --image frame_00003.png
"""

import argparse
import json
import sys

from pathlib import Path

import cv2
import numpy as np


def main(calib_dir: Path, image_name: str) -> None:
    calib_dir = calib_dir.resolve()
    print(f"calib_dir  : {calib_dir}")
    print(f"  exists   : {calib_dir.exists()}")

    json_path = calib_dir / "camera_calib_intrinsic.json"
    print(f"intrinsics : {json_path}")
    print(f"  exists   : {json_path.exists()}")
    if not json_path.exists():
        sys.exit("ERROR: intrinsics file not found.")

    with open(json_path) as f:
        calib = json.load(f)

    K = np.array(calib["camera_matrix"], dtype=np.float64)
    dist = np.array(calib["distortion_coefficients"], dtype=np.float64)
    print(f"K:\n{K}")
    print(f"dist: {dist.flatten()}")

    img_path = calib_dir / image_name
    print(f"image      : {img_path}")
    print(f"  exists   : {img_path.exists()}")
    if not img_path.exists():
        sys.exit("ERROR: image file not found.")

    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit("ERROR: cv2.imread returned None — file may not be a valid image.")

    h, w = img.shape[:2]
    print(f"image size : {w}x{h}")

    # alpha=0 crops to only valid (non-black) pixels; safer than alpha=1 for display
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    print(f"ROI        : {roi}")

    undistorted = cv2.undistort(img, K, dist, None, new_K)

    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted_show = undistorted[y : y + rh, x : x + rw]
        original_show = img[y : y + rh, x : x + rw]
    else:
        print("WARNING: ROI is degenerate — showing full frames without crop.")
        undistorted_show = undistorted
        original_show = img

    print(f"display size: {original_show.shape[1]}x{original_show.shape[0]}")

    def label(frame, text):
        out = frame.copy()
        cv2.putText(
            out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2
        )
        return out

    left = label(original_show, "Original")
    right = label(undistorted_show, "Undistorted")

    combined = np.hstack([left, right])
    scale = min(1.0, 1800 / combined.shape[1])
    if scale < 1.0:
        combined = cv2.resize(
            combined, (int(combined.shape[1] * scale), int(combined.shape[0] * scale))
        )

    print(f"window size: {combined.shape[1]}x{combined.shape[0]}")
    print("Press any key to close.")

    # cv2.imshow(f"Undistort check — {image_name}", combined)
    cv2.imshow("Original", original_show)
    cv2.imshow("Undistorted", undistorted_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show original vs undistorted image side by side.",
    )
    parser.add_argument(
        "--calib_dir",
        type=Path,
        required=True,
        help="Path to calibration folder containing camera_calib_intrinsic.json and images/",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Filename of the image inside images/ (e.g. frame_00003.png)",
    )
    args = parser.parse_args()
    main(args.calib_dir, args.image)
