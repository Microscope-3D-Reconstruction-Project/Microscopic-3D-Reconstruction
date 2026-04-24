"""
robot_scan/find_hemisphere_center.py

Given the rays.npy file saved by localize_scan_object.py, computes the
least-squares closest point to all rays as the predicted hemisphere center.

rays.npy layout: (N, 2, 3)  — axis-1 index 0 = origins, 1 = directions

Usage:
    python robot_scan/find_hemisphere_center.py <path/to/rays.npy>
    python robot_scan/find_hemisphere_center.py  # uses most recent scan session
"""

import argparse
import sys

from pathlib import Path

import numpy as np

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ray_reconstruction import least_squares_ray_intersection


def find_most_recent_rays(scans_root: Path) -> Path | None:
    sessions = sorted(scans_root.glob("*/rays.npy"))
    return sessions[-1] if sessions else None


def main(rays_path: Path) -> None:
    data = np.load(rays_path)  # (N, 2, 3)
    assert (
        data.ndim == 3 and data.shape[1] == 2 and data.shape[2] == 3
    ), f"Expected shape (N, 2, 3), got {data.shape}"
    origins = data[:, 0, :]  # (N, 3)
    directions = data[:, 1, :]  # (N, 3)
    n = len(origins)

    print(f"Loaded {n} rays from {rays_path}")
    for i, (o, d) in enumerate(zip(origins, directions)):
        print(f"  ray {i:02d}  origin={o.round(4)}  dir={d.round(4)}")

    if n < 2:
        print("Need at least 2 rays to compute an intersection.")
        return

    predicted = least_squares_ray_intersection(origins, directions)
    print(
        f"\nPredicted hemisphere center (least-squares):\n"
        f"  x = {predicted[0]:.6f}\n"
        f"  y = {predicted[1]:.6f}\n"
        f"  z = {predicted[2]:.6f}\n"
        f"  → [{predicted[0]:.6f},  {predicted[1]:.6f},  {predicted[2]:.6f}]"
    )

    # Residuals: perpendicular distance from predicted point to each ray
    residuals = []
    for o, d in zip(origins, directions):
        d = d / np.linalg.norm(d)
        diff = predicted - o
        perp = diff - np.dot(diff, d) * d
        residuals.append(np.linalg.norm(perp))
    residuals = np.array(residuals)
    print(
        f"\nPer-ray residuals (perp. distance to predicted point):\n"
        + "\n".join(f"  ray {i:02d}: {r*1000:.2f} mm" for i, r in enumerate(residuals))
        + f"\n  mean: {residuals.mean()*1000:.2f} mm  |  max: {residuals.max()*1000:.2f} mm"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rays_path",
        nargs="?",
        help="Path to rays.npy. Defaults to most recent scan session.",
    )
    args = parser.parse_args()

    if args.rays_path:
        path = Path(args.rays_path)
    else:
        scans_root = Path(__file__).parent.parent / "microscope-data" / "scans"
        path = find_most_recent_rays(scans_root)
        if path is None:
            print(f"No rays.npy found under {scans_root}")
            sys.exit(1)
        print(f"Using most recent: {path}")

    main(path)
