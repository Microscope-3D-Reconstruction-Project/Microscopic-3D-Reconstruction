#!/usr/bin/env python3
"""
microscope_scripts/plot_poses.py

Loads all .npy 4x4 pose matrices from a directory and plots each one as a
3D coordinate frame (X=red, Y=green, Z=blue) in matplotlib.

Usage:
    python microscope_scripts/plot_poses.py --poses_dir microscope-data/calibrations/20260508_164955/poses
    python microscope_scripts/plot_poses.py --poses_dir microscope-data/calibrations/20260508_164955/poses --axis_length 0.05
    python microscope_scripts/plot_poses.py --poses_dir microscope-data/calibrations/20260508_164955/poses --no_labels
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def draw_frame(ax, T: np.ndarray, length: float, label: str | None = None) -> None:
    """Draw a single 4x4 pose as three RGB arrows in a 3D axes."""
    origin = T[:3, 3]
    x_axis = T[:3, 0]
    y_axis = T[:3, 1]
    z_axis = T[:3, 2]

    for direction, color in zip([x_axis, y_axis, z_axis], ["r", "g", "b"]):
        ax.quiver(
            *origin,
            *(direction * length),
            color=color,
            linewidth=1.5,
            arrow_length_ratio=0.2,
        )

    if label is not None:
        ax.text(*origin, f" {label}", fontsize=6, color="gray")


def main(poses_dir: str, axis_length: float, no_labels: bool) -> None:
    pattern = os.path.join(poses_dir, "*.npy")
    pose_files = sorted(glob.glob(pattern))

    if not pose_files:
        raise FileNotFoundError(f"No .npy files found in: {poses_dir}")

    poses = []
    names = []
    for path in pose_files:
        T = np.load(path)
        if T.shape != (4, 4):
            print(
                f"  [skip] {os.path.basename(path)} — shape {T.shape}, expected (4,4)"
            )
            continue
        poses.append(T)
        names.append(os.path.splitext(os.path.basename(path))[0])

    print(f"Loaded {len(poses)} poses from {poses_dir}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    origins = np.array([T[:3, 3] for T in poses])

    for T, name in zip(poses, names):
        draw_frame(ax, T, axis_length, label=None if no_labels else name)

    # Scatter origin points
    ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], color="k", s=10, zorder=5)

    # Equal-ish aspect ratio
    mins = origins.min(axis=0) - axis_length
    maxs = origins.max(axis=0) + axis_length
    center = (mins + maxs) / 2
    half_range = max((maxs - mins).max() / 2, axis_length * 2)
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    session = os.path.basename(os.path.dirname(os.path.abspath(poses_dir)))
    ax.set_title(f"Camera poses — {session}  ({len(poses)} frames)\nR=X  G=Y  B=Z")

    # Legend proxy
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="r", linewidth=2, label="X"),
        Line2D([0], [0], color="g", linewidth=2, label="Y"),
        Line2D([0], [0], color="b", linewidth=2, label="Z"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot 4x4 pose matrices as 3D coordinate frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python microscope_scripts/plot_poses.py --poses_dir microscope-data/calibrations/20260508_164955/poses
  python microscope_scripts/plot_poses.py --poses_dir microscope-data/calibrations/20260508_164955/poses --axis_length 0.03
        """,
    )
    parser.add_argument(
        "--poses_dir",
        required=True,
        help="Directory containing .npy 4x4 pose matrices.",
    )
    parser.add_argument(
        "--axis_length",
        type=float,
        default=0.02,
        help="Length of each axis arrow in metres (default: 0.02).",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Suppress per-frame filename labels.",
    )

    args = parser.parse_args()
    main(args.poses_dir, args.axis_length, args.no_labels)
