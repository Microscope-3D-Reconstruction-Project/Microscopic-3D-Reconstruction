from __future__ import annotations

import argparse
import csv
import re

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from plotly.colors import qualitative
from scipy.spatial.transform import Rotation as R

OPENGL_TO_OPENCV_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class CameraPose:
    scan_name: str
    pose_path: Path
    world_from_camera: np.ndarray

    @property
    def pose_name(self) -> str:
        return self.pose_path.stem

    @property
    def label(self) -> str:
        return f"{self.scan_name}/{self.pose_name}"

    @property
    def center(self) -> np.ndarray:
        return self.world_from_camera[:3, 3]


def pose_sort_key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return (path.stem[: match.start()], int(match.group(1)), path.stem)
    return (path.stem, -1, path.stem)


def as_homogeneous_transform(matrix: np.ndarray, pose_path: Path) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        transform = np.eye(4)
        transform[:3, :] = matrix
        return transform
    raise ValueError(f"{pose_path} has shape {matrix.shape}; expected 4x4 or 3x4.")


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def convert_to_opencv_camera_frame(
    world_from_camera: np.ndarray, source_camera_convention: str
) -> np.ndarray:
    if source_camera_convention == "opencv":
        return world_from_camera
    if source_camera_convention == "opengl":
        return world_from_camera @ OPENGL_TO_OPENCV_CAMERA
    raise ValueError(f"Unknown camera convention: {source_camera_convention}")


def load_scan_poses(
    dataset_dir: Path,
    scan_prefix: str,
    is_camera_to_world: bool,
    source_camera_convention: str,
) -> dict[str, list[CameraPose]]:
    scans: dict[str, list[CameraPose]] = {}

    for scan_dir in sorted(dataset_dir.glob(f"{scan_prefix}*")):
        if not scan_dir.is_dir():
            continue

        pose_paths = sorted(scan_dir.glob("pose_*.npy"), key=pose_sort_key)
        if not pose_paths:
            singular_pose = scan_dir / "pose.npy"
            if singular_pose.exists():
                pose_paths = [singular_pose]

        scan_poses = []
        for pose_path in pose_paths:
            stored_pose = as_homogeneous_transform(np.load(pose_path), pose_path)
            world_from_camera = (
                stored_pose if is_camera_to_world else invert_transform(stored_pose)
            )
            scan_poses.append(
                CameraPose(
                    scan_name=scan_dir.name,
                    pose_path=pose_path,
                    world_from_camera=convert_to_opencv_camera_frame(
                        world_from_camera, source_camera_convention
                    ),
                )
            )

        if scan_poses:
            scans[scan_dir.name] = scan_poses

    return scans


def flatten_scan_poses(scans: dict[str, list[CameraPose]]) -> list[CameraPose]:
    return [pose for scan_poses in scans.values() for pose in scan_poses]


def choose_axis_length(camera_centers: np.ndarray, axis_length: float | None) -> float:
    if axis_length is not None:
        return axis_length

    scene_span = np.ptp(camera_centers, axis=0)
    diagonal = float(np.linalg.norm(scene_span))
    if diagonal > 0:
        return 0.025 * diagonal
    return 0.01


def append_axis_lines(
    lines: list[list[float | None]], origin: np.ndarray, tip: np.ndarray
) -> None:
    for axis_index in range(3):
        lines[axis_index].extend([origin[axis_index], tip[axis_index], None])


def plot_poses(
    scans: dict[str, list[CameraPose]],
    output_html: Path,
    axis_length: float | None = None,
    show_labels: bool = False,
    figure_size: tuple[int, int] = (1200, 900),
) -> None:
    all_poses = flatten_scan_poses(scans)
    camera_centers = np.array([pose.center for pose in all_poses])
    draw_axis_length = choose_axis_length(camera_centers, axis_length)

    fig = go.Figure()
    palette = qualitative.Dark24 + qualitative.Light24

    for scan_index, (scan_name, scan_poses) in enumerate(scans.items()):
        centers = np.array([pose.center for pose in scan_poses])
        color = palette[scan_index % len(palette)]
        labels = [pose.label for pose in scan_poses]
        hover_rows = [
            [
                pose.label,
                pose.center[0],
                pose.center[1],
                pose.center[2],
            ]
            for pose in scan_poses
        ]

        mode = "lines+markers+text" if show_labels else "lines+markers"
        fig.add_trace(
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode=mode,
                line=dict(color=color, width=4),
                marker=dict(size=4, color=color),
                text=labels if show_labels else None,
                textposition="top center",
                customdata=np.array(hover_rows, dtype=object),
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "X: %{customdata[1]:.6f}<br>"
                    "Y: %{customdata[2]:.6f}<br>"
                    "Z: %{customdata[3]:.6f}<extra></extra>"
                ),
                name=scan_name,
            )
        )

    x_lines: list[list[float | None]] = [[], [], []]
    y_lines: list[list[float | None]] = [[], [], []]
    z_lines: list[list[float | None]] = [[], [], []]

    for pose in all_poses:
        origin = pose.center
        rotation = pose.world_from_camera[:3, :3]
        append_axis_lines(x_lines, origin, origin + rotation[:, 0] * draw_axis_length)
        append_axis_lines(y_lines, origin, origin + rotation[:, 1] * draw_axis_length)
        append_axis_lines(z_lines, origin, origin + rotation[:, 2] * draw_axis_length)

    for label, color, lines in (
        ("X axis", "red", x_lines),
        ("Y axis", "green", y_lines),
        ("Z axis", "blue", z_lines),
    ):
        fig.add_trace(
            go.Scatter3d(
                x=lines[0],
                y=lines[1],
                z=lines[2],
                mode="lines",
                line=dict(color=color, width=4),
                name=label,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"Camera Poses from {output_html.parent.name}",
        width=figure_size[0],
        height=figure_size[1],
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        legend=dict(x=0.02, y=0.98),
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html)
    print(f"Saved pose plot to {output_html}")


def export_pose_table(scans: dict[str, list[CameraPose]], output_csv: Path) -> None:
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scan",
                "pose_file",
                "label",
                "qw",
                "qx",
                "qy",
                "qz",
                "tx",
                "ty",
                "tz",
            ]
        )
        for pose in flatten_scan_poses(scans):
            transform = pose.world_from_camera
            tx, ty, tz = transform[:3, 3]
            qx, qy, qz, qw = R.from_matrix(transform[:3, :3]).as_quat()
            writer.writerow(
                [
                    pose.scan_name,
                    pose.pose_path.name,
                    pose.label,
                    qw,
                    qx,
                    qy,
                    qz,
                    tx,
                    ty,
                    tz,
                ]
            )
    print(f"Saved pose table to {output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot scan camera poses from pose_*.npy files. By default, input "
            "matrices are interpreted as camera-to-world."
        )
    )
    parser.add_argument(
        "directory", type=Path, help="Dataset directory with scan*/ folders."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path. Defaults to <directory>/poses.html.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Output CSV path. Defaults to <directory>/poses.csv.",
    )
    parser.add_argument(
        "--scan-prefix",
        default="scan",
        help="Only plot subdirectories whose names start with this prefix.",
    )
    pose_direction = parser.add_mutually_exclusive_group()
    pose_direction.add_argument(
        "--is-camera-to-world",
        "--is_camera_to_world",
        dest="is_camera_to_world",
        action="store_true",
        default=True,
        help="Interpret stored poses as camera-to-world. This is the default.",
    )
    pose_direction.add_argument(
        "--world-to-camera",
        "--world_to_camera",
        dest="is_camera_to_world",
        action="store_false",
        help="Interpret stored poses as world-to-camera and invert before plotting.",
    )
    parser.add_argument(
        "--source-camera-convention",
        choices=("opengl", "opencv"),
        default="opencv",
        help=(
            "Camera convention used by the stored poses. The viewer plots OpenCV "
            "axes, so OpenGL input flips camera Y/Z after pose-direction handling."
        ),
    )
    parser.add_argument(
        "--axis-length",
        "--axis_length",
        dest="axis_length",
        type=float,
        default=None,
        help="Axis length for each camera frame. Defaults to 2.5%% of scene diagonal.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Draw scan/pose labels next to each camera center.",
    )
    parser.add_argument(
        "--width", type=int, default=1200, help="Viewer width in pixels."
    )
    parser.add_argument(
        "--height", type=int, default=900, help="Viewer height in pixels."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_dir = args.directory.resolve()
    output_html = args.output.resolve() if args.output else dataset_dir / "poses.html"
    output_csv = args.csv.resolve() if args.csv else dataset_dir / "poses.csv"

    scans = load_scan_poses(
        dataset_dir,
        args.scan_prefix,
        is_camera_to_world=args.is_camera_to_world,
        source_camera_convention=args.source_camera_convention,
    )
    if not scans:
        print(f"No pose files found under {dataset_dir}/{args.scan_prefix}*.")
        return

    pose_count = sum(len(scan_poses) for scan_poses in scans.values())
    print(f"Loaded {pose_count} poses from {len(scans)} scans.")

    plot_poses(
        scans,
        output_html,
        axis_length=args.axis_length,
        show_labels=args.show_labels,
        figure_size=(args.width, args.height),
    )
    export_pose_table(scans, output_csv)


if __name__ == "__main__":
    main()
