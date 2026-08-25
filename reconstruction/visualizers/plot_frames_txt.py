#!/usr/bin/env python3
"""
Plot interactive camera-frame poses from a COLMAP-style frames.txt file.

The file format is expected to match COLMAP's text export:
    FRAME_ID RIG_ID QW QX QY QZ TX TY TZ NUM_DATA_IDS ...

`frames.txt` stores `RIG_FROM_WORLD`, so this script inverts each pose before
plotting. With only `frames.txt` available, the plotted camera frame is treated
as the rig frame. That is correct for the common single-camera case; multi-
sensor rigs need additional sensor extrinsics to recover per-sensor poses.

The output is a Plotly HTML viewer that you can orbit with the mouse.

Examples:
    python reconstruction/visualizers/plot_frames_txt.py \
        reconstruction/outputs/circuit_v1/sparse_model/frames.txt
"""

from __future__ import annotations

import argparse

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


@dataclass(frozen=True)
class FramePose:
    frame_id: int
    rig_id: int
    rig_from_world: np.ndarray

    @property
    def world_from_rig(self) -> np.ndarray:
        rotation = self.rig_from_world[:3, :3]
        translation = self.rig_from_world[:3, 3]

        world_from_rig = np.eye(4)
        world_from_rig[:3, :3] = rotation.T
        world_from_rig[:3, 3] = -rotation.T @ translation
        return world_from_rig


def quaternion_to_rotation_matrix(
    qw: float, qx: float, qy: float, qz: float
) -> np.ndarray:
    norm = np.linalg.norm([qw, qx, qy, qz])
    if norm == 0:
        raise ValueError("Quaternion has zero norm.")

    qw, qx, qy, qz = (np.array([qw, qx, qy, qz], dtype=float) / norm).tolist()

    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=float,
    )


def parse_frames_txt(frames_path: Path) -> list[FramePose]:
    frames: list[FramePose] = []

    with frames_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = line.split()
            if len(tokens) < 9:
                raise ValueError(
                    f"{frames_path}:{line_number} does not contain enough pose fields."
                )

            frame_id = int(tokens[0])
            rig_id = int(tokens[1])
            qw, qx, qy, qz, tx, ty, tz = map(float, tokens[2:9])

            transform = np.eye(4)
            transform[:3, :3] = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            transform[:3, 3] = np.array([tx, ty, tz], dtype=float)

            frames.append(
                FramePose(
                    frame_id=frame_id,
                    rig_id=rig_id,
                    rig_from_world=transform,
                )
            )

    if not frames:
        raise ValueError(f"No frame poses found in {frames_path}.")

    return frames


def choose_axis_length(camera_centers: np.ndarray, axis_length: float | None) -> float:
    if axis_length is not None:
        return axis_length

    scene_span = np.ptp(camera_centers, axis=0)
    diagonal = float(np.linalg.norm(scene_span))
    if diagonal > 0:
        return 0.05 * diagonal
    return 0.01


def plot_frames(
    frames: list[FramePose],
    output_path: Path,
    axis_length: float | None,
    figure_size: tuple[int, int],
) -> None:
    world_poses = [frame.world_from_rig for frame in frames]
    camera_centers = np.array([pose[:3, 3] for pose in world_poses])
    draw_axis_length = choose_axis_length(camera_centers, axis_length)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=camera_centers[:, 0],
            y=camera_centers[:, 1],
            z=camera_centers[:, 2],
            mode="lines",
            line=dict(color="rgba(120, 120, 120, 0.7)", width=3),
            name="Frame path",
            hoverinfo="skip",
        )
    )

    x_lines = [[], [], []]
    y_lines = [[], [], []]
    z_lines = [[], [], []]
    labels = []
    hover_rows = []

    for frame, pose in zip(frames, world_poses):
        origin = pose[:3, 3]
        rotation = pose[:3, :3]
        x_axis = rotation[:, 0] * draw_axis_length
        y_axis = rotation[:, 1] * draw_axis_length
        z_axis = rotation[:, 2] * draw_axis_length

        for lines, tip in (
            (x_lines, origin + x_axis),
            (y_lines, origin + y_axis),
            (z_lines, origin + z_axis),
        ):
            for axis_index in range(3):
                lines[axis_index].extend([origin[axis_index], tip[axis_index], None])

        labels.append(str(frame.frame_id))
        hover_rows.append(
            [frame.frame_id, frame.rig_id, origin[0], origin[1], origin[2]]
        )

    for name, color, lines in (
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
                line=dict(color=color, width=5),
                name=name,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=camera_centers[:, 0],
            y=camera_centers[:, 1],
            z=camera_centers[:, 2],
            mode="markers+text",
            marker=dict(size=4, color="black"),
            text=labels,
            textposition="top center",
            customdata=np.array(hover_rows, dtype=float),
            hovertemplate=(
                "Frame ID: %{customdata[0]:.0f}<br>"
                "Rig ID: %{customdata[1]:.0f}<br>"
                "X: %{customdata[2]:.6f}<br>"
                "Y: %{customdata[3]:.6f}<br>"
                "Z: %{customdata[4]:.6f}<extra></extra>"
            ),
            name="Frame origins",
        )
    )

    fig.update_layout(
        title=f"Camera Frames from {output_path.parent.name}",
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True)
    print(f"Saved frame pose plot to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot labeled camera-frame poses from a COLMAP frames.txt file."
    )
    parser.add_argument("frames_txt", type=Path, help="Path to frames.txt")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path. Defaults to <frames_dir>/frames_pose_plot.html",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=None,
        help="Axis length for each camera frame. Defaults to 5%% of scene diagonal.",
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

    frames_path = args.frames_txt.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else frames_path.with_name("frames_pose_plot.html")
    )

    frames = parse_frames_txt(frames_path)
    plot_frames(
        frames=frames,
        output_path=output_path,
        axis_length=args.axis_length,
        figure_size=(args.width, args.height),
    )


if __name__ == "__main__":
    main()
