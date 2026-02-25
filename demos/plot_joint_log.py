#!/usr/bin/env python3
"""
Interactive joint trajectory visualization using Plotly.
Displays all 7 joints with their limits in separate subplots.
"""

import argparse
import pickle

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

# IIWA14 joint limits (radians)
# From KUKA iiwa14 documentation
JOINT_LIMITS = {
    "lower": np.array(
        [
            -2.96705973,
            -2.0943951,
            -2.96705973,
            -2.0943951,
            -2.96705973,
            -2.0943951,
            -3.05432619,
        ]
    ),
    "upper": np.array(
        [
            2.96705973,
            2.0943951,
            2.96705973,
            2.0943951,
            2.96705973,
            2.0943951,
            3.05432619,
        ]
    ),
}

JOINT_NAMES = [
    "Joint 1 (Base)",
    "Joint 2",
    "Joint 3",
    "Joint 4",
    "Joint 5",
    "Joint 6",
    "Joint 7 (Tip)",
]


def load_joint_log(log_path: Path):
    """Load joint log data from pickle file."""
    with open(log_path, "rb") as f:
        log_data = pickle.load(f)
    return log_data


def create_interactive_plot(times, joint_positions, output_html=None):
    """
    Create interactive Plotly visualization of joint trajectories.

    Args:
        times: (N,) array of timestamps
        joint_positions: (7, N) array of joint positions in radians
        output_html: Optional path to save HTML file
    """
    # Convert to degrees for easier interpretation
    joint_positions_deg = np.rad2deg(joint_positions)
    joint_limits_deg = {
        "lower": np.rad2deg(JOINT_LIMITS["lower"]),
        "upper": np.rad2deg(JOINT_LIMITS["upper"]),
    }

    # Create subplots - 7 rows, 1 column
    fig = make_subplots(
        rows=7,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=JOINT_NAMES,
    )

    # Add traces for each joint
    for i in range(7):
        # Joint trajectory
        fig.add_trace(
            go.Scatter(
                x=times,
                y=joint_positions_deg[i, :],
                mode="lines",
                name=f"Joint {i+1}",
                line=dict(color=f"hsl({i*50}, 70%, 50%)", width=2),
                hovertemplate="<b>Joint %{i+1}</b><br>"
                + "Time: %{x:.3f}s<br>"
                + "Angle: %{y:.2f}°<br>"
                + "<extra></extra>",
            ),
            row=i + 1,
            col=1,
        )

        # Upper limit line
        fig.add_trace(
            go.Scatter(
                x=[times[0], times[-1]],
                y=[joint_limits_deg["upper"][i], joint_limits_deg["upper"][i]],
                mode="lines",
                name="Upper Limit",
                line=dict(color="red", width=1, dash="dash"),
                showlegend=(i == 0),  # Only show in legend once
                hoverinfo="skip",
            ),
            row=i + 1,
            col=1,
        )

        # Lower limit line
        fig.add_trace(
            go.Scatter(
                x=[times[0], times[-1]],
                y=[joint_limits_deg["lower"][i], joint_limits_deg["lower"][i]],
                mode="lines",
                name="Lower Limit",
                line=dict(color="red", width=1, dash="dash"),
                showlegend=(i == 0),  # Only show in legend once
                hoverinfo="skip",
            ),
            row=i + 1,
            col=1,
        )

        # Shaded region for valid range
        fig.add_trace(
            go.Scatter(
                x=[times[0], times[-1], times[-1], times[0]],
                y=[
                    joint_limits_deg["lower"][i],
                    joint_limits_deg["lower"][i],
                    joint_limits_deg["upper"][i],
                    joint_limits_deg["upper"][i],
                ],
                fill="toself",
                fillcolor="rgba(0, 255, 0, 0.1)",
                line=dict(width=0),
                showlegend=(i == 0),
                name="Valid Range",
                hoverinfo="skip",
            ),
            row=i + 1,
            col=1,
        )

        # Update y-axis for this subplot
        fig.update_yaxes(
            title_text="Angle (deg)",
            row=i + 1,
            col=1,
            range=[
                joint_limits_deg["lower"][i] - 10,
                joint_limits_deg["upper"][i] + 10,
            ],
        )

    # Update x-axis (only bottom one since shared)
    fig.update_xaxes(title_text="Time (s)", row=7, col=1)

    # Update layout
    fig.update_layout(
        height=1400,  # Tall enough for 7 subplots
        title_text="IIWA14 Joint Trajectories",
        hovermode="x unified",
        showlegend=True,
        legend=dict(x=1.05, y=1, xanchor="left", yanchor="top"),
    )

    # Show the plot
    fig.show()

    # Optionally save to HTML
    if output_html:
        fig.write_html(output_html)
        print(f"Saved interactive plot to {output_html}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot joint trajectories from logged data"
    )
    # parser.add_argument(
    #     "log_file",
    #     type=Path,
    #     help="Path to joint log pickle file",
    # )
    # parser.add_argument(
    #     "--output",
    #     type=Path,
    #     default=None,
    #     help="Optional: Save plot to HTML file",
    # )

    args = parser.parse_args()

    # Load data
    log_file = Path(__file__).parent.parent / "outputs" / "joint_log.pkl"
    print(f"Loading joint log from {log_file}")
    log_data = load_joint_log(log_file)

    # Extract times and positions
    times = log_data["sample_times"]
    positions = log_data["data"]  # Shape: (7, N)

    print(f"Loaded {len(times)} samples spanning {times[-1]:.2f} seconds")
    print(f"Joint positions shape: {positions.shape}")

    # Create plot
    create_interactive_plot(times, positions)


if __name__ == "__main__":
    main()
