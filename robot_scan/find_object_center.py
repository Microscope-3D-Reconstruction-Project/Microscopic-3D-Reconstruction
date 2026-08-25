"""
find_object_center.py

Moves the robot to 5 predefined hemisphere viewpoints and collects user clicks
to localize the scan object center via least-squares ray intersection.

Viewpoints:
  0 - straight along -x (hemisphere axis)
  1 - 45° up
  2 - 45° down
  3 - 45° right
  4 - 45° left
"""

import argparse

from pathlib import Path

import numpy as np

from demo_config import (
    DEFAULT_POSITION,
    HEMISPHERE_ANGLE_DEG,
    HEMISPHERE_CENTER,
    HEMISPHERE_DIST,
    HEMISPHERE_RADIUS,
    HEMISPHERE_Z,
    T_CAM_TO_TIP,
    get_config,
)
from manipulation.station import LoadScenario
from pydrake.all import (
    AddFrameTriadIllustration,
    ApplySimulatorConfig,
    ConstantVectorSource,
    DiagramBuilder,
    MeshcatVisualizer,
    Simulator,
)
from pydrake.systems.primitives import VectorLogSink
from termcolor import colored

from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from iiwa_setup.util.visualizations import draw_triad
from utils.centering_waypoints import generate_centering_waypoints
from utils.plotting import plot_hemisphere_waypoints


def main(
    use_hardware: bool,
    hemisphere_dist: float = HEMISPHERE_DIST,
    hemisphere_angle_deg: float = HEMISPHERE_ANGLE_DEG,
    hemisphere_radius: float = HEMISPHERE_RADIUS,
    hemisphere_z: float = HEMISPHERE_Z,
    hemisphere_pos_override: np.ndarray | None = HEMISPHERE_CENTER,
) -> None:
    get_config(use_hardware)  # validates hardware config early

    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14_microscope.dmd.yaml
    plant_config:
        time_step: 0.005
        contact_model: "hydroelastic_with_fallback"
        discrete_contact_approximation: "sap"
    model_drivers:
        iiwa: !IiwaDriver
            lcm_bus: "default"
            control_mode: position_only
    lcm_buses:
        default:
            lcm_url: ""
    """

    # ==================================================================
    # Parameters
    # ==================================================================
    hemisphere_angle = np.deg2rad(hemisphere_angle_deg)
    hemisphere_axis = np.array(
        [-np.cos(hemisphere_angle), -np.sin(hemisphere_angle), 0.0]
    )

    if hemisphere_pos_override is not None:
        hemisphere_pos = np.asarray(hemisphere_pos_override, dtype=float)
        print(colored(f"✓ hemisphere_pos overridden to {hemisphere_pos}", "cyan"))
    else:
        hemisphere_pos = np.array(
            [
                hemisphere_dist * np.cos(hemisphere_angle),
                hemisphere_dist * np.sin(hemisphere_angle),
                hemisphere_z,
            ]
        )

    # ==================================================================
    # Waypoint generation
    # ==================================================================
    hemisphere_waypoints = generate_centering_waypoints(
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
    )

    labels = ["straight", "up", "down", "right", "left"]
    print(colored("\n5 centering waypoints:", "cyan"))
    for i, (wp, label) in enumerate(zip(hemisphere_waypoints, labels)):
        print(f"  [{i}] {label}: {np.round(wp.translation(), 4)}")

    # ==================================================================
    # Matplotlib plot
    # ==================================================================
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plot_hemisphere_waypoints(
        hemisphere_waypoints,
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        output_path=outputs_dir / "centering_waypoints.png",
        visualize=False,
    )

    # ==================================================================
    # Diagram setup
    # ==================================================================
    builder = DiagramBuilder()
    scenario = LoadScenario(data=scenario_data)

    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            hemisphere_pos=hemisphere_pos,
            hemisphere_radius=hemisphere_radius,
            use_hardware=use_hardware,
        ),
    )

    state_logger = builder.AddSystem(VectorLogSink(7))
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        state_logger.get_input_port(),
    )

    dummy = builder.AddSystem(ConstantVectorSource(DEFAULT_POSITION))
    builder.Connect(dummy.get_output_port(), station.GetInputPort("iiwa.position"))

    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    camera_frame = station.get_internal_plant().GetFrameByName("camera_link")
    AddFrameTriadIllustration(
        scene_graph=station.internal_station.get_scene_graph(),
        plant=station.get_internal_plant(),
        frame=camera_frame,
        length=0.1,
        radius=0.002,
        name="camera_link",
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    meshcat = station.internal_meshcat

    # ==================================================================
    # Draw waypoints in Meshcat
    # ==================================================================
    for i, (wp, label) in enumerate(zip(hemisphere_waypoints, labels)):
        draw_triad(
            meshcat,
            f"centering_waypoint_{i}_{label}",
            wp @ T_CAM_TO_TIP,
            length=0.02,
            radius=0.001,
            opacity=1.0,
        )

    meshcat.AddButton("Stop Simulation")
    print(
        colored(
            f"\n✓ Setup complete. Waypoints drawn in Meshcat.\n"
            f"  Plot saved to {outputs_dir / 'centering_waypoints.png'}\n"
            f"  Press 'Stop Simulation' in Meshcat to exit.",
            "green",
        )
    )

    # ==================================================================
    # Simulation loop
    # ==================================================================
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.05)

    meshcat.DeleteButton("Stop Simulation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware", action="store_true", help="Connect to real iiwa hardware."
    )
    parser.add_argument(
        "--hemisphere_dist",
        type=float,
        default=None,
        help=f"Distance from world origin to hemisphere center (default: {HEMISPHERE_DIST}).",
    )
    parser.add_argument(
        "--hemisphere_angle",
        type=float,
        default=None,
        help=f"Hemisphere approach angle in degrees (default: {HEMISPHERE_ANGLE_DEG}).",
    )
    parser.add_argument(
        "--hemisphere_radius",
        type=float,
        default=None,
        help=f"Hemisphere scan radius in meters (default: {HEMISPHERE_RADIUS}).",
    )
    parser.add_argument(
        "--hemisphere_z",
        type=float,
        default=None,
        help=f"Z height of hemisphere center in world frame (default: {HEMISPHERE_Z}).",
    )
    parser.add_argument(
        "--hemisphere_pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Directly set hemisphere center position (overrides dist/angle/z).",
    )
    args = parser.parse_args()
    main(
        use_hardware=args.use_hardware,
        hemisphere_dist=args.hemisphere_dist
        if args.hemisphere_dist is not None
        else HEMISPHERE_DIST,
        hemisphere_angle_deg=args.hemisphere_angle
        if args.hemisphere_angle is not None
        else HEMISPHERE_ANGLE_DEG,
        hemisphere_radius=args.hemisphere_radius
        if args.hemisphere_radius is not None
        else HEMISPHERE_RADIUS,
        hemisphere_z=args.hemisphere_z
        if args.hemisphere_z is not None
        else HEMISPHERE_Z,
        hemisphere_pos_override=np.array(args.hemisphere_pos)
        if args.hemisphere_pos is not None
        else HEMISPHERE_CENTER,
    )
