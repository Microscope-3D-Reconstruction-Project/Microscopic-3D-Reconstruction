# General imports
import argparse

from enum import Enum, auto
from pathlib import Path

import numpy as np

# Drake imports
from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    DiagramBuilder,
    JointSliders,
    MeshcatVisualizer,
    PiecewisePolynomial,
    Rgba,
    Simulator,
    Solve,
)
from pydrake.systems.drawing import plot_system_graphviz
from pydrake.systems.primitives import FirstOrderLowPassFilter
from termcolor import colored

# Personal files
from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra
from iiwa_setup.util.traj_planning import (
    add_collision_constraints_to_trajectory,
    resolve_with_toppra,
    setup_trajectory_optimization_from_q1_to_q2,
)
from iiwa_setup.util.visualizations import draw_sphere
from utils.hemisphere_solver import load_joint_poses_from_csv
from utils.kuka_geo_kin import KinematicsSolver


class State(Enum):
    IDLE = auto()
    MOVING = auto()


def main(use_hardware: bool) -> None:
    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14_microscope.dmd.yaml
    # - add_model:
    #     name: sphere_obstacle
    #     file: package://iiwa_setup/sphere_obstacle.sdf
    # - add_weld:
    #     parent: world
    #     child: sphere_obstacle::sphere_body
    #     X_PC:
    #         translation: [0.5, 0.0, 0.6]
    plant_config:
        # For some reason, this requires a small timestep
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

    # ===================================================================
    # Diagram Setup
    # ===================================================================
    builder = DiagramBuilder()

    # Load scenario
    scenario = LoadScenario(data=scenario_data)
    hemisphere_pos = np.array([0.6666666, 0.0, 0.444444])
    hemisphere_radius = 0.05
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            hemisphere_pos=hemisphere_pos,
            hemisphere_radius=hemisphere_radius,
            use_hardware=use_hardware,
        ),
    )

    # Load all values I use later
    controller_plant = station.get_iiwa_controller_plant()

    # Load teleop sliders
    teleop = builder.AddSystem(
        JointSliders(
            station.internal_meshcat,
            controller_plant,
        )
    )

    # Add connections
    builder.Connect(
        teleop.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )

    # Visualize internal station with Meshcat
    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    # Build diagram
    diagram = builder.Build()

    # ====================================================================
    # Simulator Setup
    # ====================================================================
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)

    station.internal_meshcat.AddButton("Stop Simulation")
    station.internal_meshcat.AddButton("Execute Trajectory")

    # ================================================================
    # Spiral trajectory generation
    # ================================================================
    t_final = 100.0
    num_points = 100
    t = np.linspace(0, t_final, num_points)

    a = 0.05  # initial radius
    b = 0.01  # growth per radian
    theta = np.linspace(0, 4 * np.pi, num_points)

    x = (a + b * theta) * np.cos(theta)
    y = (a + b * theta) * np.sin(theta)
    z = np.zeros_like(x)  # flat spiral

    points = np.vstack([x, y, z])
    # trajectory = PiecewisePolynomial.FirstOrderHold(t, points)

    # Step 1) Solve IK for desired pose
    kinematics_solver = KinematicsSolver(station)
    q_prev = None
    q_curr = None

    trajectory_joint_poses = []
    for i in range(num_points):
        eef_pos = (
            points[:, i] + hemisphere_pos
        )  # Shift spiral to be around the hemisphere center
        eef_rot = np.eye(3)  # Keep end-effector orientation fixed

        Q = kinematics_solver.IK_for_microscope(  # NOTE: Just using 0 elbow angle for now
            eef_rot, eef_pos
        )

        if q_prev is not None:
            # Choose the solution closest to the previous one for smoothness
            q_curr = kinematics_solver.find_closest_solution(Q, q_prev)
        else:
            q_curr = Q[0]  # Just pick the first solution if no previous solution exists

        trajectory_joint_poses.append(q_curr)
        q_prev = q_curr

    trajectory_joint_poses = np.array(trajectory_joint_poses).T  # Shape (7, num_points)

    # Visualize traj in matplotlib
    import os

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    for i in range(7):
        plt.plot(t, trajectory_joint_poses[i, :], label=f"Joint {i+1}")
    plt.title("Joint Trajectories for Spiral End-Effector Path")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (rad)")
    plt.legend()
    plt.grid()

    # Save to outputs folder
    current_dir = Path(__file__).parent.parent
    outputs_dir = current_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    save_path = outputs_dir / "spiral_trajectory.png"
    plt.savefig(save_path)
    print(colored(f"Trajectory plot saved to {save_path}", "cyan"))

    # Convert to PiecewisePolynomial for execution
    trajectory = PiecewisePolynomial.FirstOrderHold(t, trajectory_joint_poses)

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    state = State.IDLE

    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        if (
            state == State.IDLE
            and station.internal_meshcat.GetButtonClicks("Execute Trajectory") > 0
        ):
            print(colored("Executing trajectory!", "green"))
            state = State.MOVING
            trajectory_start_time = simulator.get_context().get_time()

        elif state == State.MOVING:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if traj_time <= trajectory.end_time():
                q_desired = trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )
            else:
                print(colored("✓ Trajectory execution complete!", "green"))
                trajectory = None
                state = State.INITIAL_GUESS_PLANNING

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Execute Trajectory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )

    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
