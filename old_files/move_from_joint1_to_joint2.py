import argparse

import matplotlib.pyplot as plt
import numpy as np

from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    CoulombFriction,
    DiagramBuilder,
    InverseKinematics,
    JointSliders,
    MeshcatVisualizer,
    MinimumDistanceLowerBoundConstraint,
    PiecewisePolynomial,
    Rgba,
    RigidTransform,
    SceneGraphCollisionChecker,
    Simulator,
    SpatialInertia,
    Sphere,
    UnitInertia,
)
from pydrake.systems.drawing import plot_system_graphviz
from pydrake.systems.primitives import FirstOrderLowPassFilter

from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra


def main(use_hardware: bool, has_wsg: bool) -> None:
    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14.dmd.yaml
    - add_model:
        name: sphere_obstacle
        file: package://iiwa_setup/sphere_obstacle.sdf
    - add_weld:
        parent: world
        child: sphere_obstacle::sphere_body
        X_PC:
            translation: [0.5, 0, 0.5]
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

    builder = DiagramBuilder()

    scenario = LoadScenario(data=scenario_data)
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario, has_wsg=has_wsg, use_hardware=use_hardware
        ),
    )

    # Set up teleop widgets
    controller_plant = station.get_iiwa_controller_plant()
    teleop = builder.AddSystem(
        JointSliders(
            station.internal_meshcat,
            controller_plant,
        )
    )

    # num_iiwa_joints = controller_plant.num_positions()
    # print("Number of iiwa joints:", num_iiwa_joints)
    # filter = builder.AddSystem(FirstOrderLowPassFilter(
    #     time_constant=1.00, size=num_iiwa_joints))

    # builder.Connect(
    #     teleop.get_output_port(), filter.get_input_port()
    # )

    builder.Connect(
        teleop.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )

    if has_wsg:
        wsg_teleop = builder.AddSystem(WsgButton(station.internal_meshcat))
        builder.Connect(
            wsg_teleop.get_output_port(0), station.GetInputPort("wsg.position")
        )

    # Required for visualizing the internal station
    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    diagram = builder.Build()

    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)

    station.internal_meshcat.AddButton("Stop Simulation")
    station.internal_meshcat.AddButton("Move to Goal")

    q_goal = np.array([0, np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0])
    vel_limits = np.full(7, 0.2)  # rad/s
    acc_limits = np.full(7, 0.2)  # rad/s^2

    move_clicks = 0
    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        if station.internal_meshcat.GetButtonClicks("Move to Goal") > move_clicks:
            move_clicks = station.internal_meshcat.GetButtonClicks("Move to Goal")
            print(f"Moving to goal: {q_goal}")

            # 1. Get current position
            # Get the real-time context for the station
            station_context = station.GetMyContextFromRoot(simulator.get_context())
            # Read the measured position from the station (works for both Sim and Hardware)
            q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )
            print("Current joint positions:", q_current)

            # 2. Create geometric path
            path = PiecewisePolynomial.FirstOrderHold(
                [0, 1], np.column_stack((q_current, q_goal))
            )

            # 3. Parameterize with TOPPRA
            print("Generating trajectory...")
            traj = reparameterize_with_toppra(
                path,
                controller_plant,
                velocity_limits=vel_limits,
                acceleration_limits=acc_limits,
            )
            print(f"Trajectory duration: {traj.end_time():.2f}s")

            # 4. Execute trajectory
            t_traj = 0.0
            dt = 0.01
            # Assuming 'simulator' is available here
            t_start = simulator.get_context().get_time()

            while t_traj < traj.end_time():
                # Get desired state
                q_d = traj.value(t_traj).flatten()

                # Send command
                teleop.SetPositions(q_d)

                # Advance simulation
                step = min(dt, traj.end_time() - t_traj)
                simulator.AdvanceTo(t_start + t_traj + step)
                t_traj += step

            print("Goal reached.")

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)
    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Move to Goal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )
    parser.add_argument(
        "--has_wsg",
        action="store_true",
        help="Whether the iiwa has a WSG gripper or not.",
    )
    args = parser.parse_args()
    main(use_hardware=args.use_hardware, has_wsg=args.has_wsg)
