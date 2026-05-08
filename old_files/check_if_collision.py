import argparse

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from pydrake.all import (
    AddFrameTriadIllustration,
    ApplySimulatorConfig,
    BsplineBasis,
    BsplineTrajectory,
    CoulombFriction,
    DiagramBuilder,
    InverseKinematics,
    JointSliders,
    KinematicTrajectoryOptimization,
    KnotVectorType,
    Meshcat,
    MeshcatVisualizer,
    MinimumDistanceLowerBoundConstraint,
    PiecewisePolynomial,
    Rgba,
    RigidTransform,
    SceneGraphCollisionChecker,
    Simulator,
    Solve,
    SpatialInertia,
    Sphere,
    UnitInertia,
)
from pydrake.systems.drawing import plot_system_graphviz
from pydrake.systems.primitives import FirstOrderLowPassFilter

from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra
from iiwa_setup.util.traj_planning import compute_simple_traj_from_q1_to_q2
from iiwa_setup.util.visualizations import draw_sphere

# Personal files
from utils.hemisphere_solver import (
    SphereScorer,
    find_best_hemisphere_center,
    generate_hemisphere_joint_poses,
)
from utils.kuka_geo_kin import KinematicsSolver

"""
Run this file to manually check for collision at different joint poses.
1) Use joint sliders to move the robot.
2) Press "Check Collision" button to see if the current pose is in collision.
    a) The optimization MeshCat will visualize the robot pose for you to see for you to check if there is a collision (localhost:7001 at the time of me testing this)
"""


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
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(scenario=scenario, use_hardware=use_hardware),
    )

    # Load all values I use later
    internal_station = station.internal_station
    internal_plant = station.get_internal_plant()
    controller_plant = station.get_iiwa_controller_plant()

    # Frames
    tip_frame = internal_plant.GetFrameByName("microscope_tip_link")
    link7_frame = internal_plant.GetFrameByName("iiwa_link_7")

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
    station.internal_meshcat.AddButton("Check Collision")

    # ====================================================================
    # Compute all joint poses for sphere scanning
    # ====================================================================
    # Solve example IK
    hemisphere_pos = np.array([0.6666666, 0.0, 0.444444])
    hemisphere_radius = 0.05
    draw_sphere(
        station.internal_meshcat,
        "target_sphere",
        position=hemisphere_pos,
        radius=hemisphere_radius,
    )

    kinematics_solver = KinematicsSolver(station)
    sphere_scorer = SphereScorer(station, kinematics_solver)

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    collision_clicks = 0
    path_idx = 0
    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        if (
            station.internal_meshcat.GetButtonClicks("Check Collision")
            > collision_clicks
        ):
            collision_clicks = station.internal_meshcat.GetButtonClicks(
                "Check Collision"
            )

            # Get values of teleop sliders, not the hardware
            context = simulator.get_context()
            teleop_context = teleop.GetMyContextFromRoot(context)
            q = teleop.get_output_port().Eval(teleop_context)

            print("Checking collision at q:", q)
            sphere_scorer.is_in_self_collision(q)
            # print("Is there a collision?", sphere_scorer.is_in_self_collision(q))

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Check Collision")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )

    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
