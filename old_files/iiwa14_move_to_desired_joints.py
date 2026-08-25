import argparse

import matplotlib.pyplot as plt
import numpy as np

from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    DiagramBuilder,
    JointSliders,
    MeshcatVisualizer,
    Simulator,
)
from pydrake.systems.drawing import plot_system_graphviz
from pydrake.systems.primitives import FirstOrderLowPassFilter

from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram


def main(use_hardware: bool, has_wsg: bool) -> None:
    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14.dmd.yaml
    plant_config:
        # For some reason, this requires a small timestep
        time_step: 0.005
        contact_model: "hydroelastic"
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

    num_iiwa_joints = controller_plant.num_positions()
    print("Number of iiwa joints:", num_iiwa_joints)
    filter = builder.AddSystem(
        FirstOrderLowPassFilter(time_constant=1.00, size=num_iiwa_joints)
    )

    builder.Connect(teleop.get_output_port(), filter.get_input_port())

    builder.Connect(
        filter.get_output_port(),
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

    # # display
    # diagram.set_name("diagram")
    # plt.figure()
    # plot_system_graphviz(diagram)
    # plt.savefig("iiwa_teleop_diagram.png")
    # print("The system diagram has been saved to iiwa_teleop_diagram.png")

    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)

    station.internal_meshcat.AddButton("Stop Simulation")
    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)
    station.internal_meshcat.DeleteButton("Stop Simulation")


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
