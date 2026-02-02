import numpy as np

from pydrake.all import PiecewisePolynomial

from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra


def compute_simple_traj_from_q1_to_q2(
    controller_plant,
    q1: np.ndarray,
    q2: np.ndarray,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
):
    print("Generating simple trajectory from q1 to q2")
    path = PiecewisePolynomial.FirstOrderHold([0, 1], np.column_stack((q1, q2)))

    print("Updating with TOPPRA to enforce velocity and acceleration limits")
    traj = reparameterize_with_toppra(
        path,
        controller_plant,
        velocity_limits=vel_limits,
        acceleration_limits=acc_limits,
    )

    print("Trajectory generation complete!")
    return traj
