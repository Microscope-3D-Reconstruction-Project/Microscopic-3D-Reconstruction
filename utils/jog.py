import numpy as np

from termcolor import colored

from utils.safety import filter_ik_solutions


def try_jog_tip(
    station,
    q_now: np.ndarray,
    direction_tip: np.ndarray,
    step_m: float,
    kinematics_solver,
    elbow_angle: float,
    joint_lower_limits,
    joint_upper_limits,
    threshold_deg: float = 5.0,
) -> np.ndarray | None:
    """
    Attempt to jog the optical_center by step_m meters in direction_tip (optical_center frame).

    Returns the target joint configuration, or None if the move is unsafe or IK fails.
    """
    plant = station.get_internal_plant()
    plant_context = station.get_internal_plant_context()

    T_oc = plant.GetFrameByName("optical_center").CalcPoseInWorld(plant_context)
    R_oc = T_oc.rotation().matrix()

    # direction is specified in the optical_center frame
    d_world = R_oc @ direction_tip
    target_pos = T_oc.translation() + step_m * d_world
    target_rot = R_oc  # keep orientation fixed while translating

    Q = kinematics_solver.IK_for_microscope(target_rot, target_pos, psi=elbow_angle)
    Q = filter_ik_solutions(
        station,
        Q,
        target_rot,
        target_pos,
        joint_lower_limits,
        joint_upper_limits,
        tip_frame_name="optical_center",
    )

    if Q.shape[0] == 0:
        print(colored("⚠ Jog IK: no valid solutions for this direction", "yellow"))
        return None

    q_des = kinematics_solver.find_closest_solution(Q, q_now)
    delta_deg = np.rad2deg(np.abs(q_des - q_now))

    if np.any(delta_deg > threshold_deg):
        worst_joint = int(np.argmax(delta_deg)) + 1
        worst_deg = delta_deg[worst_joint - 1]
        print(
            colored(
                f"⚠ Jog blocked: joint {worst_joint} would move {worst_deg:.1f}° "
                f"(threshold {threshold_deg:.0f}°)",
                "yellow",
            )
        )
        return None

    return q_des
