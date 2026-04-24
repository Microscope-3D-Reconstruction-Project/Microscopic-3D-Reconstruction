import numpy as np

from demo_config import T_CAM_TO_TIP
from termcolor import colored

from utils.safety import filter_ik_solutions

# camera_link in Drake = wp @ T_CAM_TO_TIP (tip frame), confirmed visually in Meshcat.
# IK needs wp.rotation() = camera_link.rotation() @ T_CAM_TO_TIP.rotation()
# (T_CAM_TO_TIP.rotation() is its own inverse since it's a symmetric orthogonal matrix)
_R_CAM_TO_TIP = T_CAM_TO_TIP.rotation().matrix()


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
    Attempt to jog the microscope tip by step_m meters in direction_tip (tip frame).

    Returns the target joint configuration, or None if the move is unsafe or IK fails.
    """
    plant = station.get_internal_plant()
    plant_context = station.get_internal_plant_context()

    T_cam = plant.GetFrameByName("camera_link").CalcPoseInWorld(plant_context)
    R_cam = T_cam.rotation().matrix()

    # direction is specified in the tip (= camera_link) frame
    d_world = R_cam @ direction_tip
    target_pos = T_cam.translation() + step_m * d_world

    # IK expects wp.rotation(), which is camera_link.rotation() @ R_cam_to_tip
    target_rot = R_cam @ _R_CAM_TO_TIP

    Q = kinematics_solver.IK_for_microscope(target_rot, target_pos, psi=elbow_angle)
    Q = filter_ik_solutions(
        station, Q, target_rot, target_pos, joint_lower_limits, joint_upper_limits
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
