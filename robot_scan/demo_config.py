import numpy as np

from pydrake.all import RigidTransform, RotationMatrix

# Configuration for different modes
HARDWARE_CONFIG = {
    "speed_factor": 1.0,
    "max_joint_velocity_deg": 30.0,
    "vel_limits": np.full(7, 0.5),  # rad/s
    "acc_limits": np.full(7, 0.5),  # rad/s^2
}

SIMULATION_CONFIG = {
    "speed_factor": 1.0,
    "max_joint_velocity_deg": 30.0,
    "vel_limits": np.full(7, 0.5),  # rad/s
    "acc_limits": np.full(7, 0.5),  # rad/s^2
}


def get_config(use_hardware: bool):
    cfg = HARDWARE_CONFIG if use_hardware else SIMULATION_CONFIG
    # Convert deg/s to rad/s for velocity limits
    cfg["max_joint_velocities"] = np.deg2rad(cfg["max_joint_velocity_deg"] * np.ones(7))
    return cfg


# ===========================================================================
# Shared scan parameters (used across scan_object, localize_scan_object,
# calibrate_microscope, and related demos)
# ===========================================================================

# Robot configuration
ELBOW_ANGLE_DEG = 135.0
ELBOW_ANGLE = np.deg2rad(ELBOW_ANGLE_DEG)

# Kinematic solver reference axes
R_AXIS = np.array([0, 0, -1])
V_AXIS = np.array([0, 1, 0])

# Camera-to-tip rigid transform (constant across all demos)
T_CAM_TO_TIP = RigidTransform(
    RotationMatrix(np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]))
)

# Default robot starting joint configuration (scan_object + localize_scan_object)
# deg: [-63.38, 52.15, 76.73, -121.60, 4.53, -52.45, 126.88]
DEFAULT_POSITION = np.deg2rad([-32.06, 56.57, 47.46, -115.28, -0.89, -70.31, -37.64])

# Hemisphere defaults
HEMISPHERE_DIST = 0.8
HEMISPHERE_RADIUS = 0.08
HEMISPHERE_CENTER = np.array([0.761285, -0.002479, 0.363935])
HEMISPHERE_Z = HEMISPHERE_CENTER[2]
HEMISPHERE_ANGLE_DEG = 0.0
COVERAGE = 1.0

# Camera
CAMERA_SOURCE = 4
