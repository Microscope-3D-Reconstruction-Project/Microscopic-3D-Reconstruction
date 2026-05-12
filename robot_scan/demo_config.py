from pathlib import Path

import numpy as np

from pydrake.all import RigidTransform, RotationMatrix

# Configuration for different modes
HARDWARE_CONFIG = {
    "speed_factor": 3.0,
    "max_joint_velocity_deg": 30.0,
    "vel_limits": np.full(7, 0.5),  # rad/s
    "acc_limits": np.full(7, 0.5),  # rad/s^2
}

SIMULATION_CONFIG = {
    "speed_factor": 4.0,
    "max_joint_velocity_deg": 60.0,
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
# DEFAULT_POSITION = np.deg2rad([-32.06, 56.57, 47.46, -115.28, -0.89, -70.31, -37.64])
INTRINSICS_PATH = "/home/rmineyev3/KUKA/Microscopic-3D-Reconstruction/microscope-data/calibrations/latest_2mm/images/camera_calib_intrinsic.json"
DEFAULT_POSITION = np.deg2rad([-15.34, 56.02, 47.29, -117.17, 0.38, -70.58, -37.22])

# Hemisphere defaults
HEMISPHERE_DIST = 0.8
HEMISPHERE_RADIUS = 0.16  # 0.16 for scan_object
HEMISPHERE_CENTER_PATH = Path(
    "/home/rmineyev3/KUKA/Microscopic-3D-Reconstruction/microscope-data/object_centering/20260511_222140/predicted_center.npy"
)
HEMISPHERE_CENTER = np.load(HEMISPHERE_CENTER_PATH)

# [0.7559199631817792, -0.01990998233782262, 0.35170674101999067]
# HEMISPHERE_CENTER = np.array([0.7559, -0.0199, 0.4])

HEMISPHERE_Z = HEMISPHERE_CENTER[2]
HEMISPHERE_ANGLE_DEG = 0.0
COVERAGE = 1.0

# Scan
NUM_SCAN_POINTS = 146

# Camera
CAMERA_SOURCE = 4
