"""
utils/ray_reconstruction.py

Given a clicked pixel (u, v) and the current camera pose from Drake, reconstructs
the 3-D ray that passes through that pixel in the world frame, then visualises it
in Meshcat as a thin cylinder.

Coordinate conventions
----------------------
- Camera frame: +z points along optical axis (into the scene), +x right, +y down.
- Drake plant frame for "camera_link" matches the above convention.
- T_world_cam is a Drake RigidTransform: world_T_camera.
"""

import json

import cv2
import numpy as np

from pydrake.all import Cylinder, Rgba, RigidTransform, RotationMatrix

from robot_scan.demo_config import INTRINSICS_PATH


def load_intrinsics(path: str = INTRINSICS_PATH) -> np.ndarray:
    """Load and return the 3×3 camera intrinsic matrix K from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"], dtype=float)
    assert K.shape == (3, 3), f"Expected (3,3) intrinsic matrix, got {K.shape}"
    return K


def pixel_to_world_ray(
    uv: tuple[int, int],
    K: np.ndarray,
    T_world_cam: RigidTransform,
    dist_coeffs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project a pixel into a 3-D ray expressed in the world frame.

    Args:
        uv:           (u, v) pixel in the original (undisplayed) image resolution.
        K:            3×3 intrinsic matrix.
        T_world_cam:  Drake RigidTransform — pose of camera_link in world frame.
        dist_coeffs:  (k1,k2,p1,p2[,k3,...]) distortion coefficients for
                      cv2.undistortPoints.  Pass None or zeros to skip undistortion.

    Returns:
        origin    (3,) — ray origin in world frame (= camera centre).
        direction (3,) — unit ray direction in world frame.
    """
    # NOTE: distortion coefficients are not calibrated yet — undistortion is
    # effectively skipped until a dist_coeffs array is provided.
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    # Undistort: maps pixel → normalised image plane coordinate
    pt = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pt, K, dist_coeffs)  # shape (1,1,2)
    x_n, y_n = undistorted[0, 0]  # normalised coords (z=1 plane)

    # Ray direction in camera frame
    d_cam = np.array([x_n, y_n, 1.0])
    d_cam /= np.linalg.norm(d_cam)

    # Rotate to world frame
    R_world_cam = T_world_cam.rotation().matrix()
    d_world = R_world_cam @ d_cam
    d_world /= np.linalg.norm(d_world)

    origin = T_world_cam.translation()
    return origin, d_world


def _rotation_z_to_vec(d: np.ndarray) -> RotationMatrix:
    """
    Return a RotationMatrix R such that R @ [0,0,1] = d (unit vector).
    Used to orient a z-aligned Drake Cylinder along an arbitrary direction.
    """
    z = np.array([0.0, 0.0, 1.0])
    d = d / np.linalg.norm(d)
    axis = np.cross(z, d)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(z, d))

    if sin_a < 1e-9:
        # Parallel or anti-parallel
        if cos_a > 0:
            return RotationMatrix()
        else:
            return RotationMatrix(np.diag([1.0, -1.0, -1.0]))

    axis /= sin_a
    K_mat = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    R = np.eye(3) + sin_a * K_mat + (1.0 - cos_a) * (K_mat @ K_mat)
    return RotationMatrix(R)


def visualize_ray(
    meshcat,
    origin: np.ndarray,
    direction: np.ndarray,
    name: str = "clicked_ray",
    length: float = 1.0,
    radius: float = 0.0015,
    color: Rgba = Rgba(1.0, 1.0, 1.0, 0.9),
) -> None:
    """
    Draw a cylinder in Meshcat along the ray origin + t*direction.

    The cylinder starts at `origin` and extends `length` metres along `direction`.
    """
    rot = _rotation_z_to_vec(direction)
    midpoint = origin + (length / 2.0) * direction
    transform = RigidTransform(rot, midpoint)
    meshcat.SetObject(name, Cylinder(radius, length), color)
    meshcat.SetTransform(name, transform)


def least_squares_ray_intersection(
    origins: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """
    Find the point P that minimises the sum of squared perpendicular distances
    to a set of 3-D rays (least-squares closest point to N lines).

    For each ray i:  A_i = I - d_i d_i^T   (projects onto plane ⊥ to d_i)
    Solution:        P = (Σ A_i)^{-1} (Σ A_i o_i)

    Args:
        origins:    (N, 3) ray origins.
        directions: (N, 3) unit ray directions.

    Returns:
        (3,) predicted intersection point.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, directions):
        d = d / np.linalg.norm(d)
        Ai = np.eye(3) - np.outer(d, d)
        A += Ai
        b += Ai @ o
    return np.linalg.solve(A, b)


def visualize_predicted_center(
    meshcat,
    point: np.ndarray,
    name: str = "predicted_center",
    radius: float = 0.005,
    color: Rgba = Rgba(1.0, 1.0, 1.0, 0.9),
) -> None:
    """Draw a sphere in Meshcat at the predicted hemisphere center."""
    from pydrake.all import Sphere

    meshcat.SetObject(name, Sphere(radius), color)
    meshcat.SetTransform(name, RigidTransform(point))


def print_ray(
    origin: np.ndarray,
    direction: np.ndarray,
    scan_idx: int | None = None,
) -> None:
    """Print the ray equation to stdout."""
    tag = f"[scan {scan_idx}] " if scan_idx is not None else ""
    o = origin
    d = direction
    print(
        f"\n  {tag}Ray in world frame:\n"
        f"    origin    : [{o[0]:.6f},  {o[1]:.6f},  {o[2]:.6f}]\n"
        f"    direction : [{d[0]:.6f},  {d[1]:.6f},  {d[2]:.6f}]  (unit)\n"
        f"    equation  : P(t) = [{o[0]:.4f} + t*{d[0]:.4f},  "
        f"{o[1]:.4f} + t*{d[1]:.4f},  "
        f"{o[2]:.4f} + t*{d[2]:.4f}]"
    )
