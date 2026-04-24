import numpy as np

from pydrake.all import RigidTransform, RotationMatrix

from utils.planning import sphere_frame


def generate_centering_waypoints(center, radius, hemisphere_axis=None):
    """
    Generate 5 predefined waypoints for object center localization.

    Point 0 is aimed straight along hemisphere_axis. Points 1-4 are 45° away
    in the up, down, right, and left directions relative to the camera view.

    Args:
        center:          (3,) hemisphere center in world frame
        radius:          float, hemisphere radius
        hemisphere_axis: (3,) unit vector pointing from robot toward object
                         (default: [-1, 0, 0])

    Returns:
        List of 5 RigidTransform waypoints (same format as generate_hemisphere_waypoints)
    """
    if hemisphere_axis is None:
        hemisphere_axis = np.array([-1.0, 0.0, 0.0])

    ha = np.array(hemisphere_axis, dtype=float)
    ha /= np.linalg.norm(ha)

    # Camera "up": project world z onto the plane perpendicular to ha
    world_z = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ha, world_z)) > 0.99:
        world_z = np.array([0.0, 1.0, 0.0])
    up = world_z - np.dot(world_z, ha) * ha
    up /= np.linalg.norm(up)

    # Camera "right": cross(up, ha) gives the right direction when looking along ha
    right = np.cross(up, ha)
    right /= np.linalg.norm(right)

    c = np.cos(np.pi / 4)
    s = np.sin(np.pi / 4)

    # Unit direction vectors from center to each point on the sphere
    directions = [
        ha,  # 0: straight
        c * ha + s * up,  # 1: up
        c * ha - s * up,  # 2: down
        c * ha + s * right,  # 3: right
        c * ha - s * right,  # 4: left
    ]

    waypoints = []
    for d in directions:
        d = d / np.linalg.norm(d)
        point_world = np.asarray(center, dtype=float) + radius * d
        rotation = RotationMatrix(sphere_frame(point_world, ha, center))
        waypoints.append(RigidTransform(rotation, point_world))

    return waypoints
