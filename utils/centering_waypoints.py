import numpy as np

from pydrake.all import RigidTransform, RotationMatrix


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

    # Update these to adjust angle from center point
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)

    # Unit direction vectors from center to each point on the sphere
    directions = [
        ha,  # 0: straight
        c * ha + s * up,  # 1: up
        np.cos(np.pi / 4) * ha - np.sin(np.pi / 4) * up,  # 2: down
        c * ha + s * right,  # 3: right
        c * ha - s * right,  # 4: left
    ]

    waypoints = []
    for d in directions:
        d = d / np.linalg.norm(d)
        point_world = np.asarray(center, dtype=float) + radius * d

        # Build camera frame directly:
        #   z  → inward (toward center), i.e. optical axis points at the object
        #   x  → right direction on the tangent plane (already computed above)
        #   y  → cross(z, x), completing a right-handed frame
        # Using `right` as x instead of projected world_z gives a 90° offset
        # compared to sphere_frame's default reference — no explicit rotation needed.
        z_in = -d
        x = right - np.dot(right, z_in) * z_in  # project right onto tangent plane
        x /= np.linalg.norm(x)
        y = np.cross(z_in, x)
        R = np.column_stack([x, y, z_in])
        waypoints.append(RigidTransform(RotationMatrix(R), point_world))

    return waypoints
