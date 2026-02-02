import numpy as np

from pydrake.all import Rgba, RigidTransform, Sphere


def draw_sphere(meshcat, name, position, radius=0.01):
    rgba = Rgba(0.0, 1.0, 0.1, 0.5)

    meshcat.SetObject(
        name,
        Sphere(radius),
        rgba,
    )
    meshcat.SetTransform(
        name,
        RigidTransform(np.array(position)),
    )
