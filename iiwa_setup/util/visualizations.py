from pathlib import Path

import cv2
import numpy as np

from pydrake.all import (
    Box,
    CoulombFriction,
    Cylinder,
    Mesh,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Sphere,
)

_ASSETS = Path(__file__).parent


def _generate_wood_assets(
    floor_length: float = 1.8,
    floor_width: float = 3.0,
    floor_thickness: float = 0.05,
    tex_size: int = 512,
) -> None:
    """Generate wood_floor.png, wood_floor.mtl, and wood_floor.obj in _ASSETS."""
    png_path = _ASSETS / "wood_floor.png"
    mtl_path = _ASSETS / "wood_floor.mtl"
    obj_path = _ASSETS / "wood_floor.obj"

    # --- texture --------------------------------------------------------
    rng = np.random.default_rng(42)
    base = np.array([200, 182, 152], dtype=np.float32) / 255.0  # neutral light tan
    img = np.tile(base, (tex_size, tex_size, 1)).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, (3, tex_size))
    freqs, weights = [0.25, 0.7, 1.8], [0.55, 0.30, 0.15]
    for y in range(tex_size):
        grain = sum(
            w * np.sin(f * y + phase[i, y])
            for i, (f, w) in enumerate(zip(freqs, weights))
        )
        img[y] *= 1.0 + 0.13 * grain
    img += rng.normal(0, 0.018, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    cv2.imwrite(str(png_path), (img[:, :, ::-1] * 255).astype(np.uint8))

    # --- MTL ------------------------------------------------------------
    mtl_path.write_text(
        "newmtl wood\n"
        "Ka 0.78 0.71 0.60\n"
        "Kd 0.78 0.71 0.60\n"
        "Ks 0.22 0.20 0.17\n"
        "Ns 48.0\n"
        f"map_Kd {png_path.name}\n"
    )

    # --- OBJ: top face of the slab with tiled UVs -----------------------
    hx, hy = floor_length / 2.0, floor_width / 2.0
    zt = floor_thickness / 2.0  # top face z in geometry-local frame
    # Tile the texture ~4× along the long axis and ~6× across
    u_tiles, v_tiles = 4.0, 6.0
    obj_path.write_text(
        f"mtllib {mtl_path.name}\n"
        "\n"
        f"v -{hx} -{hy}  {zt}\n"
        f"v  {hx} -{hy}  {zt}\n"
        f"v  {hx}  {hy}  {zt}\n"
        f"v -{hx}  {hy}  {zt}\n"
        "\n"
        f"vt 0.0     0.0\n"
        f"vt {u_tiles} 0.0\n"
        f"vt {u_tiles} {v_tiles}\n"
        f"vt 0.0     {v_tiles}\n"
        "\n"
        "vn 0.0 0.0 1.0\n"
        "\n"
        "usemtl wood\n"
        "f 1/1/1 2/2/1 3/3/1\n"
        "f 1/1/1 3/3/1 4/4/1\n"
    )


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


# TODO: These are collisions as well so maybe don't just add into visualizations.py?
def draw_triad(meshcat, name, transform, length=0.1, radius=0.005, opacity=1.0):
    """Draws a coordinate frame triad in Meshcat at the given RigidTransform."""
    meshcat.SetObject(f"{name}/x", Cylinder(radius, length), Rgba(1, 0, 0, opacity))
    meshcat.SetTransform(
        f"{name}/x",
        RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2, 0, 0]),
    )

    meshcat.SetObject(f"{name}/y", Cylinder(radius, length), Rgba(0, 1, 0, opacity))
    meshcat.SetTransform(
        f"{name}/y",
        RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2, 0]),
    )

    meshcat.SetObject(f"{name}/z", Cylinder(radius, length), Rgba(0, 0, 1, opacity))
    meshcat.SetTransform(f"{name}/z", RigidTransform([0, 0, length / 2]))

    # Set the overall frame transform
    meshcat.SetTransform(name, transform)


def draw_wireframe_sphere(meshcat, name, position, radius, n=64, rgba=Rgba(1, 1, 0, 1)):
    t = np.linspace(0, 2 * np.pi, n)
    c, s = np.cos(t) * radius, np.sin(t) * radius
    z = np.zeros_like(t)
    for axis, pts in [
        ("xy", np.vstack([c, s, z])),
        ("xz", np.vstack([c, z, s])),
        ("yz", np.vstack([z, c, s])),
    ]:
        meshcat.SetLine(f"{name}/{axis}", pts + np.array(position)[:, None], 2.0, rgba)


def add_sphere(
    plant,
    position,
    radius=0.01,
    name="sphere",
    color=[0.0, 1.0, 0.0, 0.2],
    collision=True,
):
    friction = CoulombFriction(static_friction=0.9, dynamic_friction=0.8)

    if radius <= 0:
        radius = 0.01  # default small radius to avoid issues with zero-size geometry
    sphere_shape = Sphere(radius)
    X_WC = RigidTransform(np.array(position))

    if collision:
        plant.RegisterCollisionGeometry(
            plant.world_body(),
            X_WC,
            sphere_shape,
            f"{name}_collision",
            friction,
        )

    # Optional: visualization
    plant.RegisterVisualGeometry(
        plant.world_body(),
        X_WC,
        sphere_shape,
        f"{name}_visual",
        color,
    )
    # draw_wireframe_sphere(
    #     plant.GetMeshcat(),
    #     f"{name}_visual",
    #     position,
    #     radius,
    #     rgba=Rgba(*color),
    # )


def add_floor(plant, floor_length: float = 1.8):
    friction = CoulombFriction(static_friction=0.9, dynamic_friction=0.8)
    floor_thickness = 0.05
    floor_width = floor_length  # square
    floor_size = Box(floor_length, floor_width, floor_thickness)
    X_WF = RigidTransform(
        [floor_length / 2 - 0.3, 0, -floor_thickness / 2]
    )  # top surface at z=0

    plant.RegisterCollisionGeometry(
        plant.world_body(),
        X_WF,
        floor_size,
        "floor_collision",
        friction,
    )

    # Always regenerate assets so dimensions stay in sync with floor_length.
    _generate_wood_assets(floor_length, floor_width, floor_thickness)

    # Visual uses the OBJ so Meshcat picks up the MTL texture; Box handles collision.
    plant.RegisterVisualGeometry(
        plant.world_body(),
        X_WF,
        Mesh(str(_ASSETS / "wood_floor.obj"), scale=1.0),
        "floor_visual",
        np.array([1.0, 1.0, 1.0, 1.0]),
    )


def add_wall(
    plant,
    wall_width=1.8,
    wall_height=1.5,
    X_WF=None,
    wall_color=[58 / 255.0, 85 / 255.0, 69 / 255.0, 0.3],
):
    if not hasattr(add_wall, "counter"):
        add_wall.counter = 0

    add_wall.counter += 1

    friction = CoulombFriction(static_friction=0.9, dynamic_friction=0.8)
    wall_thickness = 0.01

    wall_size = Box(wall_thickness, wall_width, wall_height)

    if X_WF is None:
        X_WF = RigidTransform([-0.3 - wall_thickness / 2, 0, wall_height / 2])

    plant.RegisterCollisionGeometry(
        plant.world_body(),
        X_WF,
        wall_size,
        f"wall_{add_wall.counter}_collision",
        friction,
    )

    # Optional: visualization
    plant.RegisterVisualGeometry(
        plant.world_body(),
        X_WF,
        wall_size,
        f"wall_{add_wall.counter}_visual",
        wall_color,
    )
