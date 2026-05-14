"""Convert a 3D Gaussian Splatting .pt checkpoint to a .ply file.

Usage:
    python ckpt_to_ply.py <path/to/ckpt.pt> <output.ply>
"""

import os
import sys

import torch

from gsplat import export_splats


def ckpt_to_ply(ckpt_path: str, output_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    splats = ckpt["splats"]

    print(f"Step: {ckpt['step']}")
    print(f"Number of Gaussians: {len(splats['means'])}")

    # gsplat's export_splats has a bug: it calls .any(dim=0) on the 1-D opacities
    # tensor, which collapses to a scalar — if *any* opacity is Inf the whole mask
    # becomes True and every splat is dropped.  Clamp here to avoid that.
    opacities = splats["opacities"].nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    export_splats(
        means=splats["means"],
        scales=splats["scales"],
        quats=splats["quats"],
        opacities=opacities,
        sh0=splats["sh0"],
        shN=splats["shN"],
        format="ply",
        save_to=output_path,
    )

    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"Saved to {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ckpt_to_ply.py <ckpt.pt> <output.ply>")
        sys.exit(1)
    ckpt_to_ply(sys.argv[1], sys.argv[2])
