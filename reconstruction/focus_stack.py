import glob
import json
import os
import re
import shutil
import subprocess

import hydra
import numpy as np

from omegaconf import DictConfig


def _load_pose_for_scan(
    subdir_path: str, image_files: list, ref_idx: int
) -> np.ndarray:
    """Load the pose matrix for a scan directory.

    Checks in order:
    1. pose_NNNNN.npy files whose numeric suffixes match frame_NNNNN.jpg files
       (same count and same numbers) → loads the pose whose number matches the
       reference frame at ref_idx.
    2. A singular pose.npy file → loads it directly.
    3. Neither → raises FileNotFoundError.
    """
    # Collect numeric suffixes from input frame files.
    frame_numbers = []
    for f in image_files:
        m = re.search(r"frame_(\d+)\.jpg$", os.path.basename(f))
        if m:
            frame_numbers.append(m.group(1))

    # Collect indexed pose_*.npy files.
    pose_indexed = sorted(glob.glob(os.path.join(subdir_path, "pose_*.npy")))
    pose_numbers = []
    for p in pose_indexed:
        m = re.search(r"pose_(\d+)\.npy$", os.path.basename(p))
        if m:
            pose_numbers.append(m.group(1))

    # Case 1: indexed pose files with matching numbers and count as frames.
    if (
        pose_indexed
        and len(pose_numbers) == len(frame_numbers)
        and set(pose_numbers) == set(frame_numbers)
    ):
        ref_number = frame_numbers[ref_idx]
        pose_file = os.path.join(subdir_path, f"pose_{ref_number}.npy")
        return np.load(pose_file)

    # Case 2: singular pose.npy.
    singular = os.path.join(subdir_path, "pose.npy")
    if os.path.exists(singular):
        return np.load(singular)

    raise FileNotFoundError(
        f"No pose .npy files found in {subdir_path}. "
        "Expected either pose_NNNNN.npy files matching frame images, "
        "or a single pose.npy."
    )


def run_focus_stack(cfg: DictConfig) -> None:
    dataset_dir = cfg.dataset_dir
    out_dir = cfg.output_paths.output_dir
    out_image_dir = os.path.join(out_dir, cfg.output_paths.images_subdir)
    out_depth_dir = os.path.join(out_dir, cfg.output_paths.depthmaps_subdir)
    out_mask_dir = os.path.join(out_dir, cfg.output_paths.masks_subdir)
    for path in (out_image_dir, out_depth_dir, out_mask_dir):
        if os.path.isdir(path):
            shutil.rmtree(path)
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    if cfg.scan_dirs is not None:
        scan_dirs = sorted(
            d for d in cfg.scan_dirs if os.path.isdir(os.path.join(dataset_dir, d))
        )
    else:
        scan_dirs = sorted(
            d
            for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
            and d.startswith(cfg.scan_prefix)
        )

    p = cfg.focus_stack_params
    flags = []
    if p.full_resolution_align:
        flags.append("--full-resolution-align")
    if p.global_align:
        flags.append("--global-align")
    if p.align_keep_size:
        flags.append("--align-keep-size")
    if p.no_contrast:
        flags.append("--no-contrast")

    poses = {}

    for subdir in scan_dirs:
        subdir_path = os.path.join(dataset_dir, subdir)
        image_files = sorted(glob.glob(os.path.join(subdir_path, "*.jpg")))
        if not image_files:
            print(f"No .jpg files found in {subdir_path}, skipping.")
            continue

        ref_idx = (
            len(image_files) // 2
        )  # picks the middle one TODO: take a look at this later
        output_file = os.path.join(out_image_dir, f"{subdir}.png")
        depth_file = os.path.join(out_depth_dir, f"{subdir}.png")
        mask_file = os.path.join(out_mask_dir, f"{subdir}.png")
        cmd = (
            ["focus-stack"]
            + flags
            + [
                f"--reference={ref_idx}",
                f"--validmask={mask_file}",
                f"--depthmap={depth_file}",
                f"--output={output_file}",
            ]
            + image_files
        )
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        pose_matrix = _load_pose_for_scan(subdir_path, image_files, ref_idx)
        poses[f"{subdir}.png"] = pose_matrix.tolist()

    poses_path = os.path.join(out_dir, cfg.output_paths.poses_filename)
    with open(poses_path, "w") as f:
        json.dump(poses, f, indent=2)
    print(f"Wrote poses to {poses_path}")

    # Copy intrinsics.json from dataset directory to output directory
    src_intrinsics = os.path.join(dataset_dir, cfg.output_paths.intrinsics_filename)
    dst_intrinsics = os.path.join(out_dir, cfg.output_paths.intrinsics_filename)
    if os.path.exists(src_intrinsics):
        shutil.copy2(src_intrinsics, dst_intrinsics)
        print(f"Copied intrinsics to {dst_intrinsics}")
    else:
        print(f"Warning: intrinsics file not found at {src_intrinsics}, skipping.")


@hydra.main(
    config_path="configs/focus_stack", config_name="default", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    run_focus_stack(cfg)


if __name__ == "__main__":
    main()
