import glob
import json
import os
import re
import shutil
import subprocess

import hydra
import numpy as np

from focus_stack.sharpness_scoring import (
    VALID_SHARPNESS_METHODS,
    VALID_SHARPNESS_SELECTION_MODES,
    VALID_SHARPNESS_WINDOW_WEIGHTS,
    find_sharpest_image_index,
    find_sharpest_window_reference_index,
    save_sharpness_bar_chart,
)
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
        # m = re.search(r"frame_(\d+)\.png$", os.path.basename(f))
        m = re.search(r"frame_(\d+)\.jpg$", os.path.basename(f))

        if m:
            frame_numbers.append(m.group(1))

    # Collect indexed pose_*.npy files.
    pose_indexed = sorted(glob.glob(os.path.join(subdir_path, "pose_*.npy")))
    pose_numbers = []
    for pose in pose_indexed:
        m = re.search(r"pose_(\d+)\.npy$", os.path.basename(pose))
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


def _validate_focal_stack_size(focal_stack_size: int | None) -> int | None:
    if focal_stack_size is None:
        return None

    if isinstance(focal_stack_size, bool):
        raise ValueError(
            "focus_stack_params.focal_stack_size must be an odd integer or null."
        )

    try:
        normalized_size = int(focal_stack_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "focus_stack_params.focal_stack_size must be an odd integer or null."
        ) from exc

    if isinstance(focal_stack_size, float) and not focal_stack_size.is_integer():
        raise ValueError(
            "focus_stack_params.focal_stack_size must be an odd integer or null, "
            f"got {focal_stack_size}."
        )
    focal_stack_size = normalized_size

    if focal_stack_size <= 0 or focal_stack_size % 2 == 0:
        raise ValueError(
            "focus_stack_params.focal_stack_size must be a positive odd integer "
            f"or null, got {focal_stack_size}."
        )

    return focal_stack_size


def _validate_sharpness_score(sharpness_score: str) -> str:
    if sharpness_score not in VALID_SHARPNESS_METHODS:
        raise ValueError(
            "sharpness_score must be one of "
            f"{VALID_SHARPNESS_METHODS}, got {sharpness_score!r}."
        )
    return sharpness_score


def _validate_sharpness_selection_mode(sharpness_selection_mode: str) -> str:
    if sharpness_selection_mode not in VALID_SHARPNESS_SELECTION_MODES:
        raise ValueError(
            "sharpness_selection_mode must be one of "
            f"{VALID_SHARPNESS_SELECTION_MODES}, got {sharpness_selection_mode!r}."
        )
    return sharpness_selection_mode


def _validate_sharpness_window_weights(sharpness_window_weights: str) -> str:
    if sharpness_window_weights not in VALID_SHARPNESS_WINDOW_WEIGHTS:
        raise ValueError(
            "sharpness_window_weights must be one of "
            f"{VALID_SHARPNESS_WINDOW_WEIGHTS}, got {sharpness_window_weights!r}."
        )
    return sharpness_window_weights


def _validate_sharpness_gaussian_sigma(
    sharpness_gaussian_sigma: float | None,
) -> float | None:
    if sharpness_gaussian_sigma is None:
        return None
    sharpness_gaussian_sigma = float(sharpness_gaussian_sigma)
    if sharpness_gaussian_sigma <= 0:
        raise ValueError(
            "sharpness_gaussian_sigma must be positive or null, got "
            f"{sharpness_gaussian_sigma}."
        )
    return sharpness_gaussian_sigma


def _select_focus_stack_images(
    image_files: list,
    ref_idx: int,
    focal_stack_size: int | None,
    scan_name: str | None = None,
) -> tuple[list, int, list[int]]:
    """Return the image subset and reference index to pass to focus-stack."""
    if focal_stack_size is None:
        return image_files, ref_idx, list(range(len(image_files)))

    half_window = focal_stack_size // 2
    start_idx = max(0, ref_idx - half_window)
    end_idx = min(len(image_files), ref_idx + half_window + 1)
    selected_indices = list(range(start_idx, end_idx))
    selected_ref_idx = ref_idx - start_idx

    if len(selected_indices) < focal_stack_size:
        prefix = f"{scan_name}: " if scan_name is not None else ""
        print(
            f"Warning: {prefix}requested focal_stack_size={focal_stack_size} "
            f"around reference index {ref_idx}, but scan has {len(image_files)} "
            f"images. Using {len(selected_indices)} available images instead."
        )

    return image_files[start_idx:end_idx], selected_ref_idx, selected_indices


def _select_images_by_indices(
    image_files: list, selected_indices: list[int], ref_idx: int
) -> tuple[list, int]:
    return [image_files[idx] for idx in selected_indices], selected_indices.index(
        ref_idx
    )


def run_focus_stack(cfg: DictConfig) -> None:
    dataset_dir = cfg.dataset_dir
    params = cfg.focus_stack_params
    focal_stack_size = _validate_focal_stack_size(params.focal_stack_size)
    sharpness_score = _validate_sharpness_score(cfg.sharpness_score)
    sharpness_selection_mode = _validate_sharpness_selection_mode(
        cfg.sharpness_selection_mode
    )
    sharpness_window_weights = _validate_sharpness_window_weights(
        cfg.sharpness_window_weights
    )
    sharpness_gaussian_sigma = _validate_sharpness_gaussian_sigma(
        cfg.sharpness_gaussian_sigma
    )
    find_sharpest_image = bool(cfg.find_sharpest_image)

    experiment_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    out_dir = cfg.output_paths.output_dir
    out_image_dir = os.path.join(out_dir, cfg.output_paths.images_subdir)
    out_depth_dir = os.path.join(out_dir, cfg.output_paths.depthmaps_subdir)
    out_mask_dir = os.path.join(out_dir, cfg.output_paths.masks_subdir)
    out_sharpness_dir = os.path.join(out_dir, cfg.output_paths.sharpness_scores_subdir)

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

    scan_inputs = []
    for subdir in scan_dirs:
        subdir_path = os.path.join(dataset_dir, subdir)
        # image_files = sorted(glob.glob(os.path.join(subdir_path, "*.png")))
        image_files = sorted(glob.glob(os.path.join(subdir_path, "*.jpg")))
        if not image_files:
            # print(f"No .png files found in {subdir_path}, skipping.")
            print(f"No .jpg files found in {subdir_path}, skipping.")
            continue

        sharpness_scores = None
        if find_sharpest_image:
            if sharpness_selection_mode == "window":
                (
                    ref_idx,
                    sharpness_scores,
                    selected_indices,
                    _,
                ) = find_sharpest_window_reference_index(
                    image_files,
                    sharpness_score,
                    focal_stack_size,
                    window_weights=sharpness_window_weights,
                    gaussian_sigma=sharpness_gaussian_sigma,
                    scan_name=subdir,
                )
                selected_image_files, selected_ref_idx = _select_images_by_indices(
                    image_files, selected_indices, ref_idx
                )
            else:
                ref_idx, sharpness_scores = find_sharpest_image_index(
                    image_files, sharpness_score, scan_name=subdir
                )
                (
                    selected_image_files,
                    selected_ref_idx,
                    selected_indices,
                ) = _select_focus_stack_images(
                    image_files, ref_idx, focal_stack_size, scan_name=subdir
                )
        else:
            ref_idx = (
                len(image_files) // 2
            )  # picks the middle one TODO: take a look at this later
            (
                selected_image_files,
                selected_ref_idx,
                selected_indices,
            ) = _select_focus_stack_images(
                image_files, ref_idx, focal_stack_size, scan_name=subdir
            )

        scan_inputs.append(
            (
                subdir,
                subdir_path,
                image_files,
                ref_idx,
                selected_image_files,
                selected_ref_idx,
                selected_indices,
                sharpness_scores,
            )
        )

    for path in (out_image_dir, out_depth_dir, out_mask_dir, out_sharpness_dir):
        if os.path.isdir(path):
            shutil.rmtree(path)
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(out_sharpness_dir, exist_ok=True)

    flags = []
    if params.full_resolution_align:
        flags.append("--full-resolution-align")
    if params.global_align:
        flags.append("--global-align")
    if params.align_keep_size:
        flags.append("--align-keep-size")
    if params.no_contrast:
        flags.append("--no-contrast")

    poses = {}

    for (
        subdir,
        subdir_path,
        image_files,
        ref_idx,
        selected_image_files,
        selected_ref_idx,
        selected_indices,
        sharpness_scores,
    ) in scan_inputs:
        if sharpness_scores is not None:
            save_sharpness_bar_chart(
                sharpness_scores,
                os.path.join(out_sharpness_dir, f"{subdir}.png"),
                reference_idx=ref_idx,
                selected_indices=selected_indices,
                title=(
                    f"{subdir} sharpness scores "
                    f"({sharpness_score}, {sharpness_selection_mode}, "
                    f"{sharpness_window_weights})"
                ),
            )

        output_file = os.path.join(out_image_dir, f"{subdir}.png")
        depth_file = os.path.join(out_depth_dir, f"{subdir}.png")
        mask_file = os.path.join(out_mask_dir, f"{subdir}.png")
        cmd = (
            ["focus-stack"]
            + flags
            + [
                f"--reference={selected_ref_idx}",
                f"--validmask={mask_file}",
                f"--depthmap={depth_file}",
                f"--output={output_file}",
            ]
            + selected_image_files
        )
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        pose_matrix = _load_pose_for_scan(subdir_path, image_files, ref_idx)
        poses[f"{subdir}.png"] = pose_matrix.tolist()

    poses_path = os.path.join(experiment_dir, cfg.output_paths.poses_filename)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(poses_path, "w") as f:
        json.dump(poses, f, indent=2)
    print(f"Wrote poses to {poses_path}")

    # Copy intrinsics.json from dataset directory to output directory
    src_intrinsics = os.path.join(dataset_dir, cfg.output_paths.intrinsics_filename)
    dst_intrinsics = os.path.join(experiment_dir, cfg.output_paths.intrinsics_filename)
    if os.path.exists(src_intrinsics):
        shutil.copy2(src_intrinsics, dst_intrinsics)
        print(f"Copied intrinsics to {dst_intrinsics}")
    else:
        print(f"Warning: intrinsics file not found at {src_intrinsics}, skipping.")


@hydra.main(
    config_path="../configs/focus_stack", config_name="default", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    run_focus_stack(cfg)


if __name__ == "__main__":
    main()
