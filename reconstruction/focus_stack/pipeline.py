import glob
import json
import os
import re
import shutil
import subprocess

import cv2
import hydra
import numpy as np

from focus_stack.sharpness_scoring import (
    VALID_SHARPNESS_METHODS,
    VALID_SHARPNESS_SELECTION_MODES,
    compute_sharpness_scores,
    find_gaussian_fit_window_indices,
    find_sharpest_image_index,
    find_sharpest_window_indices,
    save_sharpness_bar_chart,
    smooth_scores,
)
from omegaconf import DictConfig


def _natural_sort_key(path: str) -> list:
    """Sort strings with embedded numbers in numeric order."""
    text = os.path.basename(path)
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def _load_pose_for_scan(
    subdir_path: str, image_files: list, ref_idx: int
) -> np.ndarray:
    """Load the pose matrix for a scan directory.

    Checks in order:
    1. pose_NNNNN.npy files whose numeric suffixes match frame_NNNNN image files
       (same count and same numbers) → loads the pose whose number matches the
       reference frame at ref_idx.
    2. A singular pose.npy file → loads it directly.
    3. Neither → raises FileNotFoundError.
    """
    # Collect numeric suffixes from input frame files.
    frame_numbers = []
    for f in image_files:
        m = re.search(r"frame_(\d+)\.png$", os.path.basename(f))
        # m = re.search(r"frame_(\d+)\.jpg$", os.path.basename(f))

        if m:
            frame_numbers.append(m.group(1))

    # Collect indexed pose_*.npy files.
    pose_indexed = sorted(
        glob.glob(os.path.join(subdir_path, "pose_*.npy")), key=_natural_sort_key
    )
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


def _validate_sharpness_selection_mode(sharpness_selection_mode: str | None) -> str | None:
    if sharpness_selection_mode is None:
        return None
    if sharpness_selection_mode not in VALID_SHARPNESS_SELECTION_MODES:
        raise ValueError(
            "sharpness_selection_mode must be one of "
            f"{VALID_SHARPNESS_SELECTION_MODES} or null, got {sharpness_selection_mode!r}."
        )
    return sharpness_selection_mode


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


def _copy_gt_reference_image(
    gt_data_dir: str, subdir: str, image_files: list, ref_idx: int, out_gt_dir: str
) -> None:
    """Copy the ground-truth image matching the reference frame into out_gt_dir."""
    ref_filename = os.path.basename(image_files[ref_idx])
    gt_image_path = os.path.join(gt_data_dir, subdir, ref_filename)
    if not os.path.exists(gt_image_path):
        print(f"Warning: ground-truth image not found at {gt_image_path}, skipping.")
        return
    shutil.copy2(gt_image_path, os.path.join(out_gt_dir, f"{subdir}.png"))


def run_focus_stack(cfg: DictConfig) -> None:
    dataset_dir = cfg.dataset_dir
    gt_data_dir = cfg.get("gt_data_dir", None)
    params = cfg.focus_stack_params
    focal_stack_size = _validate_focal_stack_size(params.focal_stack_size)
    sharpness_score = _validate_sharpness_score(cfg.sharpness_score)
    sharpness_selection_mode = _validate_sharpness_selection_mode(
        cfg.sharpness_selection_mode
    )

    experiment_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    out_dir = cfg.output_paths.output_dir
    out_image_dir = os.path.join(out_dir, cfg.output_paths.images_subdir)
    out_depth_dir = os.path.join(out_dir, cfg.output_paths.depthmaps_subdir)
    out_mask_dir = os.path.join(out_dir, cfg.output_paths.masks_subdir)
    out_sharpness_dir = os.path.join(out_dir, cfg.output_paths.sharpness_scores_subdir)
    out_preprocessed_dir = os.path.join(out_dir, cfg.output_paths.preprocessed_images_subdir)
    out_gt_dir = os.path.join(out_dir, cfg.output_paths.gt_images_subdir)

    if cfg.scan_dirs is not None:
        scan_dirs = sorted(
            (d for d in cfg.scan_dirs if os.path.isdir(os.path.join(dataset_dir, d))),
            key=_natural_sort_key,
        )
    else:
        scan_dirs = sorted(
            (
                d
                for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d))
                and d.startswith(cfg.scan_prefix)
            ),
            key=_natural_sort_key,
        )

    for path in (out_image_dir, out_depth_dir, out_mask_dir, out_sharpness_dir, out_preprocessed_dir):
        if os.path.isdir(path):
            shutil.rmtree(path)
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(out_sharpness_dir, exist_ok=True)
    os.makedirs(out_preprocessed_dir, exist_ok=True)
    if gt_data_dir is not None:
        if os.path.isdir(out_gt_dir):
            shutil.rmtree(out_gt_dir)
        os.makedirs(out_gt_dir, exist_ok=True)

    preprocess_median = bool(cfg.get("median_blur_input", False))
    preprocess_lowpass = bool(cfg.get("low_pass_input", False))
    smooth_scores_enabled = bool(cfg.get("median_filter_scores", False))
    smooth_scores_kernel = int(cfg.get("median_filter_scores_kernel_size", 3))
    gaussian_fit_n_sigma = float(params.get("gaussian_fit_n_sigma", 1.0))

    scan_inputs = []
    for subdir in scan_dirs:
        subdir_path = os.path.join(dataset_dir, subdir)
        image_files = sorted(
            glob.glob(os.path.join(subdir_path, "*.png")), key=_natural_sort_key
        )
        # image_files = sorted(glob.glob(os.path.join(subdir_path, "*.jpg")))
        if not image_files:
            print(f"No .png files found in {subdir_path}, skipping.")
            # print(f"No .jpg files found in {subdir_path}, skipping.")
            continue

        scores_original = compute_sharpness_scores(image_files, sharpness_score)
        
        scores_for_selection = None
        if preprocess_median or preprocess_lowpass:
            scan_preprocess_dir = os.path.join(out_preprocessed_dir, subdir)
            os.makedirs(scan_preprocess_dir, exist_ok=True)
            input_files = []
            for img_path in image_files:
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if preprocess_median:
                        k = int(cfg.get("median_blur_kernel_size", 5))
                        img = cv2.medianBlur(img, k)
                    if preprocess_lowpass:
                        k = int(cfg.get("low_pass_input_kernel_size", 5))
                        img = cv2.GaussianBlur(img, (k, k), 0)
                    out_path = os.path.join(scan_preprocess_dir, os.path.basename(img_path))
                    cv2.imwrite(out_path, img)
                    input_files.append(out_path)
                else:
                    input_files.append(img_path)
            if preprocess_median:
                print(f"  Applied median blur (kernel={int(cfg.get('median_blur_kernel_size', 5))}) to {len(input_files)} input images.")
            if preprocess_lowpass:
                print(f"  Applied Gaussian blur (kernel={int(cfg.get('low_pass_input_kernel_size', 5))}) to {len(input_files)} input images.")
            scores_for_selection = compute_sharpness_scores(input_files, sharpness_score)
            
        else:
            input_files = image_files
            scores_for_selection = scores_original

        if smooth_scores_enabled:
            scores_for_selection = smooth_scores(scores_for_selection, window=smooth_scores_kernel)
        
        if sharpness_selection_mode is None:
            selected_image_files = input_files
            selected_ref_idx = len(input_files) // 2
            ref_idx = selected_ref_idx
            selected_indices = list(range(len(input_files)))
        elif sharpness_selection_mode == "fixed_window":
            ref_idx, selected_indices, _ = find_sharpest_window_indices(
                scores_for_selection, focal_stack_size, scan_name=subdir
            )
            selected_image_files, selected_ref_idx = _select_images_by_indices(
                input_files, selected_indices, ref_idx
            )
        elif sharpness_selection_mode == "gaussian_fit_window":
            ref_idx, selected_indices, _ = find_gaussian_fit_window_indices(
                scores_for_selection,
                n_sigma=gaussian_fit_n_sigma,
                max_frames=focal_stack_size if focal_stack_size is not None else 20,
                scan_name=subdir,
            )
            selected_image_files, selected_ref_idx = _select_images_by_indices(
                input_files, selected_indices, ref_idx
            )
        elif sharpness_selection_mode == "image":
            ref_idx = find_sharpest_image_index(scores_for_selection, scan_name=subdir)
            (
                selected_image_files,
                selected_ref_idx,
                selected_indices,
            ) = _select_focus_stack_images(
                input_files, ref_idx, focal_stack_size, scan_name=subdir
            )
        else:
            raise ValueError(
                f"Unexpected sharpness_selection_mode: {sharpness_selection_mode!r}"
            )

        save_sharpness_bar_chart(
            scores_original,
            os.path.join(out_sharpness_dir, f"{subdir}.png"),
            reference_idx=ref_idx,
            selected_indices=selected_indices,
            title=(
                f"{subdir} sharpness scores "
                f"({sharpness_score}, {sharpness_selection_mode})"
            ),
            scores_preprocessed=scores_for_selection if (preprocess_median or preprocess_lowpass or smooth_scores_enabled) else None,
        )

        scan_inputs.append(
            (
                subdir,
                subdir_path,
                image_files,
                ref_idx,
                selected_image_files,
                selected_ref_idx,
            )
        )

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
    ) in scan_inputs:
        output_file = os.path.join(out_image_dir, f"{subdir}.png")
        depth_file = os.path.join(out_depth_dir, f"{subdir}.png")
        mask_file = os.path.join(out_mask_dir, f"{subdir}.png")

        if len(selected_image_files) == 1:
            print(f"Skipping focus-stack for {subdir} (single image), copying directly.")
            shutil.copy2(selected_image_files[0], output_file)
        else:
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

        if cfg.get("low_pass_output") and os.path.exists(output_file):
            k = int(cfg.get("low_pass_kernel_size", 5))
            img = cv2.imread(output_file, cv2.IMREAD_UNCHANGED)
            if img is not None:
                img = cv2.GaussianBlur(img, (k, k), 0)
                cv2.imwrite(output_file, img)
                print(f"  Applied Gaussian blur (kernel={k}) to {output_file}")

        pose_matrix = _load_pose_for_scan(subdir_path, image_files, ref_idx)
        poses[f"{subdir}.png"] = pose_matrix.tolist()

        if gt_data_dir is not None:
            _copy_gt_reference_image(gt_data_dir, subdir, image_files, ref_idx, out_gt_dir)

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
