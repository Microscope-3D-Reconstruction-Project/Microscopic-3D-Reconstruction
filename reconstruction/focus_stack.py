import glob
import os
import subprocess

import hydra

from omegaconf import DictConfig


def run_focus_stack(cfg: DictConfig) -> None:
    dataset_dir = cfg.dataset_dir
    out_dir = cfg.output_paths.output_dir
    out_image_dir = os.path.join(out_dir, cfg.output_paths.images_subdir)
    out_depth_dir = os.path.join(out_dir, cfg.output_paths.depthmaps_subdir)
    out_mask_dir = os.path.join(out_dir, cfg.output_paths.masks_subdir)
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    scan_dirs = sorted(
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith(cfg.scan_prefix)
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

    for subdir in scan_dirs:
        subdir_path = os.path.join(dataset_dir, subdir)
        image_files = sorted(glob.glob(os.path.join(subdir_path, "*.jpg")))
        if not image_files:
            print(f"No .jpg files found in {subdir_path}, skipping.")
            continue
        output_file = os.path.join(out_image_dir, f"{subdir}.jpg")
        depth_file = os.path.join(out_depth_dir, f"{subdir}.png")
        mask_file = os.path.join(out_mask_dir, f"{subdir}.png")
        cmd = (
            ["focus-stack"]
            + flags
            + [
                f"--validmask={mask_file}",
                f"--depthmap={depth_file}",
                f"--output={output_file}",
            ]
            + image_files
        )
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


@hydra.main(
    config_path="configs/focus_stack", config_name="default", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    run_focus_stack(cfg)


if __name__ == "__main__":
    main()
