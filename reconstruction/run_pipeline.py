import os

import hydra

from focus_stack import run_focus_stack
from gs_2d.trainer_2dgs import run_gs_2d
from gs_3d.trainer_3dgs import run_gs_3d
from omegaconf import DictConfig
from process_scans import run_colmap_pipeline
from sam3_masking.runner import run_sam3_masking


def _requires_masks(cfg: DictConfig) -> bool:
    return (
        cfg.run.colmap_feat_extract_match
        or cfg.run.colmap_reconstruct
        or cfg.run.splatting_method in ("gs_3d", "gs_2d")
    )


def _validate_existing_sam3_masks(cfg: DictConfig) -> None:
    if cfg.run.sam3_masking or not _requires_masks(cfg):
        return

    masks_dir = os.path.join(
        cfg.sam3_masking.output_paths.output_dir,
        cfg.sam3_masking.output_paths.masks_subdir,
    )
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(
            "run.sam3_masking is false, but downstream stages are enabled and "
            f"expected existing SAM3 masks at {masks_dir!r}. Run the SAM3 masking "
            "stage first or enable run.sam3_masking."
        )


@hydra.main(config_path="configs", config_name="pipeline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.run.focus_stack:
        run_focus_stack(cfg.focus_stack)

    if cfg.run.sam3_masking:
        run_sam3_masking(cfg.sam3_masking)
    else:
        _validate_existing_sam3_masks(cfg)

    if cfg.run.colmap_feat_extract_match or cfg.run.colmap_reconstruct:
        run_colmap_pipeline(cfg.colmap)

    if cfg.run.splatting_method == "gs_3d":
        run_gs_3d(cfg.gs_3d)
    elif cfg.run.splatting_method == "gs_2d":
        run_gs_2d(cfg.gs_2d)


if __name__ == "__main__":
    main()
