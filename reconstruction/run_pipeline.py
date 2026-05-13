import os
import time

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra

from focus_stack import run_focus_stack
from gs_2d.trainer_2dgs import run_gs_2d
from gs_3d.trainer_3dgs import run_gs_3d
from masking.runner import run_sam3_masking
from omegaconf import DictConfig
from process_scans import run_colmap_pipeline


def _init_timing_log(cfg: DictConfig) -> Path:
    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "pipeline_timings.log"
    with open(log_path, "w") as f:
        f.write(f"Pipeline timing log started: {datetime.now().isoformat()}\n")
        f.write(f"Experiment: {cfg.experiment_name}\n")
        f.write("stage\tstatus\tduration_seconds\n")
    return log_path


def _write_timing(
    log_path: Path, stage_name: str, status: str, duration: float
) -> None:
    with open(log_path, "a") as f:
        f.write(f"{stage_name}\t{status}\t{duration:.3f}\n")


@contextmanager
def _timed_stage(log_path: Path, stage_name: str):
    start_time = time.perf_counter()
    status = "success"
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        duration = time.perf_counter() - start_time
        _write_timing(log_path, stage_name, status, duration)
        print(f"{stage_name} {status} in {duration:.2f} seconds")


def _requires_masks(cfg: DictConfig) -> bool:
    return (
        cfg.run.colmap_feat_extract_match
        or cfg.run.colmap_reconstruct
        or cfg.run.splatting_method in ("gs_3d", "gs_2d")
    )


def _validate_existing_sam3_masks(cfg: DictConfig) -> None:
    if cfg.run.masking or not _requires_masks(cfg):
        return

    masks_dir = os.path.join(
        cfg.masking.output_paths.output_dir,
        cfg.masking.output_paths.masks_subdir,
    )
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(
            "run.masking is false, but downstream stages are enabled and "
            f"expected existing SAM3 masks at {masks_dir!r}. Run the masking "
            "stage first or enable run.masking."
        )


@hydra.main(config_path="configs", config_name="pipeline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    timing_log_path = _init_timing_log(cfg)

    with _timed_stage(timing_log_path, "pipeline_total"):
        if cfg.run.focus_stack:
            with _timed_stage(timing_log_path, "focus_stack"):
                run_focus_stack(cfg.focus_stack)

        if cfg.run.masking:
            with _timed_stage(timing_log_path, "masking"):
                run_sam3_masking(cfg.masking)
        else:
            _validate_existing_sam3_masks(cfg)

        if cfg.run.colmap_feat_extract_match or cfg.run.colmap_reconstruct:
            with _timed_stage(timing_log_path, "colmap"):
                run_colmap_pipeline(cfg.colmap)

        if cfg.run.splatting_method == "gs_3d":
            with _timed_stage(timing_log_path, "gs_3d"):
                run_gs_3d(cfg.gs_3d)
        elif cfg.run.splatting_method == "gs_2d":
            with _timed_stage(timing_log_path, "gs_2d"):
                run_gs_2d(cfg.gs_2d)


if __name__ == "__main__":
    main()
