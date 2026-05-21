import os
import time

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra

from colmap.runner import run_colmap_pipeline
from focus_stack import run_focus_stack
from gs_3d.trainer_3dgs import run_gs_3d
from masking.runner import run_sam2_masking
from omegaconf import DictConfig


def _init_timing_log(cfg: DictConfig) -> Path:
    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "pipeline_timings.log"
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write(f"Pipeline timing log started: {datetime.now().isoformat()}\n")
            f.write(f"Experiment: {cfg.experiment_name}\n")
            f.write("stage\tstatus\tduration_seconds\n")
            for stage in (
                "focus_stack",
                "masking",
                "colmap",
                "gs_3d",
                "pipeline_total",
            ):
                f.write(f"{stage}\tnot_ran\t0.000\n")
    return log_path


def _write_timing(
    log_path: Path, stage_name: str, status: str, duration: float
) -> None:
    new_entry = f"{stage_name}\t{status}\t{duration:.3f}"
    lines = log_path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"{stage_name}\t"):
            lines[i] = new_entry
            log_path.write_text("\n".join(lines) + "\n")
            return
    with open(log_path, "a") as f:
        f.write(new_entry + "\n")


def _write_pipeline_total(log_path: Path) -> None:
    total = 0.0
    for line in log_path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0] != "stage" and parts[0] != "pipeline_total":
            try:
                total += float(parts[2])
            except ValueError:
                pass
    _write_timing(log_path, "pipeline_total", "success", total)
    print(f"pipeline_total {total:.2f} seconds")


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
    return cfg.run.colmap or cfg.run.gs_3d


def _validate_existing_masks(cfg: DictConfig) -> None:
    if cfg.run.masking or not _requires_masks(cfg):
        return

    masks_dir = os.path.join(
        cfg.masking.output_paths.output_dir,
        cfg.masking.output_paths.masks_subdir,
    )
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(
            "run.masking is false, but downstream stages are enabled and "
            f"expected existing masks at {masks_dir!r}. Run the masking "
            "stage first or enable run.masking."
        )


@hydra.main(config_path="configs", config_name="pipeline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    timing_log_path = _init_timing_log(cfg)

    if cfg.run.focus_stack:
        with _timed_stage(timing_log_path, "focus_stack"):
            run_focus_stack(cfg.focus_stack)

    if cfg.run.masking:
        with _timed_stage(timing_log_path, "masking"):
            run_sam2_masking(cfg.masking)
    else:
        _validate_existing_masks(cfg)

    if cfg.run.colmap:
        with _timed_stage(timing_log_path, "colmap"):
            run_colmap_pipeline(cfg.colmap)

    if cfg.run.gs_3d:
        with _timed_stage(timing_log_path, "gs_3d"):
            run_gs_3d(cfg.gs_3d)

    _write_pipeline_total(timing_log_path)


if __name__ == "__main__":
    main()
