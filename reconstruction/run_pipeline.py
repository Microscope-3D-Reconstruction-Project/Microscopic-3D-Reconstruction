import hydra

from focus_stack import run_focus_stack
from gs_2d.trainer_2dgs import run_gs_2d
from gs_3d.trainer_3dgs import run_gs_3d
from omegaconf import DictConfig
from process_scans import run_colmap_pipeline


@hydra.main(config_path="configs", config_name="pipeline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.run.focus_stack:
        run_focus_stack(cfg.focus_stack)

    if cfg.run.colmap_feat_extract_match or cfg.run.colmap_reconstruct:
        run_colmap_pipeline(cfg.colmap)

    if cfg.run.splatting_method == "gs_3d":
        run_gs_3d(cfg.gs_3d)
    elif cfg.run.splatting_method == "gs_2d":
        run_gs_2d(cfg.gs_2d)


if __name__ == "__main__":
    main()
