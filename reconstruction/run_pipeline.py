import hydra

from focus_stack import run_focus_stack
from omegaconf import DictConfig
from process_scans import run_colmap_pipeline
from simple_trainer import run_gsplat


@hydra.main(config_path="configs", config_name="pipeline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.run.focus_stack:
        run_focus_stack(cfg.focus_stack)

    if cfg.run.colmap_feat_extract_match or cfg.run.colmap_reconstruct:
        run_colmap_pipeline(cfg.colmap)

    if cfg.run.gsplat:
        run_gsplat(cfg.gsplat)


if __name__ == "__main__":
    main()
