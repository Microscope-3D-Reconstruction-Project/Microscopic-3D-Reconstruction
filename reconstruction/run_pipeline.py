import hydra

from omegaconf import DictConfig
from process_scans import run_colmap_pipeline


@hydra.main(config_path="configs", config_name="pipeline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.run.colmap_feat_extract_match or cfg.run.colmap_reconstruct:
        run_colmap_pipeline(cfg.colmap)

    if cfg.run.gsplat:
        print("gsplat stage not yet integrated into pipeline.")


if __name__ == "__main__":
    main()
