import hydra

from omegaconf import DictConfig
from sam3_masking.config import build_config
from sam3_masking.pipeline import Sam3MaskingPipeline


def run_sam3_masking(cfg_raw: DictConfig) -> None:
    cfg = build_config(cfg_raw)
    Sam3MaskingPipeline(cfg).run()


@hydra.main(
    config_path="../configs/sam3_masking", config_name="default", version_base="1.3"
)
def main(cfg_raw: DictConfig) -> None:
    run_sam3_masking(cfg_raw)


if __name__ == "__main__":
    main()
