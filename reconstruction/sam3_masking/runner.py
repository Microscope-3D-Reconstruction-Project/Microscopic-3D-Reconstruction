import random

import hydra
import numpy as np
import torch

from omegaconf import DictConfig
from sam3_masking.config import build_config
from sam3_masking.pipeline import Sam3MaskingPipeline


def set_random_seed(seed: int) -> None:
    """Seed random sources used by Python, NumPy, and Torch-backed models."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_sam3_masking(cfg_raw: DictConfig) -> None:
    cfg = build_config(cfg_raw)
    set_random_seed(cfg.random_seed)
    Sam3MaskingPipeline(cfg).run()


@hydra.main(
    config_path="../configs/sam3_masking", config_name="default", version_base="1.3"
)
def main(cfg_raw: DictConfig) -> None:
    run_sam3_masking(cfg_raw)


if __name__ == "__main__":
    main()
