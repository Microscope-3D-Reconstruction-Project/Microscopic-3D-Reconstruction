"""trainer_2dgs.py — Hydra entry point for 2D Gaussian Splatting training.
"""

import importlib
import inspect
import time

import hydra
import torch

from gs_2d import Config, Runner
from gsplat.distributed import cli
from omegaconf import DictConfig, OmegaConf


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config) -> None:
    """Distributed entry point called by gsplat's ``cli`` helper."""
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # Eval-only: load checkpoint(s) and run evaluation
        ckpts = [
            torch.load(f, map_location=runner.device, weights_only=True)
            for f in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.evaluator.eval(step=step)
        runner.evaluator.render_traj(step=step)
        if cfg.compression is not None:
            runner.evaluator.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1_000_000)


def _instantiate_strategy(strategy_cfg: DictConfig):
    raw = OmegaConf.to_container(strategy_cfg, resolve=True)
    target = raw.pop("_target_")
    module_name, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    valid = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return cls(**{k: v for k, v in raw.items() if k in valid})


def _build_config(cfg_raw: DictConfig) -> Config:
    # strategy = _instantiate_strategy(cfg_raw.strategy)
    cfg_dict = OmegaConf.to_container(cfg_raw, resolve=True)
    # cfg_dict.pop("strategy")
    # Flatten nested path sections into the top-level namespace.
    cfg_dict.update(cfg_dict.pop("input_paths", {}))
    cfg_dict.update(cfg_dict.pop("output_paths", {}))
    # Drop interpolation-only or Hydra-internal keys not present in Config.
    for key in ("output_dir", "experiment_name", "hydra"):
        cfg_dict.pop(key, None)
    cfg = Config(**cfg_dict)
    cfg.adjust_steps(cfg.steps_scaler)
    return cfg


def run_gs_2d(cfg_raw: DictConfig) -> None:
    """Entry point for the 2D Gaussian Splatting stage; callable from ``run_pipeline.py``."""
    cfg = _build_config(cfg_raw)
    cli(main, cfg, verbose=True)


@hydra.main(config_path="../configs/gs_2d", config_name="default", version_base="1.3")
def train(cfg_raw: DictConfig) -> None:
    """Hydra entry point: parse config, validate deps, launch distributed run."""
    cfg = _build_config(cfg_raw)
    cli(main, cfg, verbose=True)


if __name__ == "__main__":
    train()
