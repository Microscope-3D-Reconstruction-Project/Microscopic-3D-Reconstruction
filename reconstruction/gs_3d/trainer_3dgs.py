"""trainer_3dgs.py — Hydra entry point for 3D Gaussian Splatting training.

All training logic lives in the ``gs_3d/`` package:
  gs_3d/config.py  — Config dataclass
  gs_3d/model.py   — Gaussian splat initialisation
  gs_3d/runner.py  — Runner training / evaluation engine
  gs_3d/utils.py   — Optimization modules and math helpers

Configuration is loaded from configs/gs_3d/default.yaml (Hydra).

Usage::

    # Default run (DefaultStrategy)
    CUDA_VISIBLE_DEVICES=0 python gs_3d/trainer_3dgs.py \
        input_paths.model_dir=outputs/my_scan/sparse_model \
        input_paths.images_dir=outputs/my_scan/images \
        output_paths.splat_dir=outputs/my_scan/gs_3d

    # MCMC preset
    python gs_3d/trainer_3dgs.py +experiment=mcmc \
        input_paths.model_dir=outputs/my_scan/sparse_model \
        input_paths.images_dir=outputs/my_scan/images

    # Distributed training on 4 GPUs (4x fewer steps)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python gs_3d/trainer_3dgs.py steps_scaler=0.25

    # Eval-only from a checkpoint
    python gs_3d/trainer_3dgs.py ckpt='["outputs/my_scan/gs_3d/ckpts/ckpt_29999_rank0.pt"]'

"""

import importlib
import inspect
import time

import hydra
import torch

from gs_3d import Config, Runner
from gsplat.distributed import cli
from omegaconf import DictConfig, OmegaConf


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config) -> None:
    """Distributed entry point called by gsplat's ``cli`` helper."""
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # Eval-only: load checkpoint(s) and run evaluation
        ckpts = [
            torch.load(f, map_location=runner.device, weights_only=True)
            for f in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1_000_000)


def _instantiate_strategy(strategy_cfg: DictConfig):
    """Instantiate the strategy, passing only kwargs accepted by its constructor.

    Hydra deep-merges plain dict keys from parent configs, so a strategy block
    may contain keys from the default variant (e.g. ``reset_every`` from
    DefaultStrategy) even after overriding with a different ``_target_``. This
    function filters to only valid constructor parameters before instantiating.
    """
    raw = OmegaConf.to_container(strategy_cfg, resolve=True)
    target = raw.pop("_target_")
    module_name, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    valid = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return cls(**{k: v for k, v in raw.items() if k in valid})


def _build_config(cfg_raw: DictConfig) -> Config:
    """Convert a (potentially nested) Hydra DictConfig to a flat ``Config``.

    Lifts ``input_paths.*`` and ``output_paths.*`` to top-level keys and drops
    the ``output_dir`` / ``experiment_name`` interpolation helpers that exist
    solely for Hydra variable substitution.
    """
    strategy = _instantiate_strategy(cfg_raw.strategy)
    cfg_dict = OmegaConf.to_container(cfg_raw, resolve=True)
    cfg_dict.pop("strategy")
    # Flatten nested path sections into the top-level namespace.
    cfg_dict.update(cfg_dict.pop("input_paths", {}))
    cfg_dict.update(cfg_dict.pop("output_paths", {}))
    # Drop interpolation-only or Hydra-internal keys not present in Config.
    for key in ("output_dir", "experiment_name", "hydra"):
        cfg_dict.pop(key, None)
    cfg_dict["bilateral_grid_shape"] = tuple(cfg_dict["bilateral_grid_shape"])
    cfg = Config(**cfg_dict, strategy=strategy)
    cfg.adjust_steps(cfg.steps_scaler)
    return cfg


def run_gs_3d(cfg_raw: DictConfig) -> None:
    """Entry point for the 3D Gaussian Splatting stage; callable from ``run_pipeline.py``."""
    cfg = _build_config(cfg_raw)
    _validate_deps(cfg)
    cli(main, cfg, verbose=True)


def _validate_deps(cfg: Config) -> None:
    """Raise early with a clear message if optional deps are missing."""
    if cfg.compression == "png":
        try:
            import plas  # noqa: F401
            import torchpq  # noqa: F401
        except ImportError:
            raise ImportError(
                "PNG compression requires torchpq and plas. Install via:\n"
                "  pip install torchpq  # see https://github.com/DeMoriarty/TorchPQ\n"
                "  pip install git+https://github.com/fraunhoferhhi/PLAS.git"
            )
    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d=true`."


@hydra.main(config_path="../configs/gs_3d", config_name="default", version_base="1.3")
def train(cfg_raw: DictConfig) -> None:
    """Hydra entry point: parse config, validate deps, launch distributed run."""
    cfg = _build_config(cfg_raw)
    _validate_deps(cfg)

    # Validate optional dependencies upfront for a clear error message
    cli(main, cfg, verbose=True)


if __name__ == "__main__":
    train()
