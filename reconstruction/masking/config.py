from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class Sam2MaskingConfig:
    images_dir: str
    valid_regions_dir: str
    output_dir: str
    random_seed: int = 0
    masks_subdir: str = "masks"

    min_contour_area: float = 100.0
    morph_kernel_size: int = 5
    bootstrap_stem: str = "scan00"
    bbox_padding: int = 20

    sam2_model_id: str = "facebook/sam2.1-hiera-large"
    device: Optional[str] = "cuda"
    offload_video_to_cpu: bool = True

    overlay_alpha: float = 0.4
    overlay_color: List[int] = field(default_factory=lambda: [255, 0, 0])


def build_config(cfg_raw: DictConfig) -> Sam2MaskingConfig:
    """Convert nested Hydra config into the flat config used by the pipeline."""
    cfg = OmegaConf.to_container(cfg_raw, resolve=True)
    input_paths = cfg["input_paths"]
    output_paths = cfg["output_paths"]
    bootstrap = cfg["bootstrap"]
    sam2 = cfg["sam2"]
    visualization = cfg["visualization"]

    return Sam2MaskingConfig(
        images_dir=input_paths["images_dir"],
        valid_regions_dir=input_paths["valid_regions_dir"],
        output_dir=output_paths["output_dir"],
        random_seed=cfg["random_seed"],
        masks_subdir=output_paths["masks_subdir"],
        min_contour_area=bootstrap["min_contour_area"],
        morph_kernel_size=bootstrap["morph_kernel_size"],
        bootstrap_stem=bootstrap["bootstrap_stem"],
        bbox_padding=bootstrap["bbox_padding"],
        sam2_model_id=sam2["model_id"],
        device=sam2["device"],
        offload_video_to_cpu=sam2["offload_video_to_cpu"],
        overlay_alpha=visualization["overlay_alpha"],
        overlay_color=list(visualization["overlay_color"]),
    )
