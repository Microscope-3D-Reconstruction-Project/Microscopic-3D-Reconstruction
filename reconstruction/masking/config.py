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
    gt_images_dir: Optional[str] = None  # optional; masked GT images written only if this dir exists
    masked_gt_images_subdir: str = "masked_gt_images"

    bootstrap_stem: str = "scan00"
    focus_mask_border_padding: int = 8
    edge_blur_kernel_size: int = 5
    canny_threshold1: float = 25.0
    canny_threshold2: float = 150.0
    point_sampling_scale: float = 0.9
    point_prompt_count: int = 32
    firstpass_shrink_scale: float = 0.9
    points_refine_kernel_size: int = 9
    points_refine_dilate_iterations: int = 1

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
        gt_images_dir=input_paths.get("gt_images_dir"),
        masked_gt_images_subdir=output_paths.get("masked_gt_images_subdir", "masked_gt_images"),
        bootstrap_stem=bootstrap["bootstrap_stem"],
        focus_mask_border_padding=bootstrap["focus_mask_border_padding"],
        edge_blur_kernel_size=bootstrap["edge_blur_kernel_size"],
        canny_threshold1=bootstrap["canny_threshold1"],
        canny_threshold2=bootstrap["canny_threshold2"],
        point_sampling_scale=bootstrap["point_sampling_scale"],
        point_prompt_count=bootstrap["point_prompt_count"],
        firstpass_shrink_scale=bootstrap["firstpass_shrink_scale"],
        points_refine_kernel_size=bootstrap["points_refine_kernel_size"],
        points_refine_dilate_iterations=bootstrap["points_refine_dilate_iterations"],
        sam2_model_id=sam2["model_id"],
        device=sam2["device"],
        offload_video_to_cpu=sam2["offload_video_to_cpu"],
        overlay_alpha=visualization["overlay_alpha"],
        overlay_color=list(visualization["overlay_color"]),
    )
