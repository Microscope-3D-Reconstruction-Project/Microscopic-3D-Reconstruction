from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class Sam3MaskingConfig:
    input_dir: str
    output_dir: str
    masks_subdir: str = "masks"

    black_threshold: int = 20
    min_contour_area: float = 100.0
    morph_kernel_size: int = 5
    bootstrap_stem: str = "scan00"
    bbox_padding: int = 20

    checkpoint_path: Optional[str] = None
    bpe_path: Optional[str] = None
    sam_compile: bool = False
    prompt: Optional[str] = None
    sam_output_prob_thresh: float = 0.5
    device: Optional[str] = None

    overlay_alpha: float = 0.4
    overlay_color: List[int] = field(default_factory=lambda: [255, 0, 0])

    appearance_filter: bool = True
    rerun_outliers: bool = True
    appearance_top_k: int = 5
    appearance_tau: float = 3.0
    appearance_min_mutual_neighbors: int = 1

    min_mask_area_frac: float = 1e-5
    max_bbox_area_frac: float = 0.8
    min_mask_fill_ratio: float = 0.02
    max_mask_components: int = 3

    rerun_anchor_top_k: int = 3
    rerun_anchor_strategy: str = "hybrid"
    anchor_index_weight: float = 0.15

    dinov3_model: str = "vit_small_patch16_dinov3"
    dinov3_pretrained: bool = True
    dinov3_image_size: int = 512
    dinov3_device: Optional[str] = None
    dinov3_patch_similarity_threshold: float = 0.45
    dinov3_max_bad_patch_fraction: float = 0.2
    dinov3_min_patch_mean_score: float = -1.0
    dinov3_max_patch_bank: int = 20000


def build_config(cfg_raw: DictConfig) -> Sam3MaskingConfig:
    """Convert nested Hydra config into the flat config used by the pipeline."""
    cfg = OmegaConf.to_container(cfg_raw, resolve=True)
    input_paths = cfg["input_paths"]
    output_paths = cfg["output_paths"]
    bootstrap = cfg["bootstrap"]
    sam3 = cfg["sam3"]
    visualization = cfg["visualization"]
    appearance = cfg["appearance"]
    mask_sanity = cfg["mask_sanity"]
    rerun = cfg["rerun"]
    dinov3 = cfg["dinov3"]

    input_dir = f"{input_paths['focus_stack_dir']}/{input_paths['images_subdir']}"
    output_dir = output_paths["output_dir"]

    config = Sam3MaskingConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        masks_subdir=output_paths["masks_subdir"],
        black_threshold=bootstrap["black_threshold"],
        min_contour_area=bootstrap["min_contour_area"],
        morph_kernel_size=bootstrap["morph_kernel_size"],
        bootstrap_stem=bootstrap["bootstrap_stem"],
        bbox_padding=bootstrap["bbox_padding"],
        checkpoint_path=sam3["checkpoint_path"],
        bpe_path=sam3["bpe_path"],
        sam_compile=sam3["compile"],
        prompt=sam3["prompt"],
        sam_output_prob_thresh=sam3["output_prob_thresh"],
        device=sam3["device"],
        overlay_alpha=visualization["overlay_alpha"],
        overlay_color=list(visualization["overlay_color"]),
        appearance_filter=appearance["enabled"],
        rerun_outliers=rerun["enabled"],
        appearance_top_k=appearance["top_k"],
        appearance_tau=appearance["tau"],
        appearance_min_mutual_neighbors=appearance["min_mutual_neighbors"],
        min_mask_area_frac=mask_sanity["min_mask_area_frac"],
        max_bbox_area_frac=mask_sanity["max_bbox_area_frac"],
        min_mask_fill_ratio=mask_sanity["min_mask_fill_ratio"],
        max_mask_components=mask_sanity["max_mask_components"],
        rerun_anchor_top_k=rerun["anchor_top_k"],
        rerun_anchor_strategy=rerun["anchor_strategy"],
        anchor_index_weight=rerun["anchor_index_weight"],
        dinov3_model=dinov3["model"],
        dinov3_pretrained=dinov3["pretrained"],
        dinov3_image_size=dinov3["image_size"],
        dinov3_device=dinov3["device"],
        dinov3_patch_similarity_threshold=dinov3["patch_similarity_threshold"],
        dinov3_max_bad_patch_fraction=dinov3["max_bad_patch_fraction"],
        dinov3_min_patch_mean_score=dinov3["min_patch_mean_score"],
        dinov3_max_patch_bank=dinov3["max_patch_bank"],
    )
    if config.rerun_outliers:
        config.appearance_filter = True
    return config
