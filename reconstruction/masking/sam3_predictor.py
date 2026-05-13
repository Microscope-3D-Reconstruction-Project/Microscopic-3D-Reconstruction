import torch

from .io_utils import save_binary_mask, save_sam3_candidate_masks
from .mask_utils import (
    bbox_xyxy_to_normalized_cxcywh,
    create_concatenated_pair,
    pick_right_side_mask,
)


def run_paired_sam_prediction(
    processor,
    anchor_image,
    anchor_bbox,
    target_image,
    prompt,
    device,
    min_area,
    sam3_candidate_masks_dir=None,
    sam3_selected_masks_dir=None,
    sam3_debug_base_name=None,
):
    """Run SAM3 on an anchor|target image pair and return target mask/bbox."""
    paired_image, right_offset = create_concatenated_pair(anchor_image, target_image)
    prompt_bbox = bbox_xyxy_to_normalized_cxcywh(anchor_bbox, paired_image.size)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=str(device).startswith("cuda"),
    ):
        state = processor.set_image(paired_image)
        if prompt:
            state = processor.set_text_prompt(prompt=prompt, state=state)
        outputs = processor.add_geometric_prompt(
            box=prompt_bbox,
            label=True,
            state=state,
        )

    if sam3_candidate_masks_dir is not None and sam3_debug_base_name is not None:
        save_sam3_candidate_masks(
            outputs=outputs,
            right_offset=right_offset,
            right_image_size=target_image.size,
            output_dir=sam3_candidate_masks_dir,
            base_name=sam3_debug_base_name,
        )

    mask, bbox = pick_right_side_mask(
        outputs=outputs,
        right_offset=right_offset,
        right_image_size=target_image.size,
        min_area=min_area,
    )
    if (
        mask is not None
        and mask.any()
        and sam3_selected_masks_dir is not None
        and sam3_debug_base_name is not None
    ):
        save_binary_mask(
            mask, sam3_selected_masks_dir, f"{sam3_debug_base_name}_selected"
        )
    return mask, bbox, paired_image, right_offset


class Sam3PairPredictor:
    """Own SAM3 model loading and paired anchor-target inference."""

    def __init__(self, cfg):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.cfg = cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading native SAM3 image model on {self.device}")
        model = build_sam3_image_model(
            checkpoint_path=cfg.checkpoint_path,
            bpe_path=cfg.bpe_path,
            device=self.device,
            compile=cfg.sam_compile,
        )
        self.processor = Sam3Processor(
            model,
            device=self.device,
            confidence_threshold=cfg.sam_output_prob_thresh,
        )

    def predict_pair(
        self,
        anchor_image,
        anchor_bbox,
        target_image,
        prompt,
        min_area,
        sam3_candidate_masks_dir=None,
        sam3_selected_masks_dir=None,
        sam3_debug_base_name=None,
    ):
        return run_paired_sam_prediction(
            processor=self.processor,
            anchor_image=anchor_image,
            anchor_bbox=anchor_bbox,
            target_image=target_image,
            prompt=prompt,
            device=self.device,
            min_area=min_area,
            sam3_candidate_masks_dir=sam3_candidate_masks_dir,
            sam3_selected_masks_dir=sam3_selected_masks_dir,
            sam3_debug_base_name=sam3_debug_base_name,
        )
