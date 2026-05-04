import os

import cv2
import numpy as np
import torch

from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_foreground_mask(
    image_rgb,
    black_threshold,
    min_contour_area,
    morph_kernel_size,
    keep_largest,
):
    """Segment non-black foreground with thresholding and connected components."""
    image_np = np.array(image_rgb)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    value_mask = hsv[..., 2] > black_threshold
    color_mask = np.max(image_np, axis=2) > black_threshold
    threshold_mask = np.logical_or(value_mask, color_mask).astype(np.uint8) * 255

    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel)
        threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        threshold_mask, connectivity=8
    )

    if num_labels <= 1:
        return np.zeros(threshold_mask.shape, dtype=bool), 0

    component_ids = [
        label_id
        for label_id in range(1, num_labels)
        if stats[label_id, cv2.CC_STAT_AREA] >= min_contour_area
    ]

    if keep_largest and component_ids:
        component_ids = [
            max(component_ids, key=lambda label_id: stats[label_id, cv2.CC_STAT_AREA])
        ]

    component_mask = np.isin(labels, component_ids)
    if not component_mask.any():
        return np.zeros(threshold_mask.shape, dtype=bool), 0

    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        component_mask = cv2.morphologyEx(
            component_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel
        )
        component_mask = component_mask > 0

    return component_mask, len(component_ids)


def get_mask_bbox(mask, padding=0):
    """Return an xyxy bounding box around nonzero mask pixels."""
    y_indices, x_indices = np.nonzero(mask)
    if len(x_indices) == 0:
        return None

    height, width = mask.shape
    x_min = max(0, int(x_indices.min()) - padding)
    y_min = max(0, int(y_indices.min()) - padding)
    x_max = min(width - 1, int(x_indices.max()) + padding)
    y_max = min(height - 1, int(y_indices.max()) + padding)
    return [x_min, y_min, x_max, y_max]


def bbox_area(bbox):
    """Return inclusive xyxy bbox area in pixels."""
    if bbox is None:
        return 0
    x_min, y_min, x_max, y_max = bbox
    return max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)


def keep_largest_component(mask, min_area=0):
    """Keep only the largest connected foreground component in a binary mask."""
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    if num_labels <= 1:
        return np.zeros(mask.shape, dtype=bool)

    component_ids = [
        label_id
        for label_id in range(1, num_labels)
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area
    ]
    if not component_ids:
        return np.zeros(mask.shape, dtype=bool)

    largest_id = max(
        component_ids, key=lambda label_id: stats[label_id, cv2.CC_STAT_AREA]
    )
    return labels == largest_id


def count_components(mask, min_area=0):
    """Count connected foreground components at least min_area pixels large."""
    mask_u8 = mask.astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return 0
    return int(
        sum(
            stats[label_id, cv2.CC_STAT_AREA] >= min_area
            for label_id in range(1, num_labels)
        )
    )


def list_input_images(input_dir):
    """Return sorted image paths from a flat directory."""
    image_paths = []
    for filename in os.listdir(input_dir):
        path = os.path.join(input_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        if os.path.isfile(path) and ext in VALID_EXTENSIONS:
            image_paths.append(path)
    try:
        return sorted(
            image_paths,
            key=lambda path: int(os.path.splitext(os.path.basename(path))[0]),
        )
    except ValueError:
        return sorted(image_paths)


def find_bootstrap_image(image_paths, bootstrap_stem):
    """Find the scan00 image, falling back to the first sorted image."""
    for image_path in image_paths:
        if os.path.splitext(os.path.basename(image_path))[0] == bootstrap_stem:
            return image_path
    return image_paths[0] if image_paths else None


def bbox_xyxy_to_normalized_cxcywh(bbox, image_size):
    """Convert pixel xyxy bbox to normalized cxcywh for SAM3 image prompts."""
    width, height = image_size
    x_min, y_min, x_max, y_max = bbox
    bbox_width = max(1, x_max - x_min + 1)
    bbox_height = max(1, y_max - y_min + 1)
    return [
        (x_min + bbox_width / 2.0) / width,
        (y_min + bbox_height / 2.0) / height,
        bbox_width / width,
        bbox_height / height,
    ]


def create_concatenated_pair(left_image, right_image, background_color=(0, 0, 0)):
    """Create a single RGB canvas with scan00 on the left and scanXX on the right."""
    paired_width = left_image.width + right_image.width
    paired_height = max(left_image.height, right_image.height)
    paired_image = Image.new("RGB", (paired_width, paired_height), background_color)
    paired_image.paste(left_image, (0, 0))
    paired_image.paste(right_image, (left_image.width, 0))
    return paired_image, left_image.width


def output_masks_to_numpy(masks):
    """Normalize SAM3 output masks to a boolean numpy array shaped NxHxW."""
    if masks is None:
        return np.zeros((0, 0, 0), dtype=bool)
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None, ...]
    return masks.astype(bool)


def pick_right_side_mask(outputs, right_offset, right_image_size, min_area):
    """Pick the SAM3 mask with the strongest foreground on the right-side image."""
    masks = output_masks_to_numpy(outputs.get("masks"))
    if len(masks) == 0:
        return None, None

    right_width, right_height = right_image_size
    best_mask = None
    best_bbox = None
    best_area = 0

    for mask in masks:
        right_crop = mask[:right_height, right_offset : right_offset + right_width]
        if not right_crop.any():
            continue

        right_crop = keep_largest_component(right_crop.astype(bool), min_area=min_area)
        area = int(right_crop.sum())
        if area <= best_area:
            continue

        bbox = get_mask_bbox(right_crop)
        if bbox is None:
            continue

        best_mask = right_crop
        best_bbox = bbox
        best_area = area

    return best_mask, best_bbox


def l2_normalize(vector, eps=1e-12):
    """Return a float32 vector normalized to unit L2 length."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return None
    return vector / norm


def mask_quality_stats(mask, bbox, image_size, min_component_area):
    """Compute loose sanity metrics for a predicted binary mask."""
    width, height = image_size
    image_area = max(1, width * height)
    mask_area = int(mask.sum()) if mask is not None else 0
    box_area = bbox_area(bbox)
    fill_ratio = float(mask_area / box_area) if box_area > 0 else 0.0
    return {
        "mask_area": mask_area,
        "bbox_area": box_area,
        "mask_area_frac": float(mask_area / image_area),
        "bbox_area_frac": float(box_area / image_area),
        "fill_ratio": fill_ratio,
        "component_count": count_components(mask, min_area=min_component_area)
        if mask is not None
        else 0,
    }


def passes_mask_sanity(stats, args):
    """Apply intentionally loose guardrails before trusting appearance scores."""
    return (
        stats["mask_area_frac"] >= args.min_mask_area_frac
        and stats["bbox_area_frac"] <= args.max_bbox_area_frac
        and stats["fill_ratio"] >= args.min_mask_fill_ratio
        and stats["component_count"] <= args.max_mask_components
    )
