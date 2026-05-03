import argparse
import json
import os

import cv2
import numpy as np
import torch

from PIL import Image, ImageDraw

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_overlay_image(image, combined_mask, overlay_color=(255, 0, 0), alpha=0.4):
    """Blend a semi-transparent color overlay onto masked pixels."""
    image_np = np.array(image, dtype=np.float32)
    overlay_np = np.zeros_like(image_np)
    overlay_np[..., 0] = overlay_color[0]
    overlay_np[..., 1] = overlay_color[1]
    overlay_np[..., 2] = overlay_color[2]

    mask_np = combined_mask[..., None].astype(np.float32)
    blended = image_np * (1.0 - alpha * mask_np) + overlay_np * (alpha * mask_np)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


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


def save_outputs(
    image,
    combined_mask,
    output_dirs,
    base_name,
    overlay_color,
    overlay_alpha,
):
    mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8), mode="L")

    image_rgba = image.convert("RGBA")
    image_rgba.putalpha(mask_img)
    overlay_img = create_overlay_image(
        image=image,
        combined_mask=combined_mask,
        overlay_color=tuple(overlay_color),
        alpha=overlay_alpha,
    )

    masks_dir, masked_images_dir, overlay_images_dir = output_dirs
    mask_out_path = os.path.join(masks_dir, f"{base_name}.png")
    masked_out_path = os.path.join(masked_images_dir, f"{base_name}.png")
    overlay_out_path = os.path.join(overlay_images_dir, f"{base_name}.png")

    mask_img.save(mask_out_path, format="PNG")
    image_rgba.save(masked_out_path, format="PNG")
    overlay_img.save(overlay_out_path, format="PNG")
    print(f"  Saved mask: {mask_out_path}")
    print(f"  Saved masked image: {masked_out_path}")
    print(f"  Saved overlay image: {overlay_out_path}")


def save_bbox_visualization(image, bbox, output_dir, base_name, color=(0, 255, 0)):
    """Save an image with the prompt bbox drawn on top."""
    bbox_image = image.copy()
    draw = ImageDraw.Draw(bbox_image)
    line_width = max(2, round(min(image.size) * 0.004))
    draw.rectangle(bbox, outline=tuple(color), width=line_width)

    bbox_out_path = os.path.join(output_dir, f"{base_name}_bbox.png")
    bbox_image.save(bbox_out_path, format="PNG")
    print(f"  Saved bbox visualization: {bbox_out_path}")


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


def save_binary_mask(mask, output_dir, base_name):
    """Save a boolean mask as a 0/255 PNG."""
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_path = os.path.join(output_dir, f"{base_name}.png")
    mask_img.save(mask_path, format="PNG")
    print(f"  Saved SAM3-only mask: {mask_path}")


def save_sam3_candidate_masks(
    outputs,
    right_offset,
    right_image_size,
    output_dir,
    base_name,
):
    """Save raw right-side SAM3 candidate masks before component filtering."""
    masks = output_masks_to_numpy(outputs.get("masks"))
    if len(masks) == 0:
        return 0

    right_width, right_height = right_image_size
    num_saved = 0
    for idx, mask in enumerate(masks):
        right_crop = mask[:right_height, right_offset : right_offset + right_width]
        if not right_crop.any():
            continue
        save_binary_mask(right_crop, output_dir, f"{base_name}_candidate_{idx:02d}")
        num_saved += 1
    return num_saved


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


def save_pair_bbox_visualization(
    paired_image,
    left_bbox,
    right_bbox,
    right_offset,
    output_dir,
    base_name,
    prompt_color=(0, 255, 0),
    result_color=(255, 0, 0),
):
    """Save the concatenated pair with the example and extracted boxes drawn."""
    bbox_image = paired_image.copy()
    draw = ImageDraw.Draw(bbox_image)
    line_width = max(2, round(min(paired_image.size) * 0.004))
    draw.rectangle(left_bbox, outline=tuple(prompt_color), width=line_width)

    if right_bbox is not None:
        x_min, y_min, x_max, y_max = right_bbox
        shifted_bbox = [
            x_min + right_offset,
            y_min,
            x_max + right_offset,
            y_max,
        ]
        draw.rectangle(shifted_bbox, outline=tuple(result_color), width=line_width)

    pair_out_path = os.path.join(output_dir, f"{base_name}_pair_bbox.png")
    bbox_image.save(pair_out_path, format="PNG")
    print(f"  Saved paired bbox visualization: {pair_out_path}")


def l2_normalize(vector, eps=1e-12):
    """Return a float32 vector normalized to unit L2 length."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return None
    return vector / norm


def _normalized_histogram(values, bins, hist_range):
    """Compute a normalized 1D histogram for a single channel."""
    hist, _ = np.histogram(values, bins=bins, range=hist_range)
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist


def _color_histogram_embedding(rgb_pixels, hist_bins):
    """Build an explainable RGB/HSV/LAB histogram embedding from RGB pixels."""
    if rgb_pixels.size == 0:
        return None

    rgb_pixels = rgb_pixels.astype(np.uint8)
    hsv_pixels = cv2.cvtColor(rgb_pixels[:, None, :], cv2.COLOR_RGB2HSV)[:, 0, :]
    lab_pixels = cv2.cvtColor(rgb_pixels[:, None, :], cv2.COLOR_RGB2LAB)[:, 0, :]

    features = []
    for channel in range(3):
        features.append(
            _normalized_histogram(rgb_pixels[:, channel], hist_bins, (0, 256))
        )
    features.append(_normalized_histogram(hsv_pixels[:, 0], hist_bins, (0, 180)))
    features.append(_normalized_histogram(hsv_pixels[:, 1], hist_bins, (0, 256)))
    features.append(_normalized_histogram(hsv_pixels[:, 2], hist_bins, (0, 256)))
    for channel in range(3):
        features.append(
            _normalized_histogram(lab_pixels[:, channel], hist_bins, (0, 256))
        )

    return l2_normalize(np.concatenate(features))


def masked_crop_embedding(image_rgb, mask, bbox, padding, hist_bins, min_pixels=25):
    """Embed the predicted object appearance using masked crop color histograms."""
    if bbox is None or mask is None or not mask.any():
        return None

    image_np = np.array(image_rgb)
    height, width = mask.shape
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = min(width - 1, int(x_max) + padding)
    y_max = min(height - 1, int(y_max) + padding)

    crop = image_np[y_min : y_max + 1, x_min : x_max + 1]
    crop_mask = mask[y_min : y_max + 1, x_min : x_max + 1]
    pixels = crop[crop_mask]
    if len(pixels) < min_pixels:
        return None
    return _color_histogram_embedding(pixels, hist_bins)


def full_image_embedding(image_rgb, black_threshold, hist_bins, min_pixels=25):
    """Embed full-image appearance while downweighting black background pixels."""
    image_np = np.array(image_rgb)
    foreground = np.max(image_np, axis=2) > black_threshold
    pixels = image_np[foreground]
    if len(pixels) < min_pixels:
        pixels = image_np.reshape(-1, 3)
    return _color_histogram_embedding(pixels, hist_bins)


class HistogramAppearanceExtractor:
    """Deterministic color histogram embeddings for masked object pixels."""

    name = "masked RGB/HSV/LAB histogram, L2-normalized"

    def __init__(self, args):
        self.args = args

    def extract(self, image_rgb, mask, bbox):
        object_embedding = masked_crop_embedding(
            image_rgb=image_rgb,
            mask=mask,
            bbox=bbox,
            padding=self.args.embedding_crop_padding,
            hist_bins=self.args.embedding_hist_bins,
        )
        image_embedding = full_image_embedding(
            image_rgb=image_rgb,
            black_threshold=self.args.black_threshold,
            hist_bins=self.args.embedding_hist_bins,
        )
        return {
            "object_embedding": object_embedding,
            "image_embedding": image_embedding,
            "patch_embeddings": None,
        }


class DinoV3PatchAppearanceExtractor:
    """DINOv3 dense patch embeddings sampled from the predicted mask region."""

    name = "DINOv3 masked patch embeddings with mean-pooled object descriptor"

    def __init__(self, args):
        import timm

        self.args = args
        self.device = args.dinov3_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.image_size = int(args.dinov3_image_size)
        self.model = timm.create_model(
            args.dinov3_model,
            pretrained=args.dinov3_pretrained,
            img_size=self.image_size,
        )
        self.model.eval().to(self.device)
        patch_size = getattr(self.model.patch_embed, "patch_size", (16, 16))
        self.patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"--dinov3_image_size must be divisible by patch size "
                f"{self.patch_size}."
            )
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 0) or 0)
        self.mean = torch.tensor(
            [0.430, 0.411, 0.296], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.213, 0.156, 0.143], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

    def _image_tensor(self, image_rgb):
        resized = image_rgb.resize((self.image_size, self.image_size), Image.BICUBIC)
        image_np = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image_np).permute(2, 0, 1)[None].to(self.device)
        return (tensor - self.mean) / self.std

    def _patch_tokens(self, image_rgb):
        with torch.inference_mode():
            batch = self._image_tensor(image_rgb)
            tokens = self.model.forward_features(batch)

        if isinstance(tokens, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "tokens"):
                if key in tokens:
                    tokens = tokens[key]
                    break
            else:
                raise RuntimeError(
                    "DINOv3 forward_features returned a dict without patch tokens."
                )

        if tokens.ndim == 4:
            tokens = tokens.flatten(2).transpose(1, 2)
        if tokens.ndim != 3:
            raise RuntimeError(
                f"Expected DINOv3 patch tokens shaped BxNxD, got {tokens.shape}."
            )

        if tokens.shape[1] >= self.num_prefix_tokens + self.num_patches:
            tokens = tokens[
                :, self.num_prefix_tokens : self.num_prefix_tokens + self.num_patches
            ]
        elif tokens.shape[1] != self.num_patches:
            raise RuntimeError(
                f"Expected {self.num_patches} DINOv3 patch tokens, got "
                f"{tokens.shape[1]}."
            )

        tokens = torch.nn.functional.normalize(tokens[0], dim=-1)
        return tokens.detach().cpu().numpy().astype(np.float32)

    def _mask_grid(self, mask):
        if mask is None or not mask.any():
            return np.zeros((self.grid_size, self.grid_size), dtype=bool)
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        mask_image = mask_image.resize(
            (self.grid_size, self.grid_size), Image.Resampling.NEAREST
        )
        return np.asarray(mask_image) > 0

    def extract(self, image_rgb, mask, bbox):
        del bbox
        patch_tokens = self._patch_tokens(image_rgb)
        mask_grid = self._mask_grid(mask).reshape(-1)
        patch_embeddings = patch_tokens[mask_grid]
        if len(patch_embeddings) == 0:
            object_embedding = None
        else:
            object_embedding = l2_normalize(patch_embeddings.mean(axis=0))

        image_embedding = l2_normalize(patch_tokens.mean(axis=0))
        return {
            "object_embedding": object_embedding,
            "image_embedding": image_embedding,
            "patch_embeddings": patch_embeddings,
        }


def build_appearance_extractor(args):
    """Construct the configured appearance embedding backend."""
    if not args.appearance_filter:
        return None
    if args.embedding_backend == "histogram":
        return HistogramAppearanceExtractor(args)
    if args.embedding_backend == "dinov3":
        return DinoV3PatchAppearanceExtractor(args)
    raise ValueError(f"Unknown embedding backend: {args.embedding_backend}")


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


def create_prediction_record(
    base_name,
    frame_idx,
    image_path,
    image,
    mask,
    bbox,
    args,
    source,
    appearance_extractor,
    anchor_name=None,
):
    """Package a mask prediction with quality and appearance metadata."""
    raw_bbox = get_mask_bbox(mask, padding=0) if mask is not None else None
    prompt_bbox = bbox if bbox is not None else raw_bbox
    stats = mask_quality_stats(
        mask=mask,
        bbox=prompt_bbox,
        image_size=image.size,
        min_component_area=args.min_contour_area,
    )
    embeddings = (
        appearance_extractor.extract(image, mask, raw_bbox)
        if appearance_extractor is not None
        else {
            "object_embedding": None,
            "image_embedding": None,
            "patch_embeddings": None,
        }
    )
    return {
        "base_name": base_name,
        "frame_idx": frame_idx,
        "image_path": image_path,
        "mask": mask,
        "raw_bbox": raw_bbox,
        "bbox": prompt_bbox,
        "source": source,
        "anchor_name": anchor_name,
        "stats": stats,
        "object_embedding": embeddings["object_embedding"],
        "image_embedding": embeddings["image_embedding"],
        "patch_embeddings": embeddings["patch_embeddings"],
        "mask_sane": passes_mask_sanity(stats, args),
        "appearance_score": None,
        "patch_mean_score": None,
        "patch_bad_fraction": None,
        "mutual_neighbor_count": 0,
        "inlier": False,
        "rejection_reasons": [],
    }


def robust_lower_threshold(values, tau):
    """Median minus tau scaled-MAD robust spread."""
    values = np.asarray(values, dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma < 1e-8:
        robust_sigma = float(np.std(values))
    if robust_sigma < 1e-8:
        robust_sigma = 1e-6
    return median - tau * robust_sigma, median, robust_sigma


def score_nearest_neighbor_consistency(records, top_k, tau, min_mutual_neighbors):
    """Classify records with top-k embedding consistency and robust MAD threshold."""
    valid_indices = [
        idx
        for idx, record in enumerate(records)
        if record["object_embedding"] is not None
    ]
    if len(valid_indices) <= 2:
        for record in records:
            record["appearance_score"] = (
                1.0 if record["object_embedding"] is not None else None
            )
            record["mutual_neighbor_count"] = 1
            record["inlier"] = (
                record["mask_sane"] and record["object_embedding"] is not None
            )
            if not record["mask_sane"]:
                record["rejection_reasons"].append("mask_sanity")
            if record["object_embedding"] is None:
                record["rejection_reasons"].append("missing_embedding")
        return {
            "threshold": None,
            "median": None,
            "robust_sigma": None,
            "top_k": 0,
            "num_scored": len(valid_indices),
        }

    embeddings = np.stack([records[idx]["object_embedding"] for idx in valid_indices])
    similarities = embeddings @ embeddings.T
    np.fill_diagonal(similarities, -np.inf)

    actual_top_k = max(1, min(top_k, len(valid_indices) - 1))
    neighbor_sets = []
    scores = []
    for row in similarities:
        neighbors = np.argsort(row)[-actual_top_k:][::-1]
        neighbor_sets.append(set(int(neighbor) for neighbor in neighbors))
        scores.append(float(np.mean(row[neighbors])))

    threshold, median, robust_sigma = robust_lower_threshold(scores, tau=tau)

    for local_idx, record_idx in enumerate(valid_indices):
        mutual_count = sum(
            local_idx in neighbor_sets[neighbor_idx]
            for neighbor_idx in neighbor_sets[local_idx]
        )
        record = records[record_idx]
        record["appearance_score"] = scores[local_idx]
        record["mutual_neighbor_count"] = int(mutual_count)

        appearance_inlier = scores[local_idx] >= threshold
        mutual_inlier = mutual_count >= min_mutual_neighbors
        record["inlier"] = record["mask_sane"] and appearance_inlier and mutual_inlier
        if not record["mask_sane"]:
            record["rejection_reasons"].append("mask_sanity")
        if not appearance_inlier:
            record["rejection_reasons"].append("appearance")
        if not mutual_inlier:
            record["rejection_reasons"].append("mutual_neighbors")

    for idx, record in enumerate(records):
        if idx in valid_indices:
            continue
        record["inlier"] = False
        record["rejection_reasons"].append("missing_embedding")

    return {
        "threshold": threshold,
        "median": median,
        "robust_sigma": robust_sigma,
        "top_k": actual_top_k,
        "num_scored": len(valid_indices),
    }


def score_embedding_against_inliers(embedding, inlier_records, top_k):
    """Score a candidate by mean cosine similarity to its top-k inlier embeddings."""
    if embedding is None:
        return None
    inlier_embeddings = [
        record["object_embedding"]
        for record in inlier_records
        if record["object_embedding"] is not None
    ]
    if not inlier_embeddings:
        return None
    similarities = np.stack(inlier_embeddings) @ embedding
    actual_top_k = max(1, min(top_k, len(similarities)))
    top_scores = np.sort(similarities)[-actual_top_k:]
    return float(np.mean(top_scores))


def build_patch_bank(inlier_records, max_patches):
    """Collect normalized DINO patch embeddings from current inlier masks."""
    patch_sets = [
        record["patch_embeddings"]
        for record in inlier_records
        if record.get("patch_embeddings") is not None
        and len(record["patch_embeddings"]) > 0
    ]
    if not patch_sets:
        return None
    patch_bank = np.concatenate(patch_sets, axis=0).astype(np.float32)
    if len(patch_bank) > max_patches:
        indices = np.linspace(0, len(patch_bank) - 1, max_patches).astype(int)
        patch_bank = patch_bank[indices]
    return patch_bank


def score_patches_against_bank(patch_embeddings, patch_bank, similarity_threshold):
    """Return mean nearest-patch similarity and low-similarity patch fraction."""
    if patch_embeddings is None or len(patch_embeddings) == 0 or patch_bank is None:
        return None, None
    similarities = patch_embeddings @ patch_bank.T
    best_patch_scores = similarities.max(axis=1)
    return (
        float(best_patch_scores.mean()),
        float(np.mean(best_patch_scores < similarity_threshold)),
    )


def apply_dinov3_patch_consistency(records, patch_bank, args):
    """Reject masks containing too many DINO patches unlike trusted object patches."""
    if patch_bank is None:
        return {"enabled": True, "num_patch_bank": 0, "num_rejected": 0}

    num_rejected = 0
    for record in records:
        mean_score, bad_fraction = score_patches_against_bank(
            patch_embeddings=record.get("patch_embeddings"),
            patch_bank=patch_bank,
            similarity_threshold=args.dinov3_patch_similarity_threshold,
        )
        record["patch_mean_score"] = mean_score
        record["patch_bad_fraction"] = bad_fraction
        patch_inlier = (
            mean_score is not None
            and mean_score >= args.dinov3_min_patch_mean_score
            and bad_fraction <= args.dinov3_max_bad_patch_fraction
        )
        if record["inlier"] and not patch_inlier:
            record["inlier"] = False
            record["rejection_reasons"].append("dinov3_patches")
            num_rejected += 1

    return {
        "enabled": True,
        "num_patch_bank": int(len(patch_bank)),
        "similarity_threshold": args.dinov3_patch_similarity_threshold,
        "max_bad_patch_fraction": args.dinov3_max_bad_patch_fraction,
        "min_patch_mean_score": args.dinov3_min_patch_mean_score,
        "num_rejected": num_rejected,
    }


def select_anchor_records(target_record, inlier_records, args):
    """Choose top inlier anchors by index, full-image appearance, or a hybrid."""
    if not inlier_records:
        return []

    target_embedding = target_record["image_embedding"]
    max_index_distance = max(
        1,
        max(
            abs(record["frame_idx"] - target_record["frame_idx"])
            for record in inlier_records
        ),
    )
    scored = []
    for record in inlier_records:
        index_distance = abs(record["frame_idx"] - target_record["frame_idx"])
        index_score = -float(index_distance / max_index_distance)
        if target_embedding is not None and record["image_embedding"] is not None:
            appearance_score = float(target_embedding @ record["image_embedding"])
        else:
            appearance_score = 0.0

        if args.rerun_anchor_strategy == "index":
            score = index_score
        elif args.rerun_anchor_strategy == "appearance":
            score = appearance_score
        else:
            score = appearance_score + args.anchor_index_weight * index_score
        scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[: args.rerun_anchor_top_k]]


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


def serializable_record(record):
    """Drop arrays from a prediction record so it can be written as JSON."""
    return {
        "frame_idx": record["frame_idx"],
        "image_path": record["image_path"],
        "source": record["source"],
        "anchor_name": record["anchor_name"],
        "raw_bbox": record["raw_bbox"],
        "bbox": record["bbox"],
        "stats": record["stats"],
        "mask_sane": record["mask_sane"],
        "appearance_score": record["appearance_score"],
        "patch_mean_score": record["patch_mean_score"],
        "patch_bad_fraction": record["patch_bad_fraction"],
        "mutual_neighbor_count": record["mutual_neighbor_count"],
        "inlier": record["inlier"],
        "rejection_reasons": record["rejection_reasons"],
    }


def write_appearance_diagnostics(output_dir, records, summary):
    """Save inlier/outlier metadata for manual inspection and rerun auditing."""
    diagnostics_path = os.path.join(output_dir, "appearance_consistency.json")
    payload = {
        "summary": summary,
        "predictions": {
            record["base_name"]: serializable_record(record) for record in records
        },
    }
    with open(diagnostics_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote appearance diagnostics: {diagnostics_path}")


def write_bootstrap_debug_metadata(output_dir, metadata):
    """Save threshold bootstrap prompt metadata for later inspection."""
    metadata_path = os.path.join(output_dir, "bootstrap_debug", "bootstrap_bbox.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote bootstrap debug metadata: {metadata_path}")


def process_images():
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap an object mask from scan00 using thresholding, convert it "
            "to a box prompt, then run SAM3 on concatenated scan00|scanXX image "
            "pairs and extract the right-side box/mask."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to save outputs.",
    )
    parser.add_argument(
        "--black_threshold",
        type=int,
        default=20,
        help="Pixels brighter than this 0-255 threshold are treated as foreground.",
    )
    parser.add_argument(
        "--min_contour_area",
        type=float,
        default=100.0,
        help="Discard connected components smaller than this area in pixels.",
    )
    parser.add_argument(
        "--morph_kernel_size",
        type=int,
        default=5,
        help="Kernel size for opening/closing cleanup. Use 0 to disable.",
    )
    parser.add_argument(
        "--keep_largest",
        action="store_true",
        help="Deprecated compatibility flag. The bootstrap mask always keeps the largest component.",
    )
    parser.add_argument(
        "--bootstrap_stem",
        type=str,
        default="scan00",
        help="Image stem to threshold for the initial SAM3 image box prompt.",
    )
    parser.add_argument(
        "--bbox_padding",
        type=int,
        default=20,
        help="Pixels to expand mask-derived boxes in saved bbox visualizations.",
    )
    parser.add_argument(
        "--sam_model_id",
        type=str,
        default="facebook/sam3",
        help="Deprecated compatibility argument. Native SAM3 uses --checkpoint_path.",
    )
    parser.add_argument(
        "--sam_version",
        type=str,
        default="sam3",
        choices=("sam3", "sam3.1"),
        help="Deprecated compatibility argument. The paired-image path uses the SAM3 image model.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to a native SAM3 image checkpoint. If omitted, SAM3 downloads/uses its default checkpoint.",
    )
    parser.add_argument(
        "--bpe_path",
        type=str,
        default=None,
        help="Optional path to the native SAM3 BPE vocabulary.",
    )
    parser.add_argument(
        "--sam_compile",
        action="store_true",
        help="Enable native SAM3 torch.compile where supported by the image model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Optional text prompt to combine with the scan00 box prompt "
            "(for example, 'flower' or 'rock'). If omitted, SAM3 uses only "
            "the visual box prompt."
        ),
    )
    parser.add_argument(
        "--sam_output_prob_thresh",
        type=float,
        default=0.5,
        help="Native SAM3 image output confidence threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for SAM3. Defaults to cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.4,
        help="Blend strength for the colored overlay on masked pixels.",
    )
    parser.add_argument(
        "--overlay_color",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(255, 0, 0),
        help="Overlay color as three integers in RGB order.",
    )
    parser.add_argument(
        "--appearance_filter",
        action="store_true",
        help=(
            "Score first-pass predictions with nearest-neighbor masked-appearance "
            "consistency and write appearance_consistency.json."
        ),
    )
    parser.add_argument(
        "--rerun_outliers",
        action="store_true",
        help=(
            "After appearance filtering, rerun rejected frames using the closest "
            "inlier anchors and overwrite accepted improved masks."
        ),
    )
    parser.add_argument(
        "--embedding_hist_bins",
        type=int,
        default=16,
        help="Histogram bins per color channel for appearance embeddings.",
    )
    parser.add_argument(
        "--embedding_backend",
        choices=("histogram", "dinov3"),
        default="histogram",
        help="Appearance embedding backend used for inlier detection.",
    )
    parser.add_argument(
        "--embedding_crop_padding",
        type=int,
        default=20,
        help="Pixels to pad around the raw mask bbox before extracting embeddings.",
    )
    parser.add_argument(
        "--appearance_top_k",
        type=int,
        default=5,
        help="Number of nearest neighbors averaged for appearance consistency.",
    )
    parser.add_argument(
        "--appearance_tau",
        type=float,
        default=3.0,
        help="Robust MAD multiplier for the lower inlier threshold.",
    )
    parser.add_argument(
        "--appearance_min_mutual_neighbors",
        type=int,
        default=1,
        help="Minimum mutual top-k neighbor count required for an inlier.",
    )
    parser.add_argument(
        "--min_mask_area_frac",
        type=float,
        default=1e-5,
        help="Loose sanity check: minimum mask area as a fraction of image area.",
    )
    parser.add_argument(
        "--max_bbox_area_frac",
        type=float,
        default=0.8,
        help="Loose sanity check: maximum bbox area as a fraction of image area.",
    )
    parser.add_argument(
        "--min_mask_fill_ratio",
        type=float,
        default=0.02,
        help="Loose sanity check: minimum mask_area / bbox_area.",
    )
    parser.add_argument(
        "--max_mask_components",
        type=int,
        default=3,
        help="Loose sanity check: maximum connected components in a saved mask.",
    )
    parser.add_argument(
        "--rerun_anchor_top_k",
        type=int,
        default=3,
        help="Number of closest inlier anchors to try for each rejected frame.",
    )
    parser.add_argument(
        "--rerun_anchor_strategy",
        choices=("hybrid", "appearance", "index"),
        default="hybrid",
        help="How to rank inlier anchors for outlier reruns.",
    )
    parser.add_argument(
        "--anchor_index_weight",
        type=float,
        default=0.15,
        help="Hybrid anchor score penalty weight for normalized scan-index distance.",
    )
    parser.add_argument(
        "--dinov3_model",
        type=str,
        default="vit_small_patch16_dinov3",
        help="timm DINOv3 model name for patch-level appearance scoring.",
    )
    parser.add_argument(
        "--dinov3_pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load pretrained DINOv3 weights through timm.",
    )
    parser.add_argument(
        "--dinov3_image_size",
        type=int,
        default=512,
        help="Square DINOv3 input size; must be divisible by the patch size.",
    )
    parser.add_argument(
        "--dinov3_device",
        type=str,
        default=None,
        help="Device for DINOv3 feature extraction. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--dinov3_patch_similarity_threshold",
        type=float,
        default=0.45,
        help="Patch is suspicious if its nearest trusted DINO patch cosine is below this.",
    )
    parser.add_argument(
        "--dinov3_max_bad_patch_fraction",
        type=float,
        default=0.2,
        help="Reject a mask if more than this fraction of masked DINO patches is suspicious.",
    )
    parser.add_argument(
        "--dinov3_min_patch_mean_score",
        type=float,
        default=-1.0,
        help="Reject a mask if mean nearest trusted DINO patch cosine is below this.",
    )
    parser.add_argument(
        "--dinov3_max_patch_bank",
        type=int,
        default=20000,
        help="Maximum trusted object DINO patches kept for nearest-patch scoring.",
    )
    args = parser.parse_args()
    if args.rerun_outliers:
        args.appearance_filter = True

    os.makedirs(args.output_dir, exist_ok=True)
    masks_dir = os.path.join(args.output_dir, "masks")
    masked_images_dir = os.path.join(args.output_dir, "masked_images")
    overlay_images_dir = os.path.join(args.output_dir, "overlay_images")
    bbox_images_dir = os.path.join(args.output_dir, "bbox_images")
    paired_images_dir = os.path.join(args.output_dir, "paired_images")
    sam3_only_masks_dir = os.path.join(args.output_dir, "sam3_only_masks")
    sam3_candidate_masks_dir = os.path.join(sam3_only_masks_dir, "candidates")
    sam3_selected_masks_dir = os.path.join(sam3_only_masks_dir, "selected")
    bootstrap_debug_dir = os.path.join(args.output_dir, "bootstrap_debug")
    threshold_masks_dir = os.path.join(bootstrap_debug_dir, "threshold_masks")
    threshold_masked_images_dir = os.path.join(
        bootstrap_debug_dir, "threshold_masked_images"
    )
    threshold_overlay_images_dir = os.path.join(
        bootstrap_debug_dir, "threshold_overlay_images"
    )
    threshold_bbox_images_dir = os.path.join(
        bootstrap_debug_dir, "threshold_bbox_images"
    )
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masked_images_dir, exist_ok=True)
    os.makedirs(overlay_images_dir, exist_ok=True)
    os.makedirs(bbox_images_dir, exist_ok=True)
    os.makedirs(paired_images_dir, exist_ok=True)
    os.makedirs(sam3_candidate_masks_dir, exist_ok=True)
    os.makedirs(sam3_selected_masks_dir, exist_ok=True)
    os.makedirs(threshold_masks_dir, exist_ok=True)
    os.makedirs(threshold_masked_images_dir, exist_ok=True)
    os.makedirs(threshold_overlay_images_dir, exist_ok=True)
    os.makedirs(threshold_bbox_images_dir, exist_ok=True)

    appearance_extractor = build_appearance_extractor(args)
    if appearance_extractor is not None:
        print(f"Using appearance backend: {appearance_extractor.name}")

    image_paths = list_input_images(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {args.input_dir!r}.")

    bootstrap_path = find_bootstrap_image(image_paths, args.bootstrap_stem)
    if bootstrap_path is None:
        raise FileNotFoundError(f"No bootstrap image found in {args.input_dir!r}.")
    bootstrap_frame_idx = image_paths.index(bootstrap_path)

    print(f"Bootstrapping threshold mask from: {os.path.basename(bootstrap_path)}")
    bootstrap_image = Image.open(bootstrap_path).convert("RGB")
    bootstrap_mask, num_components = create_foreground_mask(
        image_rgb=bootstrap_image,
        black_threshold=args.black_threshold,
        min_contour_area=args.min_contour_area,
        morph_kernel_size=args.morph_kernel_size,
        keep_largest=True,
    )

    if not bootstrap_mask.any():
        raise RuntimeError(
            f"No foreground component found in {os.path.basename(bootstrap_path)!r}."
        )

    bbox = get_mask_bbox(bootstrap_mask, padding=args.bbox_padding)
    if bbox is None:
        raise RuntimeError("Could not compute a bounding box from the bootstrap mask.")

    output_dirs = (masks_dir, masked_images_dir, overlay_images_dir)
    threshold_output_dirs = (
        threshold_masks_dir,
        threshold_masked_images_dir,
        threshold_overlay_images_dir,
    )
    bootstrap_base_name = os.path.splitext(os.path.basename(bootstrap_path))[0]
    records = []
    save_outputs(
        image=bootstrap_image,
        combined_mask=bootstrap_mask,
        output_dirs=threshold_output_dirs,
        base_name=bootstrap_base_name,
        overlay_color=args.overlay_color,
        overlay_alpha=args.overlay_alpha,
    )
    save_bbox_visualization(
        image=bootstrap_image,
        bbox=bbox,
        output_dir=threshold_bbox_images_dir,
        base_name=f"{bootstrap_base_name}_threshold_prompt",
        color=args.overlay_color,
    )
    write_bootstrap_debug_metadata(
        output_dir=args.output_dir,
        metadata={
            "bootstrap_image": os.path.basename(bootstrap_path),
            "bootstrap_stem": bootstrap_base_name,
            "threshold_components_kept": num_components,
            "threshold_bbox_xyxy": bbox,
            "threshold_mask_path": os.path.join(
                "bootstrap_debug",
                "threshold_masks",
                f"{bootstrap_base_name}.png",
            ),
            "threshold_bbox_visualization": os.path.join(
                "bootstrap_debug",
                "threshold_bbox_images",
                f"{bootstrap_base_name}_threshold_prompt_bbox.png",
            ),
        },
    )
    save_outputs(
        image=bootstrap_image,
        combined_mask=bootstrap_mask,
        output_dirs=output_dirs,
        base_name=bootstrap_base_name,
        overlay_color=args.overlay_color,
        overlay_alpha=args.overlay_alpha,
    )
    save_bbox_visualization(
        image=bootstrap_image,
        bbox=bbox,
        output_dir=bbox_images_dir,
        base_name=bootstrap_base_name,
        color=args.overlay_color,
    )
    print(f"  Bootstrap components kept: {num_components}")
    print(f"  Bootstrap bbox xyxy: {bbox}")
    records.append(
        create_prediction_record(
            base_name=bootstrap_base_name,
            frame_idx=bootstrap_frame_idx,
            image_path=bootstrap_path,
            image=bootstrap_image,
            mask=bootstrap_mask,
            bbox=bbox,
            args=args,
            source="bootstrap_threshold",
            appearance_extractor=appearance_extractor,
        )
    )
    prompt = args.prompt.strip() if args.prompt and args.prompt.strip() else None
    if prompt:
        print(f"  SAM3 text prompt: {prompt!r}")
    else:
        print("  SAM3 text prompt: none; using scan00 bbox prompt only")

    if len(image_paths) == 1:
        print("No remaining images to process with SAM3.")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading native SAM3 image model on {device}")
    if args.sam_version != "sam3":
        print(
            "  Warning: --sam_version is kept for compatibility; "
            "paired-image inference uses the SAM3 image model."
        )

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint_path,
        bpe_path=args.bpe_path,
        device=device,
        compile=args.sam_compile,
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.sam_output_prob_thresh,
    )

    print("Running SAM3 on concatenated scan00|scanXX image pairs...")
    for frame_idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        if frame_idx == bootstrap_frame_idx:
            continue

        print(f"Processing paired SAM3 image: scan00 | {filename}")
        image = Image.open(img_path).convert("RGB")
        (
            combined_mask,
            tracked_bbox,
            paired_image,
            right_offset,
        ) = run_paired_sam_prediction(
            processor=processor,
            anchor_image=bootstrap_image,
            anchor_bbox=bbox,
            target_image=image,
            prompt=prompt,
            device=device,
            min_area=args.min_contour_area,
            sam3_candidate_masks_dir=sam3_candidate_masks_dir,
            sam3_selected_masks_dir=sam3_selected_masks_dir,
            sam3_debug_base_name=f"{base_name}_initial",
        )
        save_pair_bbox_visualization(
            paired_image=paired_image,
            left_bbox=bbox,
            right_bbox=tracked_bbox,
            right_offset=right_offset,
            output_dir=paired_images_dir,
            base_name=base_name,
            prompt_color=(0, 255, 0),
            result_color=args.overlay_color,
        )

        if combined_mask is None or not combined_mask.any():
            print(f"  No right-side SAM3 mask found in {filename}. Marking outlier...")
            empty_mask = np.zeros((image.height, image.width), dtype=bool)
            records.append(
                create_prediction_record(
                    base_name=base_name,
                    frame_idx=frame_idx,
                    image_path=img_path,
                    image=image,
                    mask=empty_mask,
                    bbox=None,
                    args=args,
                    source="initial_missing",
                    appearance_extractor=appearance_extractor,
                    anchor_name=bootstrap_base_name,
                )
            )
            continue

        tracked_bbox = get_mask_bbox(combined_mask, padding=args.bbox_padding)
        if tracked_bbox is not None:
            save_bbox_visualization(
                image=image,
                bbox=tracked_bbox,
                output_dir=bbox_images_dir,
                base_name=base_name,
                color=args.overlay_color,
            )

        save_outputs(
            image=image,
            combined_mask=combined_mask,
            output_dirs=output_dirs,
            base_name=base_name,
            overlay_color=args.overlay_color,
            overlay_alpha=args.overlay_alpha,
        )
        records.append(
            create_prediction_record(
                base_name=base_name,
                frame_idx=frame_idx,
                image_path=img_path,
                image=image,
                mask=combined_mask,
                bbox=tracked_bbox,
                args=args,
                source="initial_pair",
                appearance_extractor=appearance_extractor,
                anchor_name=bootstrap_base_name,
            )
        )

    if not args.appearance_filter:
        return

    print("Scoring predictions with nearest-neighbor appearance consistency...")
    consistency_summary = score_nearest_neighbor_consistency(
        records=records,
        top_k=args.appearance_top_k,
        tau=args.appearance_tau,
        min_mutual_neighbors=args.appearance_min_mutual_neighbors,
    )
    patch_summary = {"enabled": False}
    patch_bank = None
    if args.embedding_backend == "dinov3":
        initial_inlier_records = [record for record in records if record["inlier"]]
        patch_bank = build_patch_bank(
            inlier_records=initial_inlier_records,
            max_patches=args.dinov3_max_patch_bank,
        )
        patch_summary = apply_dinov3_patch_consistency(
            records=records,
            patch_bank=patch_bank,
            args=args,
        )
        patch_bank = build_patch_bank(
            inlier_records=[record for record in records if record["inlier"]],
            max_patches=args.dinov3_max_patch_bank,
        )
    num_inliers = sum(record["inlier"] for record in records)
    num_outliers = len(records) - num_inliers
    print(
        "  Appearance inliers/outliers: "
        f"{num_inliers}/{num_outliers} "
        f"(threshold={consistency_summary['threshold']})"
    )
    if patch_summary["enabled"]:
        print(
            "  DINOv3 patch bank/rejections: "
            f"{patch_summary['num_patch_bank']}/"
            f"{patch_summary['num_rejected']}"
        )

    rerun_summary = {
        "enabled": args.rerun_outliers,
        "attempted": 0,
        "accepted": 0,
        "failed": 0,
    }

    if args.rerun_outliers:
        inlier_records = [record for record in records if record["inlier"]]
        outlier_records = [record for record in records if not record["inlier"]]
        appearance_threshold = consistency_summary["threshold"]
        print(
            "Rerunning outliers with closest inlier anchors: "
            f"{len(outlier_records)} target(s), {len(inlier_records)} anchor(s)"
        )

        record_by_name = {record["base_name"]: record for record in records}
        for target_record in outlier_records:
            if target_record["base_name"] == bootstrap_base_name:
                print("  Skipping bootstrap outlier; no alternate bootstrap logic.")
                rerun_summary["failed"] += 1
                continue

            anchors = select_anchor_records(
                target_record=target_record,
                inlier_records=inlier_records,
                args=args,
            )
            if not anchors:
                print(
                    f"  No inlier anchors available for "
                    f"{target_record['base_name']}."
                )
                rerun_summary["failed"] += 1
                continue

            target_image = Image.open(target_record["image_path"]).convert("RGB")
            best_candidate = None
            print(
                f"  Rerunning {target_record['base_name']} with anchors: "
                + ", ".join(anchor["base_name"] for anchor in anchors)
            )
            for anchor_record in anchors:
                if anchor_record["bbox"] is None:
                    continue

                rerun_summary["attempted"] += 1
                anchor_image = Image.open(anchor_record["image_path"]).convert("RGB")
                (
                    candidate_mask,
                    candidate_bbox,
                    paired_image,
                    right_offset,
                ) = run_paired_sam_prediction(
                    processor=processor,
                    anchor_image=anchor_image,
                    anchor_bbox=anchor_record["bbox"],
                    target_image=target_image,
                    prompt=prompt,
                    device=device,
                    min_area=args.min_contour_area,
                    sam3_candidate_masks_dir=sam3_candidate_masks_dir,
                    sam3_selected_masks_dir=sam3_selected_masks_dir,
                    sam3_debug_base_name=(
                        f"{target_record['base_name']}_rerun_"
                        f"{anchor_record['base_name']}"
                    ),
                )
                save_pair_bbox_visualization(
                    paired_image=paired_image,
                    left_bbox=anchor_record["bbox"],
                    right_bbox=candidate_bbox,
                    right_offset=right_offset,
                    output_dir=paired_images_dir,
                    base_name=(
                        f"{target_record['base_name']}_rerun_"
                        f"{anchor_record['base_name']}"
                    ),
                    prompt_color=(0, 255, 0),
                    result_color=args.overlay_color,
                )
                if candidate_mask is None or not candidate_mask.any():
                    continue

                candidate_prompt_bbox = get_mask_bbox(
                    candidate_mask, padding=args.bbox_padding
                )
                candidate_record = create_prediction_record(
                    base_name=target_record["base_name"],
                    frame_idx=target_record["frame_idx"],
                    image_path=target_record["image_path"],
                    image=target_image,
                    mask=candidate_mask,
                    bbox=candidate_prompt_bbox,
                    args=args,
                    source="rerun_pair",
                    appearance_extractor=appearance_extractor,
                    anchor_name=anchor_record["base_name"],
                )
                candidate_score = score_embedding_against_inliers(
                    embedding=candidate_record["object_embedding"],
                    inlier_records=inlier_records,
                    top_k=args.appearance_top_k,
                )
                candidate_record["appearance_score"] = candidate_score
                threshold_passed = candidate_score is not None and (
                    appearance_threshold is None
                    or candidate_score >= appearance_threshold
                )
                patch_passed = True
                if args.embedding_backend == "dinov3":
                    mean_score, bad_fraction = score_patches_against_bank(
                        patch_embeddings=candidate_record.get("patch_embeddings"),
                        patch_bank=patch_bank,
                        similarity_threshold=args.dinov3_patch_similarity_threshold,
                    )
                    candidate_record["patch_mean_score"] = mean_score
                    candidate_record["patch_bad_fraction"] = bad_fraction
                    patch_passed = (
                        mean_score is not None
                        and mean_score >= args.dinov3_min_patch_mean_score
                        and bad_fraction <= args.dinov3_max_bad_patch_fraction
                    )
                candidate_record["inlier"] = (
                    candidate_record["mask_sane"] and threshold_passed and patch_passed
                )
                if not candidate_record["mask_sane"]:
                    candidate_record["rejection_reasons"].append("mask_sanity")
                if not threshold_passed:
                    candidate_record["rejection_reasons"].append("appearance")
                if not patch_passed:
                    candidate_record["rejection_reasons"].append("dinov3_patches")

                best_score = (
                    best_candidate["appearance_score"]
                    if best_candidate is not None
                    and best_candidate["appearance_score"] is not None
                    else -np.inf
                )
                candidate_sort_score = (
                    candidate_score if candidate_score is not None else -np.inf
                )
                if best_candidate is None:
                    best_candidate = candidate_record
                elif candidate_record["inlier"] and not best_candidate["inlier"]:
                    best_candidate = candidate_record
                elif candidate_record["inlier"] == best_candidate["inlier"] and (
                    candidate_sort_score > best_score
                ):
                    best_candidate = candidate_record

            if best_candidate is None or not best_candidate["inlier"]:
                print(
                    f"    No accepted rerun candidate for "
                    f"{target_record['base_name']}."
                )
                rerun_summary["failed"] += 1
                continue

            replacement_bbox = best_candidate["bbox"]
            if replacement_bbox is not None:
                save_bbox_visualization(
                    image=target_image,
                    bbox=replacement_bbox,
                    output_dir=bbox_images_dir,
                    base_name=target_record["base_name"],
                    color=args.overlay_color,
                )
            save_outputs(
                image=target_image,
                combined_mask=best_candidate["mask"],
                output_dirs=output_dirs,
                base_name=target_record["base_name"],
                overlay_color=args.overlay_color,
                overlay_alpha=args.overlay_alpha,
            )
            best_candidate["rejection_reasons"] = []
            record_by_name[target_record["base_name"]] = best_candidate
            rerun_summary["accepted"] += 1
            print(
                f"    Accepted rerun for {target_record['base_name']} "
                f"from anchor {best_candidate['anchor_name']} "
                f"(score={best_candidate['appearance_score']:.4f})"
            )

        records = [record_by_name[record["base_name"]] for record in records]

    bootstrap_update_summary = {
        "enabled": True,
        "attempted": 0,
        "updated": False,
        "anchor_name": None,
        "reason": None,
    }
    record_by_name = {record["base_name"]: record for record in records}
    bootstrap_record = record_by_name.get(bootstrap_base_name)
    final_inlier_records = [
        record
        for record in records
        if record["base_name"] != bootstrap_base_name
        and record["inlier"]
        and record["bbox"] is not None
    ]
    if bootstrap_record is None:
        bootstrap_update_summary["reason"] = "missing_bootstrap_record"
    elif not final_inlier_records:
        bootstrap_update_summary["reason"] = "no_non_bootstrap_inlier_anchor"
        print("No non-bootstrap inlier anchor available to update scan00.")
    else:
        if args.embedding_backend == "dinov3":
            patch_bank = build_patch_bank(
                inlier_records=final_inlier_records,
                max_patches=args.dinov3_max_patch_bank,
            )
        anchors = select_anchor_records(
            target_record=bootstrap_record,
            inlier_records=final_inlier_records,
            args=args,
        )
        target_image = Image.open(bootstrap_record["image_path"]).convert("RGB")
        best_candidate = None
        print(
            f"Updating {bootstrap_base_name} at end with inlier anchors: "
            + ", ".join(anchor["base_name"] for anchor in anchors)
        )
        for anchor_record in anchors:
            bootstrap_update_summary["attempted"] += 1
            anchor_image = Image.open(anchor_record["image_path"]).convert("RGB")
            (
                candidate_mask,
                candidate_bbox,
                paired_image,
                right_offset,
            ) = run_paired_sam_prediction(
                processor=processor,
                anchor_image=anchor_image,
                anchor_bbox=anchor_record["bbox"],
                target_image=target_image,
                prompt=prompt,
                device=device,
                min_area=args.min_contour_area,
                sam3_candidate_masks_dir=sam3_candidate_masks_dir,
                sam3_selected_masks_dir=sam3_selected_masks_dir,
                sam3_debug_base_name=(
                    f"{bootstrap_base_name}_final_anchor_"
                    f"{anchor_record['base_name']}"
                ),
            )
            save_pair_bbox_visualization(
                paired_image=paired_image,
                left_bbox=anchor_record["bbox"],
                right_bbox=candidate_bbox,
                right_offset=right_offset,
                output_dir=paired_images_dir,
                base_name=(
                    f"{bootstrap_base_name}_final_anchor_"
                    f"{anchor_record['base_name']}"
                ),
                prompt_color=(0, 255, 0),
                result_color=args.overlay_color,
            )
            if candidate_mask is None or not candidate_mask.any():
                continue

            candidate_prompt_bbox = get_mask_bbox(
                candidate_mask, padding=args.bbox_padding
            )
            candidate_record = create_prediction_record(
                base_name=bootstrap_base_name,
                frame_idx=bootstrap_record["frame_idx"],
                image_path=bootstrap_record["image_path"],
                image=target_image,
                mask=candidate_mask,
                bbox=candidate_prompt_bbox,
                args=args,
                source="final_bootstrap_pair",
                appearance_extractor=appearance_extractor,
                anchor_name=anchor_record["base_name"],
            )
            candidate_score = score_embedding_against_inliers(
                embedding=candidate_record["object_embedding"],
                inlier_records=final_inlier_records,
                top_k=args.appearance_top_k,
            )
            candidate_record["appearance_score"] = candidate_score
            patch_passed = True
            if args.embedding_backend == "dinov3":
                mean_score, bad_fraction = score_patches_against_bank(
                    patch_embeddings=candidate_record.get("patch_embeddings"),
                    patch_bank=patch_bank,
                    similarity_threshold=args.dinov3_patch_similarity_threshold,
                )
                candidate_record["patch_mean_score"] = mean_score
                candidate_record["patch_bad_fraction"] = bad_fraction
                patch_passed = (
                    mean_score is not None
                    and mean_score >= args.dinov3_min_patch_mean_score
                    and bad_fraction <= args.dinov3_max_bad_patch_fraction
                )
            threshold_passed = candidate_score is not None and (
                consistency_summary["threshold"] is None
                or candidate_score >= consistency_summary["threshold"]
            )
            candidate_record["inlier"] = (
                candidate_record["mask_sane"] and threshold_passed and patch_passed
            )
            if not candidate_record["mask_sane"]:
                candidate_record["rejection_reasons"].append("mask_sanity")
            if not threshold_passed:
                candidate_record["rejection_reasons"].append("appearance")
            if not patch_passed:
                candidate_record["rejection_reasons"].append("dinov3_patches")

            best_score = (
                best_candidate["appearance_score"]
                if best_candidate is not None
                and best_candidate["appearance_score"] is not None
                else -np.inf
            )
            candidate_sort_score = (
                candidate_score if candidate_score is not None else -np.inf
            )
            if best_candidate is None:
                best_candidate = candidate_record
            elif candidate_record["mask_sane"] and not best_candidate["mask_sane"]:
                best_candidate = candidate_record
            elif candidate_record["mask_sane"] == best_candidate["mask_sane"] and (
                candidate_sort_score > best_score
            ):
                best_candidate = candidate_record

        if best_candidate is None:
            bootstrap_update_summary["reason"] = "no_sam3_candidate"
            print(f"No SAM3 candidate found for final {bootstrap_base_name} update.")
        elif not best_candidate["mask_sane"]:
            bootstrap_update_summary["reason"] = "best_candidate_failed_mask_sanity"
            print(
                f"Best final {bootstrap_base_name} candidate failed mask sanity; "
                "leaving threshold mask in place."
            )
        else:
            replacement_bbox = best_candidate["bbox"]
            if replacement_bbox is not None:
                save_bbox_visualization(
                    image=target_image,
                    bbox=replacement_bbox,
                    output_dir=bbox_images_dir,
                    base_name=bootstrap_base_name,
                    color=args.overlay_color,
                )
            save_outputs(
                image=target_image,
                combined_mask=best_candidate["mask"],
                output_dirs=output_dirs,
                base_name=bootstrap_base_name,
                overlay_color=args.overlay_color,
                overlay_alpha=args.overlay_alpha,
            )
            record_by_name[bootstrap_base_name] = best_candidate
            records = [record_by_name[record["base_name"]] for record in records]
            bootstrap_update_summary.update(
                {
                    "updated": True,
                    "anchor_name": best_candidate["anchor_name"],
                    "reason": "updated_from_inlier_anchor_pair",
                }
            )
            print(
                f"Updated {bootstrap_base_name} from inlier anchor "
                f"{best_candidate['anchor_name']}."
            )

    summary = {
        "embedding": appearance_extractor.name
        if appearance_extractor is not None
        else None,
        "consistency_equation": (
            "a_i = mean(top_k({z_i^T z_j | j != i})); "
            "inlier if a_i >= median(a) - tau * 1.4826 * "
            "median(|a_i - median(a)|)"
        ),
        "patch_consistency_equation": (
            "For DINOv3, each masked patch p is scored as "
            "max_q p^T q over trusted inlier object patches q; reject if "
            "mean score is too low or the fraction below threshold is too high."
        )
        if args.embedding_backend == "dinov3"
        else None,
        "consistency": consistency_summary,
        "patch_consistency": patch_summary,
        "num_predictions": len(records),
        "num_inliers": sum(record["inlier"] for record in records),
        "num_outliers": sum(not record["inlier"] for record in records),
        "rerun": rerun_summary,
        "bootstrap_update": bootstrap_update_summary,
    }
    write_appearance_diagnostics(args.output_dir, records, summary)


if __name__ == "__main__":
    process_images()
