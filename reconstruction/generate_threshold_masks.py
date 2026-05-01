import argparse
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

    mask_img.save(mask_out_path)
    image_rgba.save(masked_out_path)
    overlay_img.save(overlay_out_path)
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
    bbox_image.save(bbox_out_path)
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
    bbox_image.save(pair_out_path)
    print(f"  Saved paired bbox visualization: {pair_out_path}")


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
        "--sam_threshold",
        type=float,
        default=0.3,
        help="Deprecated compatibility argument. Use --sam_output_prob_thresh.",
    )
    parser.add_argument(
        "--sam_mask_threshold",
        type=float,
        default=0.5,
        help="Deprecated compatibility argument. Use --sam_output_prob_thresh.",
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    masks_dir = os.path.join(args.output_dir, "masks")
    masked_images_dir = os.path.join(args.output_dir, "masked_images")
    overlay_images_dir = os.path.join(args.output_dir, "overlay_images")
    bbox_images_dir = os.path.join(args.output_dir, "bbox_images")
    paired_images_dir = os.path.join(args.output_dir, "paired_images")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masked_images_dir, exist_ok=True)
    os.makedirs(overlay_images_dir, exist_ok=True)
    os.makedirs(bbox_images_dir, exist_ok=True)
    os.makedirs(paired_images_dir, exist_ok=True)

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
    bootstrap_base_name = os.path.splitext(os.path.basename(bootstrap_path))[0]
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
        output_dir=args.output_dir,
        base_name=bootstrap_base_name,
        color=args.overlay_color,
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
        paired_image, right_offset = create_concatenated_pair(
            bootstrap_image,
            image,
        )
        prompt_bbox = bbox_xyxy_to_normalized_cxcywh(bbox, paired_image.size)
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

        combined_mask, tracked_bbox = pick_right_side_mask(
            outputs=outputs,
            right_offset=right_offset,
            right_image_size=image.size,
            min_area=args.min_contour_area,
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
            print(f"  No right-side SAM3 mask found in {filename}. Skipping...")
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


if __name__ == "__main__":
    process_images()
