import json
import os

import numpy as np

from PIL import Image, ImageDraw

from .mask_utils import output_masks_to_numpy
from .records import serializable_record


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


def save_contour_visualization(image, contours, hull, output_dir, base_name):
    """Save a debug image showing all contours (cyan) and the selected convex hull (green)."""
    import cv2

    vis = np.array(image.convert("RGB"))
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)  # all contours: cyan
    if hull is not None:
        cv2.drawContours(vis, [hull], -1, (0, 255, 0), 3)  # selected hull: green
    out_path = os.path.join(output_dir, f"{base_name}_contours.png")
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(vis).save(out_path, format="PNG")
    print(f"  Saved contour visualization: {out_path}")


def save_bbox_visualization(image, bbox, output_dir, base_name, color=(0, 255, 0)):
    """Save an image with the prompt bbox drawn on top."""
    bbox_image = image.copy()
    draw = ImageDraw.Draw(bbox_image)
    line_width = max(2, round(min(image.size) * 0.004))
    draw.rectangle(bbox, outline=tuple(color), width=line_width)

    bbox_out_path = os.path.join(output_dir, f"{base_name}_bbox.png")
    bbox_image.save(bbox_out_path, format="PNG")
    print(f"  Saved bbox visualization: {bbox_out_path}")


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
