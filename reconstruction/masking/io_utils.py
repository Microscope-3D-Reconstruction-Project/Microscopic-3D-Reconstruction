import json
import os

import numpy as np

from PIL import Image, ImageDraw


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


def write_bootstrap_debug_metadata(output_dir, metadata):
    """Save threshold bootstrap prompt metadata for later inspection."""
    metadata_path = os.path.join(output_dir, "bootstrap_debug", "bootstrap_bbox.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote bootstrap debug metadata: {metadata_path}")
