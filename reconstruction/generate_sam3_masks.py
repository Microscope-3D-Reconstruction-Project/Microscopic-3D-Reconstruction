import argparse
import os

import numpy as np
import torch

from PIL import Image
from transformers import Sam3Model, Sam3Processor


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


def process_images():
    # Setup command-line arguments
    parser = argparse.ArgumentParser(
        description="Batch process images using SAM 3 and a text prompt."
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
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for segmentation (e.g., 'dog', 'car').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for segmentation.",
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

    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    masks_dir = os.path.join(args.output_dir, "masks")
    masked_images_dir = os.path.join(args.output_dir, "masked_images")
    overlay_images_dir = os.path.join(args.output_dir, "overlay_images")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masked_images_dir, exist_ok=True)
    os.makedirs(overlay_images_dir, exist_ok=True)

    # Determine device (Use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load SAM 3 Model and Processor from Hugging Face
    print("Loading SAM 3 model and processor...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Iterate over all images in the input directory
    for filename in os.listdir(args.input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            continue

        img_path = os.path.join(args.input_dir, filename)
        print(f"Processing: {filename}")

        try:
            # Read image and convert to RGB
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # Prepare inputs for the model
        inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(
            device
        )

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process results to get binary masks
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results["masks"]

        if len(masks) == 0:
            print(
                f"  No objects found for prompt '{args.prompt}' in {filename}. Skipping..."
            )
            continue

        # Combine all found instance masks into a single global mask using Logical OR
        combined_mask = masks.any(dim=0).cpu().numpy()

        # Convert the boolean mask to a grayscale PIL Image (0 or 255)
        mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8), mode="L")

        # Create a masked version of the original image (transparent background)
        image_rgba = image.convert("RGBA")
        image_rgba.putalpha(mask_img)
        overlay_img = create_overlay_image(
            image=image,
            combined_mask=combined_mask,
            overlay_color=tuple(args.overlay_color),
            alpha=args.overlay_alpha,
        )

        # Construct output filenames
        base_name = os.path.splitext(filename)[0]
        mask_out_path = os.path.join(masks_dir, f"{base_name}.png")
        # mask_out_path = os.path.join(masks_dir, f"{base_name}_mask.png")
        masked_out_path = os.path.join(masked_images_dir, f"{base_name}.png")
        # masked_out_path = os.path.join(masked_images_dir, f"{base_name}_masked.png")
        overlay_out_path = os.path.join(overlay_images_dir, f"{base_name}.png")
        # overlay_out_path = os.path.join(overlay_images_dir, f"{base_name}_overlay.png")

        # Save results
        mask_img.save(mask_out_path)
        image_rgba.save(masked_out_path)
        overlay_img.save(overlay_out_path)
        print(f"  Saved mask: {mask_out_path}")
        print(f"  Saved masked image: {masked_out_path}")
        print(f"  Saved overlay image: {overlay_out_path}")


if __name__ == "__main__":
    process_images()
