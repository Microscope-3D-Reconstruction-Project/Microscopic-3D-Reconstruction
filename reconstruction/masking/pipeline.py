import os

import numpy as np

from PIL import Image

from .io_utils import (
    save_bbox_visualization,
    save_contour_visualization,
    save_outputs,
    write_bootstrap_debug_metadata,
)
from .mask_utils import (
    create_foreground_mask,
    find_bootstrap_image,
    get_mask_bbox,
    list_input_images,
)
from .sam2_mask_predictor import Sam2MaskPredictor


class Sam2MaskingPipeline:
    """Threshold bootstrap -> SAM2 image refinement -> SAM2 video propagation."""

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        args = self.cfg

        os.makedirs(args.output_dir, exist_ok=True)
        masks_dir = os.path.join(args.output_dir, args.masks_subdir)
        masked_images_dir = os.path.join(args.output_dir, "masked_images")
        overlay_images_dir = os.path.join(args.output_dir, "overlay_images")
        bbox_images_dir = os.path.join(args.output_dir, "bbox_images")
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
        for d in (
            masks_dir,
            masked_images_dir,
            overlay_images_dir,
            bbox_images_dir,
            threshold_masks_dir,
            threshold_masked_images_dir,
            threshold_overlay_images_dir,
            threshold_bbox_images_dir,
        ):
            os.makedirs(d, exist_ok=True)

        # Step 1: Load images and find bootstrap (scan00)
        image_paths = list_input_images(args.input_dir)
        if not image_paths:
            raise FileNotFoundError(f"No image files found in {args.input_dir!r}.")

        bootstrap_path = find_bootstrap_image(image_paths, args.bootstrap_stem)
        if bootstrap_path is None:
            raise FileNotFoundError(f"No bootstrap image found in {args.input_dir!r}.")
        bootstrap_frame_idx = image_paths.index(bootstrap_path)
        bootstrap_base_name = os.path.splitext(os.path.basename(bootstrap_path))[0]

        print(
            f"Bootstrap image: {os.path.basename(bootstrap_path)} "
            f"(frame {bootstrap_frame_idx} of {len(image_paths)})"
        )
        bootstrap_image = Image.open(bootstrap_path).convert("RGB")
        bootstrap_image_np = np.array(bootstrap_image)

        # Step 2: Otsu threshold to get coarse mask + bounding box
        threshold_mask, num_components, contours, hull = create_foreground_mask(
            image_rgb=bootstrap_image,
            min_contour_area=args.min_contour_area,
            morph_kernel_size=args.morph_kernel_size,
            keep_largest=True,
        )
        if not threshold_mask.any():
            raise RuntimeError(
                f"No foreground found in {os.path.basename(bootstrap_path)!r}."
            )
        coarse_bbox = get_mask_bbox(threshold_mask, padding=args.bbox_padding)
        if coarse_bbox is None:
            raise RuntimeError("Could not compute bounding box from threshold mask.")

        print(f"  Threshold components kept: {num_components}")
        print(f"  Coarse bbox xyxy: {coarse_bbox}")

        save_outputs(
            image=bootstrap_image,
            combined_mask=threshold_mask,
            output_dirs=(
                threshold_masks_dir,
                threshold_masked_images_dir,
                threshold_overlay_images_dir,
            ),
            base_name=bootstrap_base_name,
            overlay_color=args.overlay_color,
            overlay_alpha=args.overlay_alpha,
        )
        save_bbox_visualization(
            image=bootstrap_image,
            bbox=coarse_bbox,
            output_dir=threshold_bbox_images_dir,
            base_name=f"{bootstrap_base_name}_threshold_prompt",
            color=args.overlay_color,
        )
        save_contour_visualization(
            image=bootstrap_image,
            contours=contours,
            hull=hull,
            output_dir=threshold_bbox_images_dir,
            base_name=bootstrap_base_name,
        )
        write_bootstrap_debug_metadata(
            output_dir=args.output_dir,
            metadata={
                "bootstrap_image": os.path.basename(bootstrap_path),
                "bootstrap_stem": bootstrap_base_name,
                "threshold_components_kept": num_components,
                "coarse_bbox_xyxy": coarse_bbox,
                "threshold_mask_path": os.path.join(
                    "bootstrap_debug",
                    "threshold_masks",
                    f"{bootstrap_base_name}.png",
                ),
            },
        )

        # Step 3: SAM2 image predictor on scan00 -> precise mask
        print("Running SAM2 image predictor on bootstrap frame...")
        predictor = Sam2MaskPredictor(args)
        precise_mask = predictor.predict_image_from_bbox(
            image_np=bootstrap_image_np,
            box_xyxy=coarse_bbox,
        )
        if not precise_mask.any():
            print(
                "  SAM2 image predictor returned empty mask; "
                "falling back to threshold mask."
            )
            precise_mask = threshold_mask

        precise_bbox = get_mask_bbox(precise_mask, padding=args.bbox_padding)
        print(f"  Precise mask bbox xyxy: {precise_bbox}")

        output_dirs = (masks_dir, masked_images_dir, overlay_images_dir)
        save_outputs(
            image=bootstrap_image,
            combined_mask=precise_mask,
            output_dirs=output_dirs,
            base_name=bootstrap_base_name,
            overlay_color=args.overlay_color,
            overlay_alpha=args.overlay_alpha,
        )
        if precise_bbox is not None:
            save_bbox_visualization(
                image=bootstrap_image,
                bbox=precise_bbox,
                output_dir=bbox_images_dir,
                base_name=bootstrap_base_name,
                color=args.overlay_color,
            )

        if len(image_paths) == 1:
            print("Single image - skipping video propagation.")
            return

        # Step 4: SAM2 video predictor propagates mask across all frames
        print("Running SAM2 video propagation across all frames...")
        frame_masks = predictor.predict_video_from_mask(
            image_paths=image_paths,
            bootstrap_frame_idx=bootstrap_frame_idx,
            bootstrap_mask=precise_mask,
            offload_to_cpu=args.offload_video_to_cpu,
        )

        # Step 5: Save per-frame mask outputs
        print("Saving per-frame mask outputs...")
        for frame_idx, img_path in enumerate(image_paths):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask = frame_masks.get(frame_idx)
            if mask is None:
                print(
                    f"  Warning: no mask for frame {frame_idx} ({base_name}); skipping."
                )
                continue

            image = Image.open(img_path).convert("RGB")
            save_outputs(
                image=image,
                combined_mask=mask,
                output_dirs=output_dirs,
                base_name=base_name,
                overlay_color=args.overlay_color,
                overlay_alpha=args.overlay_alpha,
            )
            frame_bbox = get_mask_bbox(mask, padding=args.bbox_padding)
            if frame_bbox is not None:
                save_bbox_visualization(
                    image=image,
                    bbox=frame_bbox,
                    output_dir=bbox_images_dir,
                    base_name=base_name,
                    color=args.overlay_color,
                )

        print(f"Masking complete. Masks saved to: {masks_dir}")
