import os

import numpy as np

from PIL import Image

from .io_utils import (
    save_outputs,
    save_point_visualization,
)
from .mask_utils import (
    create_blurred_edge_mask,
    dilate_and_fill_mask,
    find_bootstrap_image,
    keep_largest_component,
    list_input_images,
    load_valid_focus_region,
    sample_uniform_points_from_mask,
    scale_mask_toward_centroid,
)
from .sam2_mask_predictor import Sam2MaskPredictor


class Sam2MaskingPipeline:
    """Canny edge bootstrap -> point-prompt SAM2 video propagation."""

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        args = self.cfg
        os.makedirs(args.output_dir, exist_ok=True)

        self._run_pass(
            images_dir=args.images_dir,
            masks_dir_name=args.masks_subdir,
            masked_images_dir_name="masked_images",
            overlay_images_dir_name="overlay_images",
            bootstrap_debug_dir_name="bootstrap_debug",
        )

        gt_images_dir = args.gt_images_dir
        if gt_images_dir is None:
            return
        if not os.path.isdir(gt_images_dir):
            print(f"Warning: gt_images_dir={gt_images_dir!r} does not exist; skipping GT masking.")
            return

        print(f"Ground-truth images dir found at {gt_images_dir!r}; masking GT images too.")
        self._run_pass(
            images_dir=gt_images_dir,
            masks_dir_name="gt_masks",
            masked_images_dir_name=args.masked_gt_images_subdir,
            overlay_images_dir_name="gt_overlay_images",
            bootstrap_debug_dir_name="gt_bootstrap_debug",
        )

    def _run_pass(
        self,
        images_dir: str,
        masks_dir_name: str,
        masked_images_dir_name: str,
        overlay_images_dir_name: str,
        bootstrap_debug_dir_name: str,
    ):
        args = self.cfg

        masks_dir = os.path.join(args.output_dir, masks_dir_name)
        masked_images_dir = os.path.join(args.output_dir, masked_images_dir_name)
        overlay_images_dir = os.path.join(args.output_dir, overlay_images_dir_name)
        bootstrap_debug_dir = os.path.join(args.output_dir, bootstrap_debug_dir_name)
        points_debug_dir = os.path.join(bootstrap_debug_dir, "point_images")
        firstpass_masks_dir = os.path.join(bootstrap_debug_dir, "firstpass_masks")
        firstpass_masked_dir = os.path.join(bootstrap_debug_dir, "firstpass_masked_images")
        firstpass_overlays_dir = os.path.join(bootstrap_debug_dir, "firstpass_overlay_images")
        refine_masks_dir = os.path.join(bootstrap_debug_dir, "refine_prompt_masks")
        refine_masked_dir = os.path.join(bootstrap_debug_dir, "refine_prompt_masked_images")
        refine_overlays_dir = os.path.join(bootstrap_debug_dir, "refine_prompt_overlay_images")
        canny_masks_dir = os.path.join(bootstrap_debug_dir, "canny_masks")
        canny_masked_dir = os.path.join(bootstrap_debug_dir, "canny_masked_images")
        canny_overlays_dir = os.path.join(bootstrap_debug_dir, "canny_overlay_images")
        for d in (
            masks_dir,
            masked_images_dir,
            overlay_images_dir,
            points_debug_dir,
            firstpass_masks_dir,
            firstpass_masked_dir,
            firstpass_overlays_dir,
            refine_masks_dir,
            refine_masked_dir,
            refine_overlays_dir,
            canny_masks_dir,
            canny_masked_dir,
            canny_overlays_dir,
            bootstrap_masks_dir,
            bootstrap_masked_dir,
            bootstrap_overlays_dir,
        ):
            os.makedirs(d, exist_ok=True)

        # Step 1: Load images and find bootstrap (scan00)
        image_paths = list_input_images(images_dir)
        if not image_paths:
            raise FileNotFoundError(f"No image files found in {images_dir!r}.")

        bootstrap_path = find_bootstrap_image(image_paths, args.bootstrap_stem)
        if bootstrap_path is None:
            raise FileNotFoundError(f"No bootstrap image found in {images_dir!r}.")
        bootstrap_frame_idx = image_paths.index(bootstrap_path)
        bootstrap_base_name = os.path.splitext(os.path.basename(bootstrap_path))[0]

        print(
            f"Bootstrap image: {os.path.basename(bootstrap_path)} "
            f"(frame {bootstrap_frame_idx} of {len(image_paths)})"
        )
        bootstrap_image = Image.open(bootstrap_path).convert("RGB")
        bootstrap_image_np = np.array(bootstrap_image)

        # Step 2: Load valid focus region for the bootstrap frame (eroded by border padding)
        focus_mask_path = os.path.join(args.valid_regions_dir, os.path.basename(bootstrap_path))
        if not os.path.isfile(focus_mask_path):
            focus_mask_path = None
        valid_region = load_valid_focus_region(focus_mask_path, args.focus_mask_border_padding)
        if focus_mask_path is None:
            print("  Warning: no focus-stack mask found; using the full image for edge detection.")

        # Step 3: Canny edge detection constrained to valid focus region
        print("Running Canny edge detection on bootstrap frame...")
        edge_mask, edge_components = create_blurred_edge_mask(
            image_rgb=bootstrap_image,
            blur_kernel_size=args.edge_blur_kernel_size,
            canny_threshold1=args.canny_threshold1,
            canny_threshold2=args.canny_threshold2,
            valid_region=valid_region,
        )
        print(f"  Edge components: {edge_components}")
        if not edge_mask.any():
            raise RuntimeError(
                f"Canny edge detection found no edges in {os.path.basename(bootstrap_path)!r}."
            )

        save_outputs(
            image=bootstrap_image,
            combined_mask=edge_mask,
            output_dirs=(canny_masks_dir, canny_masked_dir, canny_overlays_dir),
            base_name=bootstrap_base_name,
            overlay_color=args.overlay_color,
            overlay_alpha=args.overlay_alpha,
        )

        # Step 4: Scale edge mask toward centroid to move sample points inward
        scaled_mask = scale_mask_toward_centroid(edge_mask, scale=args.point_sampling_scale)
        if not scaled_mask.any() and edge_mask.any():
            print(
                "  Warning: scaled mask is empty; falling back to raw edge mask for sampling."
            )
            scaled_mask = edge_mask

        # Step 5: Sample uniform points from the scaled mask
        point_prompts = sample_uniform_points_from_mask(scaled_mask, args.point_prompt_count)
        print(f"  Point prompts sampled: {len(point_prompts)}")
        if len(point_prompts) == 0:
            raise RuntimeError("No points could be sampled from the scaled edge mask.")

        save_point_visualization(
            image=bootstrap_image,
            points_xy=point_prompts,
            output_dir=points_debug_dir,
            base_name=bootstrap_base_name,
            color=args.overlay_color,
        )

        # Step 6: First-pass SAM2 image prediction from points
        print("Running first-pass SAM2 image prediction from points...")
        predictor = Sam2MaskPredictor(args)
        first_pass_mask = predictor.predict_image_from_points(
            image_np=bootstrap_image_np,
            points_xy=point_prompts,
        )
        if not first_pass_mask.any():
            print(
                "  Warning: first-pass SAM2 point prompting returned an empty mask; "
                "falling back to the scaled edge mask."
            )
            first_pass_mask = scaled_mask

        save_outputs(
            image=bootstrap_image,
            combined_mask=first_pass_mask,
            output_dirs=(firstpass_masks_dir, firstpass_masked_dir, firstpass_overlays_dir),
            base_name=bootstrap_base_name,
            overlay_color=args.overlay_color,
            overlay_alpha=args.overlay_alpha,
        )

        # Step 7: Keep largest component of first-pass mask, then dilate and fill
        largest_firstpass_mask = keep_largest_component(first_pass_mask)
        if largest_firstpass_mask.any():
            first_pass_mask = largest_firstpass_mask
        else:
            print(
                "  Warning: keep_largest_component returned an empty mask; "
                "keeping the full first-pass mask."
            )

        refine_prompt_mask = dilate_and_fill_mask(
            first_pass_mask,
            kernel_size=args.points_refine_kernel_size,
            dilate_iterations=args.points_refine_dilate_iterations,
        )
        if not refine_prompt_mask.any() and first_pass_mask.any():
            print(
                "  Warning: refine prompt mask is empty after dilation/fill; "
                "falling back to the first-pass mask."
            )
            refine_prompt_mask = first_pass_mask

        save_outputs(
            image=bootstrap_image,
            combined_mask=refine_prompt_mask,
            output_dirs=(refine_masks_dir, refine_masked_dir, refine_overlays_dir),
            base_name=bootstrap_base_name,
            overlay_color=args.overlay_color,
            overlay_alpha=args.overlay_alpha,
        )

        # # Step 8: Second-pass SAM2 image refinement from dilated mask
        # print("Running second-pass SAM2 image refinement from mask...")
        # bootstrap_mask = predictor.predict_image_from_mask(
        #     image_np=bootstrap_image_np,
        #     prompt_mask=refine_prompt_mask,
        # )
        # if not bootstrap_mask.any():
        #     print(
        #         "  Warning: second-pass SAM2 refinement returned an empty mask; "
        #         "falling back to the refine prompt mask."
        #     )
        #     bootstrap_mask = refine_prompt_mask

        # save_outputs(
        #     image=bootstrap_image,
        #     combined_mask=bootstrap_mask,
        #     output_dirs=(bootstrap_masks_dir, bootstrap_masked_dir, bootstrap_overlays_dir),
        #     base_name=bootstrap_base_name,
        #     overlay_color=args.overlay_color,
        #     overlay_alpha=args.overlay_alpha,
        # )

        # Step 9: SAM2 video propagation using the refined bootstrap mask
        print("Running SAM2 video propagation from refined bootstrap mask...")
        frame_masks = predictor.predict_video_from_mask(
            image_paths=image_paths,
            bootstrap_frame_idx=bootstrap_frame_idx,
            bootstrap_mask=refine_prompt_mask,
            offload_to_cpu=args.offload_video_to_cpu,
        )

        # Step 7: Save per-frame mask outputs
        print("Saving per-frame mask outputs...")
        output_dirs = (masks_dir, masked_images_dir, overlay_images_dir)
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

        print(f"Masking complete for {images_dir!r}. Masks saved to: {masks_dir}")
