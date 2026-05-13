import os

import numpy as np
import torch

from PIL import Image

from .appearance import (
    apply_dinov3_patch_consistency,
    build_appearance_extractor,
    build_patch_bank,
    evaluate_record_against_inliers,
    score_nearest_neighbor_consistency,
    select_anchor_records,
)
from .io_utils import (
    save_bbox_visualization,
    save_contour_visualization,
    save_outputs,
    save_pair_bbox_visualization,
    write_appearance_diagnostics,
    write_bootstrap_debug_metadata,
)
from .mask_utils import (
    create_foreground_mask,
    find_bootstrap_image,
    get_mask_bbox,
    list_input_images,
)
from .records import (
    attach_replacement_audit,
    create_prediction_record,
    prefer_inlier_candidate,
    prefer_mask_sane_candidate,
    serializable_candidate_attempt,
    serializable_missing_attempt,
)
from .sam3_predictor import Sam3PairPredictor


class Sam3MaskingPipeline:
    """Run threshold bootstrap, paired SAM3 masking, and DINOv3 consistency checks."""

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        args = self.cfg
        if args.rerun_outliers:
            args.appearance_filter = True

        os.makedirs(args.output_dir, exist_ok=True)
        masks_dir = os.path.join(args.output_dir, args.masks_subdir)
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
        (
            bootstrap_mask,
            num_components,
            bootstrap_contours,
            bootstrap_hull,
        ) = create_foreground_mask(
            image_rgb=bootstrap_image,
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
            raise RuntimeError(
                "Could not compute a bounding box from the bootstrap mask."
            )

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
        save_contour_visualization(
            image=bootstrap_image,
            contours=bootstrap_contours,
            hull=bootstrap_hull,
            output_dir=threshold_bbox_images_dir,
            base_name=bootstrap_base_name,
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

        predictor = Sam3PairPredictor(args)

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
            ) = predictor.predict_pair(
                anchor_image=bootstrap_image,
                anchor_bbox=bbox,
                target_image=image,
                prompt=prompt,
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
                print(
                    f"  No right-side SAM3 mask found in {filename}. Marking outlier..."
                )
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
        initial_num_inliers = num_inliers
        initial_num_outliers = num_outliers
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
                    target_record["replacement_reason"] = "bootstrap_outlier_not_rerun"
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
                    target_record["replacement_reason"] = "no_inlier_anchors"
                    rerun_summary["failed"] += 1
                    continue

                target_image = Image.open(target_record["image_path"]).convert("RGB")
                best_candidate = None
                best_candidate_attempt_index = None
                attempts = []
                attempt_index = 0
                print(
                    f"  Rerunning {target_record['base_name']} with anchors: "
                    + ", ".join(anchor["base_name"] for anchor in anchors)
                )
                for anchor_record in anchors:
                    if anchor_record["bbox"] is None:
                        continue

                    rerun_summary["attempted"] += 1
                    attempt_index += 1
                    anchor_image = Image.open(anchor_record["image_path"]).convert(
                        "RGB"
                    )
                    (
                        candidate_mask,
                        candidate_bbox,
                        paired_image,
                        right_offset,
                    ) = predictor.predict_pair(
                        anchor_image=anchor_image,
                        anchor_bbox=anchor_record["bbox"],
                        target_image=target_image,
                        prompt=prompt,
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
                        attempts.append(
                            serializable_missing_attempt(
                                target_record=target_record,
                                anchor_record=anchor_record,
                                source="rerun_pair",
                                attempt_index=attempt_index,
                            )
                        )
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
                    evaluate_record_against_inliers(
                        record=candidate_record,
                        inlier_records=inlier_records,
                        appearance_threshold=appearance_threshold,
                        patch_bank=patch_bank,
                        args=args,
                    )
                    attempts.append(
                        serializable_candidate_attempt(
                            record=candidate_record,
                            attempt_index=attempt_index,
                        )
                    )

                    if prefer_inlier_candidate(candidate_record, best_candidate):
                        best_candidate = candidate_record
                        best_candidate_attempt_index = attempt_index

                if best_candidate is None or not best_candidate["inlier"]:
                    print(
                        f"    No accepted rerun candidate for "
                        f"{target_record['base_name']}."
                    )
                    target_record["rerun_attempts"] = attempts
                    target_record[
                        "replacement_reason"
                    ] = "rerun_failed_no_inlier_candidate"
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
                attach_replacement_audit(
                    replacement=best_candidate,
                    original=target_record,
                    attempts=attempts,
                    reason="accepted_rerun_after_initial_rejection",
                    selected_index=best_candidate_attempt_index,
                )
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
            best_candidate_attempt_index = None
            attempts = []
            attempt_index = 0
            print(
                f"Updating {bootstrap_base_name} at end with inlier anchors: "
                + ", ".join(anchor["base_name"] for anchor in anchors)
            )
            for anchor_record in anchors:
                bootstrap_update_summary["attempted"] += 1
                attempt_index += 1
                anchor_image = Image.open(anchor_record["image_path"]).convert("RGB")
                (
                    candidate_mask,
                    candidate_bbox,
                    paired_image,
                    right_offset,
                ) = predictor.predict_pair(
                    anchor_image=anchor_image,
                    anchor_bbox=anchor_record["bbox"],
                    target_image=target_image,
                    prompt=prompt,
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
                    attempts.append(
                        serializable_missing_attempt(
                            target_record=bootstrap_record,
                            anchor_record=anchor_record,
                            source="final_bootstrap_pair",
                            attempt_index=attempt_index,
                        )
                    )
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
                evaluate_record_against_inliers(
                    record=candidate_record,
                    inlier_records=final_inlier_records,
                    appearance_threshold=consistency_summary["threshold"],
                    patch_bank=patch_bank,
                    args=args,
                )
                attempts.append(
                    serializable_candidate_attempt(
                        record=candidate_record,
                        attempt_index=attempt_index,
                    )
                )

                if prefer_mask_sane_candidate(candidate_record, best_candidate):
                    best_candidate = candidate_record
                    best_candidate_attempt_index = attempt_index

            if best_candidate is None:
                bootstrap_update_summary["reason"] = "no_sam3_candidate"
                bootstrap_record["rerun_attempts"] = attempts
                bootstrap_record[
                    "replacement_reason"
                ] = "final_bootstrap_no_sam3_candidate"
                print(
                    f"No SAM3 candidate found for final {bootstrap_base_name} update."
                )
            elif not best_candidate["mask_sane"]:
                bootstrap_update_summary["reason"] = "best_candidate_failed_mask_sanity"
                bootstrap_record["rerun_attempts"] = attempts
                bootstrap_record[
                    "replacement_reason"
                ] = "final_bootstrap_best_candidate_failed_mask_sanity"
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
                attach_replacement_audit(
                    replacement=best_candidate,
                    original=bootstrap_record,
                    attempts=attempts,
                    reason="final_bootstrap_update",
                    selected_index=best_candidate_attempt_index,
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
            ),
            "consistency": consistency_summary,
            "patch_consistency": patch_summary,
            "num_predictions": len(records),
            "num_initial_inliers": initial_num_inliers,
            "num_initial_outliers": initial_num_outliers,
            "num_inliers": sum(record["inlier"] for record in records),
            "num_outliers": sum(not record["inlier"] for record in records),
            "rerun": rerun_summary,
            "bootstrap_update": bootstrap_update_summary,
        }
        write_appearance_diagnostics(args.output_dir, records, summary)
