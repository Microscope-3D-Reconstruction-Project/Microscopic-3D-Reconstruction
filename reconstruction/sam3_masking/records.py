import numpy as np

from .mask_utils import get_mask_bbox, mask_quality_stats, passes_mask_sanity


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
        "initial_prediction": None,
        "rerun_attempts": [],
        "replacement_reason": None,
    }


def add_rejection_reason(record, reason):
    """Append a rejection reason once, preserving reason order."""
    if reason not in record["rejection_reasons"]:
        record["rejection_reasons"].append(reason)


def apply_record_decision(
    record,
    appearance_passed=True,
    mutual_passed=True,
    patch_passed=True,
    missing_embedding=False,
):
    """Set final inlier state from the standard consistency gates."""
    record["rejection_reasons"] = []
    if not record["mask_sane"]:
        add_rejection_reason(record, "mask_sanity")
    if missing_embedding:
        add_rejection_reason(record, "missing_embedding")
    if not appearance_passed:
        add_rejection_reason(record, "appearance")
    if not mutual_passed:
        add_rejection_reason(record, "mutual_neighbors")
    if not patch_passed:
        add_rejection_reason(record, "dinov3_patches")
    record["inlier"] = not record["rejection_reasons"]


def serializable_record(record, include_audit=True):
    """Drop arrays from a prediction record so it can be written as JSON."""
    data = {
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
    if include_audit:
        initial_prediction = record.get("initial_prediction")
        initial_state = initial_prediction if initial_prediction is not None else data
        data["initial_inlier"] = initial_state["inlier"]
        data["initial_rejection_reasons"] = initial_state["rejection_reasons"]
        data["final_inlier"] = data["inlier"]
        data["initial_prediction"] = record.get("initial_prediction")
        data["rerun_attempts"] = record.get("rerun_attempts", [])
        data["replacement_reason"] = record.get("replacement_reason")
    return data


def serializable_missing_attempt(target_record, anchor_record, source, attempt_index):
    """Create an audit row for an anchor attempt that returned no target mask."""
    return {
        "attempt_index": attempt_index,
        "selected": False,
        "frame_idx": target_record["frame_idx"],
        "image_path": target_record["image_path"],
        "source": source,
        "anchor_name": anchor_record["base_name"],
        "raw_bbox": None,
        "bbox": None,
        "stats": None,
        "mask_sane": False,
        "appearance_score": None,
        "patch_mean_score": None,
        "patch_bad_fraction": None,
        "mutual_neighbor_count": 0,
        "inlier": False,
        "rejection_reasons": ["missing_sam3_mask"],
    }


def serializable_candidate_attempt(record, attempt_index, selected=False):
    """Create an audit row for a scored rerun/final-update candidate."""
    data = serializable_record(record, include_audit=False)
    data["attempt_index"] = attempt_index
    data["selected"] = selected
    return data


def attach_replacement_audit(replacement, original, attempts, reason, selected_index):
    """Keep the original rejected record and all attempts on the replacement."""
    for attempt in attempts:
        attempt["selected"] = attempt["attempt_index"] == selected_index
    replacement["initial_prediction"] = serializable_record(
        original, include_audit=False
    )
    replacement["rerun_attempts"] = attempts
    replacement["replacement_reason"] = reason
    return replacement


def appearance_sort_score(record):
    """Return a sortable score for candidate selection."""
    if record is None or record["appearance_score"] is None:
        return -np.inf
    return record["appearance_score"]


def prefer_inlier_candidate(candidate, best_candidate):
    """Prefer inlier rerun candidates, then the strongest appearance score."""
    if best_candidate is None:
        return True
    if candidate["inlier"] != best_candidate["inlier"]:
        return candidate["inlier"]
    return appearance_sort_score(candidate) > appearance_sort_score(best_candidate)


def prefer_mask_sane_candidate(candidate, best_candidate):
    """Prefer sane bootstrap-update candidates, then appearance score."""
    if best_candidate is None:
        return True
    if candidate["mask_sane"] != best_candidate["mask_sane"]:
        return candidate["mask_sane"]
    return appearance_sort_score(candidate) > appearance_sort_score(best_candidate)
