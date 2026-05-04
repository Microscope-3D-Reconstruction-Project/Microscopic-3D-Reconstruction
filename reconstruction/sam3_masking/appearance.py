import numpy as np
import torch

from PIL import Image

from .mask_utils import l2_normalize
from .records import add_rejection_reason, apply_record_decision


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
    """Construct the DINOv3 appearance embedding backend when enabled."""
    if not args.appearance_filter:
        return None
    return DinoV3PatchAppearanceExtractor(args)


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
            apply_record_decision(
                record,
                missing_embedding=record["object_embedding"] is None,
            )
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
        apply_record_decision(
            record,
            appearance_passed=appearance_inlier,
            mutual_passed=mutual_inlier,
        )

    for idx, record in enumerate(records):
        if idx in valid_indices:
            continue
        apply_record_decision(record, missing_embedding=True)

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


def evaluate_record_against_inliers(
    record,
    inlier_records,
    appearance_threshold,
    patch_bank,
    args,
):
    """Score one candidate against trusted records and update its inlier state."""
    candidate_score = score_embedding_against_inliers(
        embedding=record["object_embedding"],
        inlier_records=inlier_records,
        top_k=args.appearance_top_k,
    )
    record["appearance_score"] = candidate_score
    missing_embedding = candidate_score is None
    appearance_passed = missing_embedding or (
        appearance_threshold is None or candidate_score >= appearance_threshold
    )

    mean_score, bad_fraction = score_patches_against_bank(
        patch_embeddings=record.get("patch_embeddings"),
        patch_bank=patch_bank,
        similarity_threshold=args.dinov3_patch_similarity_threshold,
    )
    record["patch_mean_score"] = mean_score
    record["patch_bad_fraction"] = bad_fraction
    patch_passed = (
        mean_score is not None
        and mean_score >= args.dinov3_min_patch_mean_score
        and bad_fraction <= args.dinov3_max_bad_patch_fraction
    )

    apply_record_decision(
        record,
        appearance_passed=appearance_passed,
        patch_passed=patch_passed,
        missing_embedding=missing_embedding,
    )
    return record


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
            add_rejection_reason(record, "dinov3_patches")
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
