import os

import cv2
import numpy as np

VALID_SHARPNESS_METHODS = ("laplacian", "tenengrad")
VALID_SHARPNESS_SELECTION_MODES = ("image", "fixed_window", "gaussian_fit_window")


def _load_grayscale_image(image_file: str) -> np.ndarray:
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image for sharpness scoring: {image_file}")
    return image


def _score_laplacian(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return float(cv2.Laplacian(blurred, cv2.CV_64F).var())


def _score_tenengrad(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude_sq = sobel_x**2 + sobel_y**2
    return float(np.mean(gradient_magnitude_sq))


def compute_sharpness_scores(image_files: list[str], method: str) -> list[float]:
    if not image_files:
        raise ValueError("Cannot compute sharpness scores for an empty image list.")

    if method not in VALID_SHARPNESS_METHODS:
        raise ValueError(
            f"sharpness_score must be one of {VALID_SHARPNESS_METHODS}, got {method!r}."
        )

    scores = []
    for image_file in image_files:
        gray = _load_grayscale_image(image_file)
        if method == "laplacian":
            score = _score_laplacian(gray)
        else:
            score = _score_tenengrad(gray)
        scores.append(score)
    return scores


def find_sharpest_image_index(
    scores: list[float], scan_name: str | None = None
) -> int:
    sharpest_idx = int(np.argmax(scores))
    prefix = f"{scan_name}: " if scan_name is not None else ""
    print(
        f"{prefix}sharpest image is index {sharpest_idx} "
        f"with score {scores[sharpest_idx]:.6f}."
    )
    return sharpest_idx


def find_sharpest_window_reference_index(
    scores: list[float],
    window_size: int | None,
    scan_name: str | None = None,
) -> tuple[int, list[int], list[float]]:
    actual_window_size = (
        len(scores) if window_size is None else min(window_size, len(scores))
    )
    if actual_window_size <= 0:
        raise ValueError("Sharpness window size must be positive.")
    if window_size is not None and actual_window_size < window_size:
        prefix = f"{scan_name}: " if scan_name is not None else ""
        print(
            f"Warning: {prefix}requested sharpness window size {window_size}, "
            f"but scan has {len(scores)} images. Using {actual_window_size} "
            "available images instead."
        )

    kernel = np.ones(actual_window_size, dtype=np.float64)
    window_scores = np.convolve(
        np.asarray(scores, dtype=np.float64), kernel, mode="valid"
    )
    best_start_idx = int(np.argmax(window_scores))
    best_score = float(window_scores[best_start_idx])
    selected_indices = list(range(best_start_idx, best_start_idx + actual_window_size))
    reference_idx = selected_indices[actual_window_size // 2]

    prefix = f"{scan_name}: " if scan_name is not None else ""
    print(
        f"{prefix}sharpest window starts at index {best_start_idx}, "
        f"ends at index {selected_indices[-1]}, "
        f"score {best_score:.6f}, reference index {reference_idx}."
    )
    return reference_idx, selected_indices, window_scores.tolist()


def find_gaussian_fit_window_reference_index(
    scores: list[float],
    scan_name: str | None = None,
) -> tuple[int, list[int], list[float]]:
    raise NotImplementedError(
        "gaussian_fit_window selection mode is not yet implemented."
    )


def save_sharpness_bar_chart(
    scores: list[float],
    output_path: str,
    reference_idx: int,
    selected_indices: list[int],
    title: str | None = None,
    scores_preprocessed: list[float] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    selected_set = set(selected_indices)
    colors = []
    for idx in range(len(scores)):
        if idx == reference_idx:
            colors.append("#2ca02c")
        elif idx in selected_set:
            colors.append("#1f77b4")
        else:
            colors.append("#b0b0b0")

    has_preprocessed = scores_preprocessed is not None
    if has_preprocessed:
        fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(18, 5))
    else:
        fig, ax_before = plt.subplots(figsize=(10, 5))

    indices = list(range(len(scores)))
    ax_before.bar(indices, scores, color=colors)
    ax_before.set_xlabel("Image index")
    ax_before.set_ylabel("Sharpness score")
    ax_before.set_title("Before pre-processing" if has_preprocessed else (title or ""))
    ax_before.set_xticks(indices)
    ax_before.grid(axis="y", alpha=0.3)

    if has_preprocessed:
        ax_after.bar(indices, scores_preprocessed, color=colors)
        ax_after.set_xlabel("Image index")
        ax_after.set_ylabel("Sharpness score")
        ax_after.set_title("After pre-processing")
        ax_after.set_xticks(indices)
        ax_after.grid(axis="y", alpha=0.3)
        if title is not None:
            fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
