import os

import cv2
import numpy as np

VALID_SHARPNESS_METHODS = ("laplacian", "tenengrad")
VALID_SHARPNESS_SELECTION_MODES = ("image", "window")
VALID_SHARPNESS_WINDOW_WEIGHTS = ("uniform", "gaussian")


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
    image_files: list[str], method: str, scan_name: str | None = None
) -> tuple[int, list[float]]:
    scores = compute_sharpness_scores(image_files, method)
    sharpest_idx = int(np.argmax(scores))
    prefix = f"{scan_name}: " if scan_name is not None else ""
    print(
        f"{prefix}sharpest image is index {sharpest_idx} "
        f"({image_files[sharpest_idx]}) with {method} score {scores[sharpest_idx]:.6f}."
    )
    return sharpest_idx, scores


def find_sharpest_window_reference_index(
    image_files: list[str],
    method: str,
    window_size: int | None,
    window_weights: str = "flat",
    gaussian_sigma: float | None = None,
    scan_name: str | None = None,
) -> tuple[int, list[float], list[int], list[float]]:
    scores = compute_sharpness_scores(image_files, method)
    if window_weights not in VALID_SHARPNESS_WINDOW_WEIGHTS:
        raise ValueError(
            f"sharpness_window_weights must be one of "
            f"{VALID_SHARPNESS_WINDOW_WEIGHTS}, got {window_weights!r}."
        )
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

    kernel = _build_window_kernel(actual_window_size, window_weights, gaussian_sigma)
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
        f"ends at index {selected_indices[-1]}, has {window_weights}-weighted "
        f"{method} score "
        f"{best_score:.6f}, and uses reference index {reference_idx} "
        f"({image_files[reference_idx]})."
    )
    return reference_idx, scores, selected_indices, window_scores.tolist()


def _build_window_kernel(
    window_size: int, window_weights: str, gaussian_sigma: float | None
) -> np.ndarray:
    if window_weights == "flat":
        return np.ones(window_size, dtype=np.float64)

    if gaussian_sigma is None:
        gaussian_sigma = window_size / 6.0
    if gaussian_sigma <= 0:
        raise ValueError(
            f"sharpness_gaussian_sigma must be positive or null, got {gaussian_sigma}."
        )

    x = np.arange(window_size, dtype=np.float64) - (window_size - 1) / 2.0
    kernel = np.exp(-0.5 * (x / gaussian_sigma) ** 2)
    return kernel / kernel.sum()


def save_sharpness_bar_chart(
    scores: list[float],
    output_path: str,
    reference_idx: int,
    selected_indices: list[int],
    title: str | None = None,
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

    fig, ax = plt.subplots(figsize=(10, 5))
    indices = list(range(len(scores)))
    ax.bar(indices, scores, color=colors)
    ax.set_xlabel("Image index")
    ax.set_ylabel("Sharpness score")
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(indices)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
