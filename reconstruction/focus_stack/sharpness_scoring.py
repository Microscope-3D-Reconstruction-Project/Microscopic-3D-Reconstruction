import os

import cv2
import numpy as np
from scipy.ndimage import median_filter as scipy_median_filter
from scipy.optimize import curve_fit

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


def _gaussian(x: np.ndarray, amp: float, mu: float, sigma: float, offset: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def _peak_and_baseline(scores: list[float]) -> tuple[int, float, float]:
    """Return (peak_idx, peak_value, baseline_value)."""
    arr = np.array(scores, dtype=float)
    peak_idx = int(np.argmax(arr))
    baseline = float(np.percentile(arr, 10))
    return peak_idx, float(arr[peak_idx]), baseline


def _clamp_window(start: int, end: int, n: int, min_f: int, max_f: int) -> tuple[int, int]:
    """Ensure window respects min/max size and stays in bounds [0, n)."""
    size = end - start
    if size < min_f:
        center = (start + end) // 2
        half = min_f // 2
        start = max(0, center - half)
        end = start + min_f
        if end > n:
            end = n
            start = max(0, end - min_f)
    if (end - start) > max_f:
        center = (start + end) // 2
        half = max_f // 2
        start = max(0, center - half)
        end = min(n, start + max_f)
    return start, end


def smooth_scores(scores: list[float], window: int = 3) -> list[float]:
    """1-D median filter over the score sequence to remove isolated spikes.

    Edge frames are temporarily replaced by their nearest interior neighbour
    before filtering so a noisy edge frame cannot dominate its own window.
    """
    if len(scores) < 2:
        return list(scores)
    arr = np.array(scores, dtype=float)
    arr[0] = arr[1]
    arr[-1] = arr[-2]
    return scipy_median_filter(arr, size=window).tolist()


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


def find_sharpest_window_indices(
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


def find_gaussian_fit_window_indices(
    scores: list[float],
    n_sigma: float = 1.0,
    min_frames: int = 3,
    max_frames: int = 20,
    scan_name: str | None = None,
) -> tuple[int, list[int], list[float]]:
    """Fit a Gaussian to the score profile and select ±n_sigma around the mean.

    Returns (reference_idx, selected_indices, fitted_values) where fitted_values
    is the evaluated Gaussian over all frames (same length as scores).
    Falls back to find_sharpest_window_indices if the fit fails or produces an
    implausible result.
    """
    arr = np.array(scores, dtype=float)
    n = len(arr)
    x = np.arange(n, dtype=float)
    peak_idx, peak_val, baseline = _peak_and_baseline(scores)
    prefix = f"{scan_name}: " if scan_name is not None else ""

    try:
        p0 = [peak_val - baseline, float(peak_idx), n / 6.0, baseline]
        bounds = ([0, 0, 0.5, 0], [np.inf, n, n, peak_val])
        popt, _ = curve_fit(_gaussian, x, arr, p0=p0, bounds=bounds, maxfev=2000)
        _, mu, sigma, _ = popt

        if not (0 <= mu <= n) or sigma <= 0 or sigma > n:
            raise ValueError("implausible fit")

        start, end = _clamp_window(
            max(0, int(np.floor(mu - n_sigma * sigma))),
            min(n, int(np.ceil(mu + n_sigma * sigma))),
            n, min_frames, max_frames,
        )
        selected_indices = list(range(start, end))
        reference_idx = selected_indices[(end - start) // 2]
        fitted_values = _gaussian(x, *popt).tolist()

        print(
            f"{prefix}Gaussian fit: mu={mu:.2f}, sigma={sigma:.2f}, "
            f"window [{start}, {end}), reference index {reference_idx}."
        )
        return reference_idx, selected_indices, fitted_values

    except Exception:
        print(
            f"{prefix}Gaussian fit failed, falling back to fixed-window selection."
        )
        return find_sharpest_window_indices(scores, max_frames, scan_name=scan_name)


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
