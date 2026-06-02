import os
import re

import cv2
import numpy as np

from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _natural_image_sort_key(path):
    """Return a natural-sort key for image filenames with embedded digits."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = re.split(r"(\d+)", stem)
    normalized_parts = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            normalized_parts.append((0, int(part)))
        else:
            normalized_parts.append((1, part.lower()))
    return normalized_parts, os.path.basename(path).lower()


def create_foreground_mask(
    image_rgb,
    min_contour_area,
    morph_kernel_size,
    keep_largest,
):
    """Segment foreground using Otsu thresholding, contour detection, and convex hull.

    Uses Otsu's method to automatically find the binary threshold, then finds
    all contours, selects the one whose centroid is closest to the image center,
    and returns its convex hull as the mask together with all contours for debug
    visualization.
    """
    image_np = np.array(image_rgb)
    h, w = image_np.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    all_contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in all_contours if cv2.contourArea(c) >= min_contour_area]

    if not contours:
        return np.zeros((h, w), dtype=bool), 0, list(all_contours), None

    def _centroid_dist(c):
        M = cv2.moments(c)
        if M["m00"] == 0:
            return float("inf")
        return (M["m10"] / M["m00"] - cx) ** 2 + (M["m01"] / M["m00"] - cy) ** 2

    best = min(contours, key=_centroid_dist)
    hull = cv2.convexHull(best)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)
    return mask > 0, 1, contours, hull


def get_mask_bbox(mask, padding=0):
    """Return an xyxy bounding box around nonzero mask pixels."""
    y_indices, x_indices = np.nonzero(mask)
    if len(x_indices) == 0:
        return None

    height, width = mask.shape
    x_min = max(0, int(x_indices.min()) - padding)
    y_min = max(0, int(y_indices.min()) - padding)
    x_max = min(width - 1, int(x_indices.max()) + padding)
    y_max = min(height - 1, int(y_indices.max()) + padding)
    return [x_min, y_min, x_max, y_max]


def bbox_area(bbox):
    """Return inclusive xyxy bbox area in pixels."""
    if bbox is None:
        return 0
    x_min, y_min, x_max, y_max = bbox
    return max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)


def keep_largest_component(mask, min_area=0):
    """Keep only the largest connected foreground component in a binary mask."""
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    if num_labels <= 1:
        return np.zeros(mask.shape, dtype=bool)

    component_ids = [
        label_id
        for label_id in range(1, num_labels)
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area
    ]
    if not component_ids:
        return np.zeros(mask.shape, dtype=bool)

    largest_id = max(
        component_ids, key=lambda label_id: stats[label_id, cv2.CC_STAT_AREA]
    )
    return labels == largest_id


def count_components(mask, min_area=0):
    """Count connected foreground components at least min_area pixels large."""
    mask_u8 = mask.astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return 0
    return int(
        sum(
            stats[label_id, cv2.CC_STAT_AREA] >= min_area
            for label_id in range(1, num_labels)
        )
    )


def list_input_images(input_dir):
    """Return sorted image paths from a flat directory."""
    image_paths = []
    for filename in os.listdir(input_dir):
        path = os.path.join(input_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        if os.path.isfile(path) and ext in VALID_EXTENSIONS:
            image_paths.append(path)
    return sorted(image_paths, key=_natural_image_sort_key)


def find_bootstrap_image(image_paths, bootstrap_stem):
    """Find the scan00 image, falling back to the first sorted image."""
    for image_path in image_paths:
        if os.path.splitext(os.path.basename(image_path))[0] == bootstrap_stem:
            return image_path
    return image_paths[0] if image_paths else None


def get_odd_kernel_size(kernel_size):
    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def load_valid_focus_region(mask_path, border_padding):
    if mask_path is None:
        return None

    mask = np.array(Image.open(mask_path).convert("L")) > 0
    if not mask.any():
        return None

    border_padding = max(0, int(border_padding))
    if border_padding == 0:
        return mask

    kernel_size = border_padding * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    eroded = cv2.erode((mask.astype(np.uint8)) * 255, kernel, iterations=1)
    return eroded > 0


def create_blurred_edge_mask(
    image_rgb,
    blur_kernel_size,
    canny_threshold1,
    canny_threshold2,
    valid_region=None,
):
    image_np = np.array(image_rgb)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    kernel_size = get_odd_kernel_size(blur_kernel_size)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    mask = edges > 0
    if valid_region is not None:
        if valid_region.shape != mask.shape:
            raise ValueError("Focus-stack mask shape does not match the input image shape.")
        mask &= valid_region
    return mask, count_components(mask)


def scale_mask_toward_centroid(mask, scale):
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)

    scale = float(scale)
    if scale <= 0.0:
        return np.zeros(mask.shape, dtype=bool)
    if scale >= 1.0:
        return mask.copy()

    coords_yx = np.column_stack(np.nonzero(mask)).astype(np.float32)
    centroid_yx = coords_yx.mean(axis=0)
    scaled_yx = centroid_yx + scale * (coords_yx - centroid_yx)
    scaled_yx = np.rint(scaled_yx).astype(np.int32)

    height, width = mask.shape
    scaled_yx[:, 0] = np.clip(scaled_yx[:, 0], 0, height - 1)
    scaled_yx[:, 1] = np.clip(scaled_yx[:, 1], 0, width - 1)

    scaled_mask = np.zeros(mask.shape, dtype=bool)
    scaled_mask[scaled_yx[:, 0], scaled_yx[:, 1]] = True
    return scaled_mask


def _fill_mask_holes(mask):
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)

    mask_u8 = (mask.astype(np.uint8)) * 255
    flood_filled = mask_u8.copy()
    height, width = mask_u8.shape
    flood_fill_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood_filled, flood_fill_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood_filled)
    filled = cv2.bitwise_or(mask_u8, holes)
    return filled > 0


def dilate_and_fill_mask(mask, kernel_size, dilate_iterations):
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)

    kernel_size = get_odd_kernel_size(max(1, int(kernel_size)))
    dilate_iterations = max(1, int(dilate_iterations))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = (mask.astype(np.uint8)) * 255
    dilated = cv2.dilate(mask_u8, kernel, iterations=dilate_iterations)
    return _fill_mask_holes(dilated > 0)


def sample_uniform_points_from_mask(mask, num_points):
    coords_yx = np.column_stack(np.nonzero(mask))
    num_points = max(0, int(num_points))
    if num_points == 0 or len(coords_yx) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if len(coords_yx) <= num_points:
        sampled_yx = coords_yx
    else:
        centroid = coords_yx.mean(axis=0)
        first_idx = int(np.argmin(np.sum((coords_yx - centroid) ** 2, axis=1)))
        selected_indices = [first_idx]
        min_sq_dist = np.sum((coords_yx - coords_yx[first_idx]) ** 2, axis=1)

        for _ in range(1, num_points):
            next_idx = int(np.argmax(min_sq_dist))
            selected_indices.append(next_idx)
            next_sq_dist = np.sum((coords_yx - coords_yx[next_idx]) ** 2, axis=1)
            min_sq_dist = np.minimum(min_sq_dist, next_sq_dist)

        sampled_yx = coords_yx[selected_indices]

    return sampled_yx[:, [1, 0]].astype(np.float32)
