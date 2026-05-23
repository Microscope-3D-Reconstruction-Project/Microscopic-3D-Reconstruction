import os
import re

import cv2
import numpy as np

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
