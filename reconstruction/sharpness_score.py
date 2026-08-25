"""
reconstruction/sharpness_score.py

Scores every image in a folder by sharpness and prints a ranked summary.

Sharpness metric: variance of the Laplacian (standard focus measure).
Higher = sharper.

Usage:
    python reconstruction/sharpness_score.py <folder>
    python reconstruction/sharpness_score.py <folder> --ext jpg png
    python reconstruction/sharpness_score.py <folder> --sort score
    python reconstruction/sharpness_score.py <folder> --save scores.csv
"""

import argparse
import csv
import sys

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"}


def laplacian_variance(image_path: Path) -> float:
    """
    Compute the variance of the Laplacian for a single image.
    Converts to grayscale first if needed.
    Returns -1.0 if the image cannot be read.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return -1.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def collect_images(folder: Path, extensions: set[str]) -> list[Path]:
    paths = []
    for ext in extensions:
        paths.extend(folder.glob(f"*.{ext}"))
        paths.extend(folder.glob(f"*.{ext.upper()}"))
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score images in a folder by sharpness (Laplacian variance)."
    )
    parser.add_argument("folder", type=Path, help="Folder containing images.")
    parser.add_argument(
        "--ext",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        metavar="EXT",
        help="Image extensions to include (default: jpg jpeg png bmp tiff tif).",
    )
    parser.add_argument(
        "--sort",
        choices=["name", "score"],
        default="name",
        help="Sort output by filename or score (default: name).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="CSV",
        help="Optional path to save results as a CSV file.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PNG",
        help="Optional path to save the sharpness plot as an image (e.g. scores.png). Shows interactively if not given.",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory.")
        sys.exit(1)

    extensions = {e.lstrip(".").lower() for e in args.ext}
    images = collect_images(folder, extensions)

    if not images:
        print(f"No images found in '{folder}' with extensions: {extensions}")
        sys.exit(0)

    print(f"Scoring {len(images)} images in '{folder}'...\n")

    results = []
    for path in images:
        score = laplacian_variance(path)
        results.append((path.name, score))
        status = f"{score:>12.2f}" if score >= 0 else "     UNREADABLE"
        print(f"  {path.name:<50s}  {status}")

    if args.sort == "score":
        results.sort(key=lambda x: x[1], reverse=True)
    # else already sorted by name

    scores = [s for _, s in results if s >= 0]
    if scores:
        print(f"\n{'─'*60}")
        print(f"  Images scored : {len(scores)}/{len(results)}")
        print(
            f"  Sharpest      : {max(scores):.2f}  ({results[0][0] if args.sort == 'score' else max(results, key=lambda x: x[1])[0]})"
        )
        print(
            f"  Blurriest     : {min(scores):.2f}  ({min(results, key=lambda x: x[1])[0]})"
        )
        print(f"  Mean          : {np.mean(scores):.2f}")
        print(f"  Std           : {np.std(scores):.2f}")

    if args.save:
        with open(args.save, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "sharpness_score"])
            writer.writerows(results)
        print(f"\nResults saved to '{args.save}'")

    # Plot sharpness scores in image order
    names = [r[0] for r in results]
    scores_plot = [r[1] for r in results]
    indices = list(range(len(names)))

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.3), 5))
    bars = ax.bar(indices, scores_plot, color="steelblue", edgecolor="none")

    # Highlight unreadable images
    for bar, s in zip(bars, scores_plot):
        if s < 0:
            bar.set_color("red")

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Sharpness (Laplacian variance)")
    ax.set_title(f"Sharpness scores — {folder.name}")
    ax.set_xticks(indices[:: max(1, len(indices) // 20)])
    ax.set_xticklabels(
        [str(i) for i in indices[:: max(1, len(indices) // 20)]],
        rotation=45,
        ha="right",
    )
    fig.tight_layout()

    if args.plot:
        fig.savefig(args.plot, dpi=150)
        print(f"Plot saved to '{args.plot}'")
    else:
        plt.show()


if __name__ == "__main__":
    main()
