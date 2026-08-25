#!/usr/bin/env python3
"""Extract metrics across multiple seeds for a given experiment.

Extracts PSNR, LPIPS, SSIM, num_gaussians, reconstruction time, and reprojection error
across all seeds (seed_0 to seed_4), calculates mean and std dev per dataset.
"""

import json
import csv
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

OUTPUTS_DIR = Path(__file__).parent / "outputs"
SEEDS = ["seed_0", "seed_1", "seed_2", "seed_3", "seed_4"]

METRIC_KEYS = [
    "psnr",
    "ssim",
    "lpips",
    "num_gaussians",
    "reconstruction_time",
    "reprojection_error",
]


def extract_val_metrics(metrics_path: Path) -> dict:
    """Extract metrics from val_step29999.json file."""
    if not metrics_path.exists():
        return {}
    
    try:
        with open(metrics_path) as f:
            data = json.load(f)
        return {
            "psnr": data.get("psnr"),
            "ssim": data.get("ssim"),
            "lpips": data.get("lpips"),
            "num_gaussians": data.get("num_GS"),
        }
    except Exception as e:
        print(f"Error reading {metrics_path}: {e}", file=sys.stderr)
        return {}


def extract_reconstruction_time(timings_path: Path) -> float | None:
    """Extract total reconstruction time from pipeline_timings.log."""
    if not timings_path.exists():
        return None
    
    try:
        with open(timings_path) as f:
            for line in f:
                if "pipeline_total" in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        return float(parts[2])
    except Exception as e:
        print(f"Error reading {timings_path}: {e}", file=sys.stderr)
    
    return None


def extract_reprojection_error(summary_path: Path) -> float | None:
    """Extract mean reprojection error from colmap reconstruction_summary.txt."""
    if not summary_path.exists():
        return None
    
    try:
        with open(summary_path) as f:
            content = f.read()
            match = re.search(r"mean_reprojection_error\s*=\s*([\d.]+)", content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Error reading {summary_path}: {e}", file=sys.stderr)
    
    return None


def collect_metrics_for_seed(
    experiment_name: str, seed: str
) -> Dict[str, Dict[str, float]]:
    """Collect metrics for a given seed and experiment.
    
    Returns: {dataset_name: {metric_name: value}}
    """
    seed_exp_dir = OUTPUTS_DIR / seed / experiment_name
    if not seed_exp_dir.exists():
        print(f"Warning: {seed_exp_dir} does not exist", file=sys.stderr)
        return {}
    
    seed_data = {}
    for dataset_dir in sorted(seed_exp_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        metrics = {}
        
        # Extract val metrics
        val_metrics = extract_val_metrics(
            dataset_dir / "gs_3d" / "stats" / "val_step29999.json"
        )
        metrics.update(val_metrics)
        
        # Extract reconstruction time
        recon_time = extract_reconstruction_time(dataset_dir / "pipeline_timings.log")
        if recon_time is not None:
            metrics["reconstruction_time"] = recon_time
        
        # Extract reprojection error
        reproj_error = extract_reprojection_error(
            dataset_dir / "colmap" / "reconstruction_summary.txt"
        )
        if reproj_error is not None:
            metrics["reprojection_error"] = reproj_error
        
        seed_data[dataset_name] = metrics
    
    return seed_data


def compute_statistics(
    seed_metrics_list: List[Dict[str, float]]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute mean and std dev across seeds.
    
    Returns: (mean_dict, std_dict)
    """
    if not seed_metrics_list:
        return {}, {}
    
    # Filter out None values and aggregate
    aggregated = {}
    for seed_data in seed_metrics_list:
        for key, value in seed_data.items():
            if value is not None:
                if key not in aggregated:
                    aggregated[key] = []
                aggregated[key].append(value)
    
    mean_dict = {}
    std_dict = {}
    for key, values in aggregated.items():
        if values:
            mean_dict[key] = float(np.mean(values))
            std_dict[key] = float(np.std(values))
    
    return mean_dict, std_dict


def validate_dataset_counts(experiment_name: str) -> None:
    """Check that each seed has the same number of datasets and print the count."""
    print("\n" + "-" * 60)
    print("Validating dataset counts per seed:")

    counts = {}
    for seed in SEEDS:
        seed_exp_dir = OUTPUTS_DIR / seed / experiment_name
        if not seed_exp_dir.exists():
            print(f"  {seed}: MISSING ({seed_exp_dir})")
            counts[seed] = None
            continue
        datasets = sorted(d.name for d in seed_exp_dir.iterdir() if d.is_dir())
        counts[seed] = datasets
        print(f"  {seed}: {len(datasets)} datasets")

    present = {seed: ds for seed, ds in counts.items() if ds is not None}
    unique_counts = {len(ds) for ds in present.values()}

    if not present:
        print("WARNING: no seeds found for this experiment.")
    elif len(unique_counts) == 1:
        print(f"OK: all {len(present)} seeds have {unique_counts.pop()} datasets.")
    else:
        print("WARNING: seeds have differing dataset counts!")
        all_datasets = set().union(*present.values())
        for seed, ds in present.items():
            missing = sorted(all_datasets - set(ds))
            if missing:
                print(f"  {seed} missing: {', '.join(missing)}")
    print("-" * 60)


def collect_all_metrics(experiment_name: str) -> Dict[str, Dict[str, any]]:
    """Collect metrics across all seeds.
    
    Returns: {dataset_name: {"seeds": {...}, "mean": {...}, "std": {...}}}
    """
    all_data = {}
    
    for seed in SEEDS:
        seed_data = collect_metrics_for_seed(experiment_name, seed)
        for dataset_name, metrics in seed_data.items():
            if dataset_name not in all_data:
                all_data[dataset_name] = {"seeds": {}}
            all_data[dataset_name]["seeds"][seed] = metrics
    
    # Compute statistics
    for dataset_name in all_data:
        seed_metrics_list = [
            all_data[dataset_name]["seeds"].get(seed, {}) for seed in SEEDS
        ]
        mean_dict, std_dict = compute_statistics(seed_metrics_list)
        all_data[dataset_name]["mean"] = mean_dict
        all_data[dataset_name]["std"] = std_dict
    
    return all_data


def write_csv(all_data: Dict, experiment_name: str):
    """Write results to CSV."""
    safe_name = experiment_name.replace("/", "_")
    output_csv = OUTPUTS_DIR / f"metrics_{safe_name}.csv"
    
    # Build fieldnames
    fieldnames = ["dataset"]
    for metric in METRIC_KEYS:
        fieldnames.append(f"{metric}_mean")
        fieldnames.append(f"{metric}_std")
    
    rows = []
    for dataset in sorted(all_data.keys()):
        row = {"dataset": dataset}
        data_entry = all_data[dataset]
        mean_dict = data_entry.get("mean", {})
        std_dict = data_entry.get("std", {})
        
        for metric in METRIC_KEYS:
            row[f"{metric}_mean"] = mean_dict.get(metric, "")
            row[f"{metric}_std"] = std_dict.get(metric, "")
        
        rows.append(row)
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nWritten {len(rows)} rows to {output_csv}")


def print_table(all_data: Dict):
    """Print results as a readable table."""
    print("\n" + "=" * 180)
    print(f"{'Dataset':<40} | {'Metric':<20} | {'Mean':<15} | {'Std Dev':<15}")
    print("=" * 180)
    
    for dataset in sorted(all_data.keys()):
        data_entry = all_data[dataset]
        mean_dict = data_entry.get("mean", {})
        std_dict = data_entry.get("std", {})
        
        first = True
        for metric in METRIC_KEYS:
            mean_val = mean_dict.get(metric)
            std_val = std_dict.get(metric)
            
            dataset_col = dataset if first else ""
            if mean_val is not None:
                mean_str = f"{mean_val:.6f}"
            else:
                mean_str = "N/A"
            
            if std_val is not None:
                std_str = f"{std_val:.6f}"
            else:
                std_str = "N/A"
            
            print(
                f"{dataset_col:<40} | {metric:<20} | {mean_str:>15} | {std_str:>15}"
            )
            first = False
    
    print("=" * 180)


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_metrics.py <experiment_name>")
        print(f"Example: python extract_metrics.py main_paper")
        sys.exit(1)
    
    experiment_name = sys.argv[1]
    
    print(f"Collecting metrics for experiment: {experiment_name}")
    print(f"Seeds: {', '.join(SEEDS)}")
    
    validate_dataset_counts(experiment_name)
    
    all_data = collect_all_metrics(experiment_name)
    
    if not all_data:
        print("No metrics found.", file=sys.stderr)
        sys.exit(1)
    
    print_table(all_data)
    write_csv(all_data, experiment_name)


if __name__ == "__main__":
    main()
