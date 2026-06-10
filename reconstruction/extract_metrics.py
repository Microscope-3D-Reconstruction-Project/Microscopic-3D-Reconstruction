#!/usr/bin/env python3
"""Extract val_step29999.json metrics from test_baseline and test_fs_roman folders and output a comparison CSV."""

import json
import csv
import sys
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent / "outputs"
EXPERIMENT_FOLDERS = ["test_baseline", "test_fs_roman"]
METRICS_FILE = "gs_3d/stats/val_step29999.json"
OUTPUT_CSV = Path(__file__).parent / "metrics_comparison.csv"

METRIC_KEYS = ["psnr", "ssim", "lpips", "gss", "lss", "num_GS"]


def collect_metrics():
    # dataset -> experiment -> metrics
    data: dict[str, dict[str, dict]] = {}

    for exp in EXPERIMENT_FOLDERS:
        exp_dir = OUTPUTS_DIR / exp
        if not exp_dir.exists():
            print(f"Warning: {exp_dir} does not exist, skipping.", file=sys.stderr)
            continue
        for dataset_dir in sorted(exp_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            metrics_path = dataset_dir / METRICS_FILE
            if not metrics_path.exists():
                continue
            dataset = dataset_dir.name
            with open(metrics_path) as f:
                metrics = json.load(f)
            data.setdefault(dataset, {})[exp] = metrics

    return data


def write_csv(data: dict):
    # Build column headers: for each experiment, one column per metric
    experiments = EXPERIMENT_FOLDERS
    fieldnames = ["dataset"]
    for exp in experiments:
        for key in METRIC_KEYS:
            fieldnames.append(f"{exp}/{key}")

    rows = []
    for dataset in sorted(data.keys()):
        row = {"dataset": dataset}
        for exp in experiments:
            metrics = data[dataset].get(exp, {})
            for key in METRIC_KEYS:
                col = f"{exp}/{key}"
                row[col] = metrics.get(key, "")
        rows.append(row)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {OUTPUT_CSV}")


def print_table(data: dict):
    experiments = EXPERIMENT_FOLDERS
    # Print a readable table to stdout
    header_parts = [f"{'dataset':<40}"]
    for exp in experiments:
        for key in METRIC_KEYS:
            header_parts.append(f"{exp[:8]}/{key:<8}")
    print("  ".join(header_parts))
    print("-" * 160)

    for dataset in sorted(data.keys()):
        row_parts = [f"{dataset:<40}"]
        for exp in experiments:
            metrics = data[dataset].get(exp, {})
            for key in METRIC_KEYS:
                val = metrics.get(key, "N/A")
                if isinstance(val, float):
                    row_parts.append(f"{val:<16.4f}")
                else:
                    row_parts.append(f"{str(val):<16}")
        print("  ".join(row_parts))


if __name__ == "__main__":
    data = collect_metrics()
    if not data:
        print("No metrics found.", file=sys.stderr)
        sys.exit(1)
    print_table(data)
    write_csv(data)
