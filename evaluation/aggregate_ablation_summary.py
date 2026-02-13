"""
Aggregate per-benchmark R2R ablation comparison CSVs into one summary table.

Expected inputs are CSV files produced by `evaluation/compare_ablation_results.py`.

Example:
  python evaluation/aggregate_ablation_summary.py \
    --csv vidorev1=results/r2r_ablation_comparison_vidorev1.csv \
    --csv vidorev2=results/r2r_ablation_comparison_vidorev2.csv \
    --csv vidorev3=results/r2r_ablation_comparison_vidorev3.csv \
    --output-csv results/r2r_ablation_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_csv_specs(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --csv format `{item}`. Expected name=path.")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --csv format `{item}`.")
        out[name] = Path(path)
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def read_benchmark_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")

    none_vals: list[float] = []
    use_vals: list[float] = []
    shuffle_vals: list[float] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("none_ndcg_at_5"):
                none_vals.append(float(row["none_ndcg_at_5"]))
            if row.get("use_ndcg_at_5"):
                use_vals.append(float(row["use_ndcg_at_5"]))
            if row.get("shuffle_ndcg_at_5"):
                shuffle_vals.append(float(row["shuffle_ndcg_at_5"]))

    none_avg = mean(none_vals)
    use_avg = mean(use_vals)
    shuffle_avg = mean(shuffle_vals)
    return {
        "none_avg_ndcg_at_5": none_avg,
        "use_avg_ndcg_at_5": use_avg,
        "shuffle_avg_ndcg_at_5": shuffle_avg,
        "delta_use_vs_none": use_avg - none_avg,
        "delta_use_vs_shuffle": use_avg - shuffle_avg,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate R2R ablation CSV summaries")
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Benchmark CSV spec: name=path (repeat for each benchmark)",
    )
    parser.add_argument(
        "--output-csv",
        default="results/r2r_ablation_summary.csv",
        help="Path to write the aggregated summary CSV",
    )
    args = parser.parse_args()

    csv_specs = parse_csv_specs(args.csv)
    rows: list[dict[str, float | str]] = []

    for bench_name, csv_path in csv_specs.items():
        metrics = read_benchmark_csv(csv_path)
        rows.append({"benchmark": bench_name, **metrics})

    overall = {
        "benchmark": "overall",
        "none_avg_ndcg_at_5": mean([float(r["none_avg_ndcg_at_5"]) for r in rows]),
        "use_avg_ndcg_at_5": mean([float(r["use_avg_ndcg_at_5"]) for r in rows]),
        "shuffle_avg_ndcg_at_5": mean([float(r["shuffle_avg_ndcg_at_5"]) for r in rows]),
    }
    overall["delta_use_vs_none"] = (
        float(overall["use_avg_ndcg_at_5"]) - float(overall["none_avg_ndcg_at_5"])
    )
    overall["delta_use_vs_shuffle"] = (
        float(overall["use_avg_ndcg_at_5"]) - float(overall["shuffle_avg_ndcg_at_5"])
    )
    rows.append(overall)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "benchmark",
        "none_avg_ndcg_at_5",
        "use_avg_ndcg_at_5",
        "shuffle_avg_ndcg_at_5",
        "delta_use_vs_none",
        "delta_use_vs_shuffle",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved summary CSV: {out_path}")
    print("\nMarkdown Summary")
    print("| Benchmark | None Avg nDCG@5 | Use Avg nDCG@5 | Shuffle Avg nDCG@5 | Use-None | Use-Shuffle |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            "| {benchmark} | {none:.4f} | {use:.4f} | {shuffle:.4f} | {d1:.4f} | {d2:.4f} |".format(
                benchmark=row["benchmark"],
                none=float(row["none_avg_ndcg_at_5"]),
                use=float(row["use_avg_ndcg_at_5"]),
                shuffle=float(row["shuffle_avg_ndcg_at_5"]),
                d1=float(row["delta_use_vs_none"]),
                d2=float(row["delta_use_vs_shuffle"]),
            )
        )


if __name__ == "__main__":
    main()
