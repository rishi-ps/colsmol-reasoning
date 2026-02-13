"""
Compare R2R ablation runs from MTEB JSON outputs.

Expected usage:
  python evaluation/compare_ablation_results.py \
    --run use=results/r2r_use_seed42 \
    --run none=results/r2r_none_seed42 \
    --run shuffle=results/r2r_shuffle_seed42 \
    --output-csv results/r2r_ablation_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_scores(results_dir: str) -> dict[str, float]:
    """Load task_name -> ndcg_at_5 scores from MTEB result JSON files."""
    root = Path(results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    scores: dict[str, float] = {}
    for json_file in root.rglob("*.json"):
        if json_file.name == "model_meta.json":
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        task_name = data.get("task_name")
        test = data.get("scores", {}).get("test", [])
        if not task_name or not test or not isinstance(test, list):
            continue
        first = test[0] if isinstance(test[0], dict) else {}
        ndcg = first.get("ndcg_at_5")
        if ndcg is None:
            continue
        scores[task_name] = float(ndcg) * 100.0
    return scores


def parse_run_args(run_args: list[str]) -> dict[str, str]:
    runs: dict[str, str] = {}
    for item in run_args:
        if "=" not in item:
            raise ValueError(f"Invalid --run format `{item}`. Expected mode=path.")
        mode, path = item.split("=", 1)
        mode = mode.strip()
        path = path.strip()
        if not mode or not path:
            raise ValueError(f"Invalid --run format `{item}`.")
        runs[mode] = path
    return runs


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="Compare R2R ablation result folders")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec: mode=results_dir (repeat for use/none/shuffle)",
    )
    parser.add_argument(
        "--baseline-mode",
        default="none",
        help="Mode used as baseline for delta columns (default: none)",
    )
    parser.add_argument(
        "--output-csv",
        default="results/r2r_ablation_comparison.csv",
        help="Path to write comparison CSV",
    )
    args = parser.parse_args()

    runs = parse_run_args(args.run)
    if args.baseline_mode not in runs:
        raise ValueError(
            f"Baseline mode `{args.baseline_mode}` not provided. Modes: {sorted(runs.keys())}"
        )

    run_scores = {mode: load_scores(path) for mode, path in runs.items()}
    all_tasks = sorted(set().union(*[set(s.keys()) for s in run_scores.values()]))

    baseline_mode = args.baseline_mode
    mode_order = [m for m in ["none", "use", "shuffle"] if m in run_scores]
    mode_order += [m for m in sorted(run_scores.keys()) if m not in mode_order]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["task_name"]
    for mode in mode_order:
        header.append(f"{mode}_ndcg_at_5")
    for mode in mode_order:
        if mode != baseline_mode:
            header.append(f"delta_{mode}_vs_{baseline_mode}")

    rows: list[list[str]] = []
    for task in all_tasks:
        row: list[str] = [task]
        for mode in mode_order:
            value = run_scores[mode].get(task)
            row.append(f"{value:.4f}" if value is not None else "")
        base = run_scores[baseline_mode].get(task)
        for mode in mode_order:
            if mode == baseline_mode:
                continue
            cur = run_scores[mode].get(task)
            if base is None or cur is None:
                row.append("")
            else:
                row.append(f"{(cur - base):.4f}")
        rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Console summary
    print(f"Saved comparison CSV: {out_path}")
    print("\nAverage nDCG@5 by mode:")
    avg_by_mode: dict[str, float] = {}
    for mode in mode_order:
        vals = [run_scores[mode][t] for t in all_tasks if t in run_scores[mode]]
        avg_by_mode[mode] = mean(vals)
        print(f"  {mode}: {avg_by_mode[mode]:.4f}")

    print("\nAverage deltas vs baseline:")
    for mode in mode_order:
        if mode == baseline_mode:
            continue
        shared = [t for t in all_tasks if t in run_scores[mode] and t in run_scores[baseline_mode]]
        deltas = [run_scores[mode][t] - run_scores[baseline_mode][t] for t in shared]
        print(f"  {mode} - {baseline_mode}: {mean(deltas):.4f}")

    # Markdown table for quick paste into notes
    print("\nMarkdown Summary")
    md_header = ["Mode", "Avg nDCG@5", f"Delta vs {baseline_mode}"]
    print("| " + " | ".join(md_header) + " |")
    print("| --- | ---: | ---: |")
    for mode in mode_order:
        delta = 0.0 if mode == baseline_mode else avg_by_mode[mode] - avg_by_mode[baseline_mode]
        print(f"| {mode} | {avg_by_mode[mode]:.4f} | {delta:.4f} |")


if __name__ == "__main__":
    main()

