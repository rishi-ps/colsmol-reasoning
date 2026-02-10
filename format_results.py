"""
Format ViDoRe benchmark results into a table like the ColPali paper.
Displays nDCG@5 metrics across all benchmark datasets.
"""
import csv
import json
import os
from pathlib import Path

# Mapping from result file names to paper column names
DATASET_MAPPING = {
    "VidoreArxivQARetrieval": "ArxivQ",
    "VidoreDocVQARetrieval": "DocQ",
    "VidoreInfoVQARetrieval": "InfoQ",
    "VidoreTabfquadRetrieval": "TabF",
    "VidoreTatdqaRetrieval": "TATQ",
    "VidoreShiftProjectRetrieval": "Shift",
    "VidoreSyntheticDocQAAIRetrieval": "AI",
    "VidoreSyntheticDocQAEnergyRetrieval": "Energy",
    "VidoreSyntheticDocQAGovernmentReportsRetrieval": "Gov.",
    "VidoreSyntheticDocQAHealthcareIndustryRetrieval": "Health.",
}

# Order of columns as in the paper
COLUMN_ORDER = ["ArxivQ", "DocQ", "InfoQ", "TabF", "TATQ", "Shift", "AI", "Energy", "Gov.", "Health."]


def load_results(results_dir: str) -> dict:
    """Load all result JSON files and extract nDCG@5 scores."""
    scores = {}
    results_path = Path(results_dir)
    
    # Find the actual results directory
    if (results_path / "results").exists():
        results_path = results_path / "results"
    
    # Search for JSON files recursively
    for json_file in results_path.rglob("*.json"):
        if json_file.name == "model_meta.json":
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            task_name = data.get("task_name", json_file.stem)
            
            # Get the short name
            if task_name in DATASET_MAPPING:
                short_name = DATASET_MAPPING[task_name]
            else:
                continue
            
            # Extract nDCG@5 from test split
            if "scores" in data and "test" in data["scores"]:
                test_scores = data["scores"]["test"]
                if isinstance(test_scores, list) and len(test_scores) > 0:
                    ndcg_5 = test_scores[0].get("ndcg_at_5")
                    if ndcg_5 is not None:
                        scores[short_name] = ndcg_5 * 100  # Convert to percentage
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {json_file}: {e}")
    
    return scores


def print_table(scores: dict, model_name: str = "ColSmol"):
    """Print results in a formatted table similar to the paper."""
    # Header
    header = ["Model"] + COLUMN_ORDER + ["Avg."]
    
    # Calculate average
    valid_scores = [scores.get(col) for col in COLUMN_ORDER if scores.get(col) is not None]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    
    # Column widths
    col_widths = [max(len(h), 8) for h in header]
    col_widths[0] = max(len(model_name), 12)
    
    # Print separator
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    
    # Print header
    print("\n" + "=" * 80)
    print("ViDoRe Benchmark Results (nDCG@5)")
    print("=" * 80)
    print()
    
    print(separator)
    header_row = "|"
    for h, w in zip(header, col_widths):
        header_row += f" {h:^{w}} |"
    print(header_row)
    print(separator)
    
    # Print scores row
    row = f"| {model_name:<{col_widths[0]}} |"
    for col, w in zip(COLUMN_ORDER, col_widths[1:-1]):
        score = scores.get(col)
        if score is not None:
            row += f" {score:>{w}.1f} |"
        else:
            row += f" {'-':^{w}} |"
    row += f" {avg:>{col_widths[-1]}.1f} |"
    print(row)
    print(separator)
    
    # Print as markdown table
    print("\n### Markdown Table:\n")
    md_header = "| " + " | ".join(header) + " |"
    md_sep = "| " + " | ".join(["---"] * len(header)) + " |"
    
    md_row = f"| {model_name} |"
    for col in COLUMN_ORDER:
        score = scores.get(col)
        if score is not None:
            md_row += f" {score:.1f} |"
        else:
            md_row += " - |"
    md_row += f" {avg:.1f} |"
    
    print(md_header)
    print(md_sep)
    print(md_row)
    
    # Print raw scores for reference
    print("\n### Raw Scores:")
    for col in COLUMN_ORDER:
        score = scores.get(col)
        if score is not None:
            print(f"  {col}: {score:.2f}")
        else:
            print(f"  {col}: N/A")
    print(f"  Average: {avg:.2f}")


def save_csv(scores: dict, model_name: str, output_path: str = "results.csv"):
    """Save results to a CSV file."""
    # Calculate average
    valid_scores = [scores.get(col) for col in COLUMN_ORDER if scores.get(col) is not None]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ["Model"] + COLUMN_ORDER + ["Avg."]
        writer.writerow(header)
        # Data row
        row = [model_name]
        for col in COLUMN_ORDER:
            score = scores.get(col)
            row.append(f"{score:.1f}" if score is not None else "-")
        row.append(f"{avg:.1f}")
        writer.writerow(row)
    
    print(f"\nResults saved to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Format ViDoRe results like the ColPali paper table")
    parser.add_argument("--results-dir", default="results", help="Directory containing result JSON files")
    parser.add_argument("--model-name", default="ColSmol-256M", help="Model name for the table")
    parser.add_argument("--output-csv", default="results.csv", help="Output CSV file path")
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_dir}")
    scores = load_results(args.results_dir)
    
    if not scores:
        print("No results found! Make sure the results directory contains valid JSON files.")
        return
    
    print(f"Found {len(scores)} dataset results")
    print_table(scores, args.model_name)
    save_csv(scores, args.model_name, args.output_csv)


if __name__ == "__main__":
    main()
