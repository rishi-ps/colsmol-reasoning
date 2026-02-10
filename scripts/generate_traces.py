"""
Generate reasoning traces for training queries (R2R Phase 1).

Usage:
    python scripts/generate_traces.py --config configs/r2r/base.yaml --output data/traces/v1_traces.json
"""

import argparse
import json
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reasoning.trace_generator import TraceGenerator, TraceConfig
from src.data.vidore import load_vidore_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning traces")
    parser.add_argument("--config", type=str, default="configs/r2r/base.yaml")
    parser.add_argument("--output", type=str, default="data/traces/v1_traces.json")
    parser.add_argument("--version", type=str, default="v1", help="ViDoRe version")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Initialize trace generator
    trace_cfg = TraceConfig(
        model_name=cfg["reasoning_llm"]["model_name"],
        device=cfg["reasoning_llm"]["device"],
        max_new_tokens=cfg["reasoning_llm"]["max_new_tokens"],
        temperature=cfg["reasoning_llm"]["temperature"],
    )
    generator = TraceGenerator(trace_cfg).load()

    # Load queries from ViDoRe
    print(f"Loading ViDoRe {args.version} queries...")
    split = "train" if args.version == "train" else "test"
    streaming = args.version == "train"  # Use streaming for large training sets
    datasets = load_vidore_dataset(version=args.version, split=split, streaming=streaming)
    
    if streaming:
        # Shuffle with fixed seed to ensure deterministic order across runs
        # Note: streaming datasets require buffer_size for shuffling
        if isinstance(datasets, dict):
            for k in datasets:
                datasets[k] = datasets[k].shuffle(seed=42, buffer_size=1000)
        else: # single dataset
            datasets = datasets.shuffle(seed=42, buffer_size=1000)

    # Extract queries from all datasets
    all_queries = []
    if isinstance(datasets, dict):
        for name, ds in datasets.items():
            if streaming:
                # For streaming, we iterate until limit
                print(f"  Streaming queries from {name}...")
                count = 0
                for example in ds:
                    if "query" in example:
                        all_queries.append(example["query"])
                        count += 1
                        if args.limit and len(all_queries) >= args.limit:
                            break
                if args.limit and len(all_queries) >= args.limit:
                    break
            else:
                if "query" in ds.column_names:
                    queries = ds["query"]
                    all_queries.extend(queries)
                    print(f"  {name}: {len(queries)} queries")
    else:
        # Single dataset case
        ds = datasets
        if streaming:
             print(f"  Streaming queries...")
             for example in ds:
                if "query" in example:
                    all_queries.append(example["query"])
                    if args.limit and len(all_queries) >= args.limit:
                        break
        elif "query" in ds.column_names:
            all_queries = datasets["query"]

    if args.limit:
        all_queries = all_queries[:args.limit]

    print(f"\nGenerating traces for {len(all_queries)} queries...")
    traces = generator.generate_traces_batch(all_queries)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generator.save_traces(all_queries, traces, args.output)
    print(f"Done! Saved to {args.output}")


if __name__ == "__main__":
    main()
