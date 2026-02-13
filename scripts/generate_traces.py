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
import time
import gc
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reasoning.trace_generator import TraceGenerator, TraceConfig
from src.data.vidore import load_vidore_dataset
from src.utils import create_run_name, set_global_seed, write_json


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning traces")
    parser.add_argument("--config", type=str, default="configs/r2r/base.yaml")
    parser.add_argument("--output", type=str, default="data/traces/v1_traces.json")
    parser.add_argument("--version", type=str, default="v1", help="ViDoRe version")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--save-every", type=int, default=100, help="Save every N queries")
    parser.add_argument("--sleep-interval", type=float, default=0.0, help="Sleep interval (seconds) between batches to cool down")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--hf-offline", action="store_true", help="Enable offline mode for Hugging Face model loading")
    args = parser.parse_args()
    set_global_seed(args.seed)
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Initialize trace generator
    trace_cfg = TraceConfig(
        model_name=cfg["reasoning_llm"]["model_name"],
        device=cfg["reasoning_llm"]["device"],
        max_new_tokens=cfg["reasoning_llm"]["max_new_tokens"],
        temperature=cfg["reasoning_llm"]["temperature"],
        batch_size=args.batch_size # Use CLI batch size or config
    )
    generator = TraceGenerator(trace_cfg).load()

    # Load queries from ViDoRe
    print(f"Loading ViDoRe {args.version} queries...")
    split = "train" if args.version == "train" else "test"
    streaming = args.version == "train"  # Use streaming for large training sets
    datasets = load_vidore_dataset(version=args.version, split=split, streaming=streaming)
    
    if streaming:
        # Shuffle with fixed seed to ensure deterministic order across runs
        if isinstance(datasets, dict):
            for k in datasets:
                datasets[k] = datasets[k].shuffle(seed=args.seed, buffer_size=1000)
        else: # single dataset
            datasets = datasets.shuffle(seed=args.seed, buffer_size=1000)

    # Prepare for incremental processing
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_name = args.run_name or create_run_name("traces")
    
    # Load existing traces if file exists (to resume or append)
    existing_data = []
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r') as f:
                existing_data = json.load(f)
            print(f"Loaded {len(existing_data)} existing traces from {args.output}")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {args.output}, starting fresh.")
    
    existing_queries = set(item["query"] for item in existing_data)
    
    # Generator for queries
    def query_generator():
        count = 0
        if isinstance(datasets, dict):
            for name, ds in datasets.items():
                print(f"  Streaming queries from {name}...")
                for example in ds:
                    if "query" in example:
                        q = example["query"]
                        if q not in existing_queries:
                            yield q
                            count += 1
                        if args.limit and (len(existing_data) + count) >= args.limit:
                            return
        else:
             print(f"  Streaming queries...")
             ds = datasets
             for example in ds:
                if "query" in example:
                    q = example["query"]
                    if q not in existing_queries:
                        yield q
                        count += 1
                    if args.limit and (len(existing_data) + count) >= args.limit:
                        return

    # Process in batches
    current_batch = []
    new_data = []
    total_processed = 0
    
    print(f"Starting generation...")
    
    for query in query_generator():
        current_batch.append(query)
        
        if len(current_batch) >= args.batch_size:
            # Generate traces for batch
            traces = generator.generate_traces_batch(current_batch)
            new_items = [{"query": q, "trace": t} for q, t in zip(current_batch, traces)]
            new_data.extend(new_items)
            existing_data.extend(new_items)
            
            total_processed += len(current_batch)
            current_batch = []
            
            # Save incrementally
            if len(new_data) >= args.save_every:
                print(f"Saving {len(existing_data)} traces to {args.output}...")
                with open(args.output, "w") as f:
                    json.dump(existing_data, f, indent=2)
                new_data = [] # Reset new data buffer (but we keep adding to existing_data for full dump)

            # Thermal management
            if args.sleep_interval > 0:
                time.sleep(args.sleep_interval)
                
            # Memory management to reduce heat/load
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Process remaining
    if current_batch:
        traces = generator.generate_traces_batch(current_batch)
        new_items = [{"query": q, "trace": t} for q, t in zip(current_batch, traces)]
        existing_data.extend(new_items)
        total_processed += len(current_batch)

    # Final save
    print(f"Final save: {len(existing_data)} traces to {args.output}")
    with open(args.output, "w") as f:
        json.dump(existing_data, f, indent=2)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "seed": args.seed,
        "config_path": args.config,
        "output_path": args.output,
        "version": args.version,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "save_every": args.save_every,
        "sleep_interval": args.sleep_interval,
        "total_traces_in_output": len(existing_data),
        "reasoning_model": cfg["reasoning_llm"]["model_name"],
    }
    metadata_path = f"{args.output}.meta.json"
    write_json(metadata_path, metadata)
    print(f"Run metadata saved: {metadata_path}")
    print("Done!")

if __name__ == "__main__":
    main()
