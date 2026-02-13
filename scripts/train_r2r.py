"""
Train ColSmol with reasoning-augmented queries (R2R Phase 2).

Usage:
    python scripts/train_r2r.py --config configs/r2r/base.yaml --traces data/traces/train_traces.json
"""

import argparse
import yaml
import sys
import os
import json
import random
import torch
from datetime import datetime
from torch.utils.data import DataLoader, IterableDataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.colsmol import ColSmolWrapper, ColSmolConfig
from src.reasoning.augmented_retriever import ReasoningAugmentedRetriever, R2RConfig
from src.reasoning.trainer import R2RTrainer, R2RTrainerConfig
from src.data.vidore import load_vidore_dataset
from src.utils import create_run_name, set_global_seed, write_json


class R2RDataset(IterableDataset):
    """Iterable dataset for R2R training.

    Joins `colpali_train_set` (streaming) with pre-computed traces.
    Only yields examples where we have a trace for the query.
    """

    def __init__(
        self,
        traces_path: str,
        limit: int = None,
        seed: int = 42,
        trace_mode: str = "use",
    ):
        self.traces_path = traces_path
        self.limit = limit
        self.seed = seed
        self.trace_mode = trace_mode
        
        self.trace_map = {}
        if self.trace_mode in {"use", "shuffle"}:
            print(f"Loading traces from {traces_path}...")
            with open(traces_path) as f:
                trace_list = json.load(f)
            self.trace_map = {item['query']: item['trace'] for item in trace_list}
            print(f"Loaded {len(self.trace_map)} traces")
            if self.trace_mode == "shuffle":
                keys = list(self.trace_map.keys())
                vals = list(self.trace_map.values())
                rnd = random.Random(self.seed)
                rnd.shuffle(vals)
                self.trace_map = dict(zip(keys, vals))
                print("Applied shuffled trace assignment (deterministic)")

        # Load ColPali training set (streaming)
        print("Loading colpali_train_set (streaming)...")
        self.dataset = load_vidore_dataset(version="train", streaming=True)
        if isinstance(self.dataset, dict):
            if not self.dataset:
                raise RuntimeError(
                    "Failed to load training dataset `vidore/colpali_train_set`. "
                    "If running without network, ensure dataset is cached locally."
                )
            if len(self.dataset) == 1:
                self.dataset = next(iter(self.dataset.values()))
            else:
                raise RuntimeError(
                    "Unexpected multiple datasets returned for version=train."
                )
        if not hasattr(self.dataset, "shuffle"):
            raise RuntimeError(
                f"Loaded training dataset does not support shuffle: {type(self.dataset)}"
            )
        self.dataset = self.dataset.shuffle(seed=self.seed, buffer_size=1000)

    def __iter__(self):
        count = 0
        for i, example in enumerate(self.dataset):
            query = example['query']
            if self.trace_mode == "none":
                yield {
                    'query': query,
                    'trace': "",
                    'image': example['image'],  # PIL Image
                }
                count += 1
                if self.limit and count >= self.limit:
                    break
            elif query in self.trace_map:
                yield {
                    'query': query,
                    'trace': self.trace_map[query],
                    'image': example['image'],  # PIL Image
                }
                count += 1
                if self.limit and count >= self.limit:
                    break
            # else:
            #     print(f"Skipping query (no trace): {query[:30]}...")
            
            # Safety break to avoid infinite loop if no matches found quickly
            if i > 5000 and count == 0:
                print("Warning: Scanned 5000 examples but found no matches in trace_map!")
                break


def collate_fn(batch):
    """Collate batch for R2R training.
    
    Returns:
        dict with lists of queries, traces, and images.
    """
    return {
        'query': [x['query'] for x in batch],
        'trace': [x['trace'] for x in batch],
        'image': [x['image'] for x in batch],
    }


def main():
    parser = argparse.ArgumentParser(description="Train R2R ColSmol")
    parser.add_argument("--config", type=str, default="configs/r2r/base.yaml")
    parser.add_argument("--traces", type=str, default="data/traces/train_traces.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Model dtype for ColSmol loading.",
    )
    parser.add_argument(
        "--trace-mode",
        type=str,
        choices=["use", "none", "shuffle"],
        default="use",
        help="Ablation mode: use traces, no traces (query-only), or shuffled traces.",
    )
    parser.add_argument("--hf-offline", action="store_true", help="Enable offline mode for Hugging Face model/data loading")
    args = parser.parse_args()
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    set_global_seed(args.seed)

    def resolve_dtype(name: str) -> torch.dtype:
        if name == "fp16":
            return torch.float16
        if name == "bf16":
            return torch.bfloat16
        if name == "fp32":
            return torch.float32
        # auto
        if torch.cuda.is_available():
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    model_dtype = resolve_dtype(args.dtype)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_name = args.run_name or create_run_name("r2r")
    run_root = os.path.join(cfg["output"]["checkpoint_dir"], "runs", run_name)
    checkpoints_dir = os.path.join(run_root, "checkpoints")
    os.makedirs(run_root, exist_ok=True)

    run_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "seed": args.seed,
        "config_path": args.config,
        "traces_path": args.traces,
        "limit": args.limit,
        "trace_mode": args.trace_mode,
        "dtype": args.dtype,
        "training_config": cfg["training"],
        "retriever_config": cfg["retriever"],
    }
    write_json(os.path.join(run_root, "run_metadata.json"), run_metadata)
    print(f"Run directory: {run_root}")

    # Initialize retriever
    retriever_cfg = ColSmolConfig(
        model_name=cfg["retriever"]["model_name"],
        use_lora=cfg["retriever"]["use_lora"],
        lora_rank=cfg["retriever"]["lora_rank"],
        lora_alpha=cfg["retriever"]["lora_alpha"],
        dtype=model_dtype,
        enable_gradient_checkpointing=cfg["retriever"].get("enable_gradient_checkpointing", False),
    )
    # Load base model
    retriever = ColSmolWrapper(retriever_cfg).load()

    # Initialize R2R wrapper
    r2r_cfg = R2RConfig(separator=cfg["training"]["separator"])
    r2r = ReasoningAugmentedRetriever(retriever, config=r2r_cfg)

    # Initialize dataset
    train_dataset = R2RDataset(
        args.traces,
        limit=args.limit,
        seed=args.seed,
        trace_mode=args.trace_mode,
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg["training"]["batch_size"], 
        collate_fn=collate_fn
    )

    # Initialize trainer
    trainer_cfg = R2RTrainerConfig(
        learning_rate=float(cfg["training"]["learning_rate"]),
        batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_epochs=cfg["training"]["num_epochs"],
        warmup_steps=cfg["training"]["warmup_steps"],
        temperature=float(cfg["training"].get("temperature", 0.02)),
        output_dir=checkpoints_dir,
        metrics_csv_path=os.path.join(run_root, "train_metrics.csv"),
    )
    trainer = R2RTrainer(r2r, trainer_cfg)

    # Train
    trainer.train(train_dataloader)

    print("\n[INFO] R2R training finished!")
    print(f"[INFO] Run artifacts written to: {run_root}")


if __name__ == "__main__":
    main()
