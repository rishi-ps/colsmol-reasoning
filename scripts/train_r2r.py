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
import torch
from torch.utils.data import DataLoader, IterableDataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.colsmol import ColSmolWrapper, ColSmolConfig
from src.reasoning.augmented_retriever import ReasoningAugmentedRetriever, R2RConfig
from src.reasoning.trace_generator import TraceGenerator
from src.reasoning.trainer import R2RTrainer, R2RTrainerConfig
from src.data.vidore import load_vidore_dataset


class R2RDataset(IterableDataset):
    """Iterable dataset for R2R training.

    Joins `colpali_train_set` (streaming) with pre-computed traces.
    Only yields examples where we have a trace for the query.
    """

    def __init__(self, traces_path: str, limit: int = None):
        self.traces_path = traces_path
        self.limit = limit
        
        # Load traces into lookup dict
        print(f"Loading traces from {traces_path}...")
        with open(traces_path) as f:
            trace_list = json.load(f)
        
        # Map query -> trace
        self.trace_map = {item['query']: item['trace'] for item in trace_list}
        print(f"Loaded {len(self.trace_map)} traces")

        # Load ColPali training set (streaming)
        print("Loading colpali_train_set (streaming)...")
        # load_vidore_dataset returns a dict if specific subset is requested, or the dataset directly if only one
        # In src/data/vidore.py, we return a dict keyied by 'colpali_train_set'
        # But wait, let's check src/data/vidore.py logic:
        # It returns `loaded[short_name] = ...` and then `return next(iter(loaded.values()))` if len(loaded) == 1
        # So it returns the dataset directly!
        self.dataset = load_vidore_dataset(version="train", streaming=True)
        # Shuffle with same seed as generation script
        self.dataset = self.dataset.shuffle(seed=42, buffer_size=1000)

    def __iter__(self):
        count = 0
        for i, example in enumerate(self.dataset):
            query = example['query']
            if query in self.trace_map:
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
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Initialize retriever
    retriever_cfg = ColSmolConfig(
        model_name=cfg["retriever"]["model_name"],
        use_lora=cfg["retriever"]["use_lora"],
        lora_rank=cfg["retriever"]["lora_rank"],
        lora_alpha=cfg["retriever"]["lora_alpha"],
    )
    # Load base model
    retriever = ColSmolWrapper(retriever_cfg).load()

    # Initialize R2R wrapper
    r2r_cfg = R2RConfig(separator=cfg["training"]["separator"])
    r2r = ReasoningAugmentedRetriever(retriever, config=r2r_cfg)

    # Initialize dataset
    train_dataset = R2RDataset(args.traces, limit=args.limit)
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
        output_dir=cfg["output"]["checkpoint_dir"],
    )
    trainer = R2RTrainer(r2r, trainer_cfg)

    # Train
    trainer.train(train_dataloader)

    print("\n[INFO] R2R training finished!")


if __name__ == "__main__":
    main()
