"""
Train ColSmol using MaxSim Interaction Distillation (Idea 1).

Usage:
    python scripts/train_distillation.py --config configs/distillation/base.yaml
"""

import argparse
import yaml
import sys
import os
import torch
from torch.utils.data import DataLoader, IterableDataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.colsmol import ColSmolWrapper, ColSmolConfig
from src.models.teacher import TeacherModelWrapper, TeacherConfig
from src.distillation.trainer import DistillationTrainer, DistillationConfig
from src.data.vidore import load_vidore_dataset


class DistillationDataset(IterableDataset):
    """Iterable dataset for Distillation training.
    
    Wraps ViDoRe training set.
    """
    def __init__(self, limit: int = None):
        self.limit = limit
        print("Loading colpali_train_set (streaming)...")
        self.dataset = load_vidore_dataset(version="train", streaming=True)
        self.dataset = self.dataset.shuffle(seed=42, buffer_size=1000)

    def __iter__(self):
        count = 0
        for example in self.dataset:
            if 'query' in example and 'image' in example:
                yield {
                    'query': example['query'],
                    'image': example['image'],  # PIL Image
                }
                count += 1
                if self.limit and count >= self.limit:
                    break


def collate_fn(batch):
    """Collate batch for Distillation training."""
    return {
        'query': [x['query'] for x in batch],
        'image': [x['image'] for x in batch],
    }


def main():
    parser = argparse.ArgumentParser(description="Train Distillation ColSmol")
    parser.add_argument("--config", type=str, default="configs/distillation/base.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training samples")
    args = parser.parse_args()

    # Load config and create logic if file doesn't exist
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    else:
        # Default config if file missing
        print(f"Config {args.config} not found, using defaults.")
        cfg = {
            "teacher": {"model_name": "vidore/colpali-v1.3", "device": "cpu"},
            "student": {
                "model_name": "vidore/colSmol-256M", 
                "use_lora": True, "lora_rank": 32, "lora_alpha": 32,
                "device": "cuda"
            },
            "training": {
                "alpha": 1.0, "beta": 1.0, "gamma": 0.5,
                "learning_rate": 2.0e-5, "batch_size": 2,
                "num_epochs": 1, "gradient_accumulation_steps": 8
            },
            "output": {"checkpoint_dir": "checkpoints/distillation"}
        }

    # Initialize Teacher (Frozen, on CPU/Offload)
    print(f"Initializing Teacher: {cfg['teacher']['model_name']}...")
    teacher_cfg = TeacherConfig(
        model_name=cfg['teacher']['model_name'],
        device=cfg['teacher']['device']
    )
    teacher = TeacherModelWrapper(teacher_cfg).load()

    # Initialize Student (Trainable, on GPU)
    print(f"Initializing Student: {cfg['student']['model_name']}...")
    student_cfg = ColSmolConfig(
        model_name=cfg['student']['model_name'],
        use_lora=cfg['student']['use_lora'],
        lora_rank=cfg['student']['lora_rank'],
        lora_alpha=cfg['student']['lora_alpha'],
        device=cfg['student']['device']
    )
    student = ColSmolWrapper(student_cfg).load()

    # Initialize Dataset
    train_dataset = DistillationDataset(limit=args.limit)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg['training']['batch_size'], 
        collate_fn=collate_fn
    )

    # Initialize Trainer
    trainer_cfg = DistillationConfig(
        alpha=cfg['training']['alpha'],
        beta=cfg['training']['beta'],
        gamma=cfg['training']['gamma'],
        learning_rate=float(cfg['training']['learning_rate']),
        batch_size=cfg['training']['batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        num_epochs=cfg['training']['num_epochs'],
        output_dir=cfg['output']['checkpoint_dir'],
    )
    trainer = DistillationTrainer(student, teacher, trainer_cfg)

    # Train
    trainer.train(train_dataloader)

    print("\n[INFO] Distillation training finished!")


if __name__ == "__main__":
    main()
