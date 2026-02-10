"""
Train ColSmol with MaxSim Interaction Distillation (MID).

Usage:
    python scripts/train_distillation.py --config configs/distillation/base.yaml
"""

import argparse
import yaml
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.colsmol import ColSmolWrapper, ColSmolConfig
from src.models.teacher import TeacherModelWrapper, TeacherConfig
from src.distillation.trainer import DistillationTrainer, DistillationConfig


def main():
    parser = argparse.ArgumentParser(description="Train ColSmol with MID")
    parser.add_argument("--config", type=str, default="configs/distillation/base.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Initialize student
    student_cfg = ColSmolConfig(
        model_name=cfg["student"]["model_name"],
        use_lora=cfg["student"]["use_lora"],
        lora_rank=cfg["student"]["lora_rank"],
        lora_alpha=cfg["student"]["lora_alpha"],
    )
    student = ColSmolWrapper(student_cfg).load()
    print(f"Student loaded: {student_cfg.model_name}")

    # Initialize teacher
    teacher_cfg = TeacherConfig(
        model_name=cfg["teacher"]["model_name"],
        device=cfg["teacher"]["device"],
    )
    teacher = TeacherModelWrapper(teacher_cfg).load()
    print(f"Teacher loaded: {teacher_cfg.model_name}")

    # Initialize trainer
    train_cfg = DistillationConfig(
        alpha=cfg["training"]["alpha"],
        beta=cfg["training"]["beta"],
        gamma=cfg["training"]["gamma"],
        temperature=cfg["training"]["temperature"],
        learning_rate=cfg["training"]["learning_rate"],
        batch_size=cfg["training"]["batch_size"],
        num_epochs=cfg["training"]["num_epochs"],
        output_dir=cfg["output"]["checkpoint_dir"],
    )
    trainer = DistillationTrainer(student, teacher, train_cfg)

    # TODO: Create DataLoader from ViDoRe training data
    # train_dataloader = ...
    # trainer.train(train_dataloader)

    print("\n[INFO] Distillation training script ready.")
    print("[TODO] Implement data pipeline and call trainer.train()")


if __name__ == "__main__":
    main()
