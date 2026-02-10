"""
Distillation training loop for MaxSim Interaction Distillation (MID).

Trains a ColSmol student using a frozen teacher model (ColPali/ColQwen).
"""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Optional
from dataclasses import dataclass, field

from src.models.colsmol import ColSmolWrapper
from src.models.teacher import TeacherModelWrapper
from src.distillation.losses import (
    contrastive_loss,
    interaction_loss,
    ranking_loss,
    mid_loss,
)


@dataclass
class DistillationConfig:
    """Configuration for distillation training."""
    # Loss weights
    alpha: float = 1.0       # L_contrastive weight
    beta: float = 1.0        # L_interaction weight
    gamma: float = 0.5       # L_ranking weight

    # Training
    learning_rate: float = 2e-5
    batch_size: int = 2
    num_epochs: int = 3
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # KD temperature
    temperature: float = 2.0

    # Checkpointing
    save_every_n_steps: int = 500
    output_dir: str = "checkpoints/distillation"

    # Logging
    log_every_n_steps: int = 10
    use_wandb: bool = False
    wandb_project: str = "colsmol-distillation"


class DistillationTrainer:
    """Trainer for MaxSim Interaction Distillation.

    Orchestrates:
    1. Teacher forward pass (frozen) → embeddings + attention maps
    2. Student forward pass (trainable) → embeddings
    3. MID loss computation
    4. Gradient update on student only
    """

    def __init__(
        self,
        student: ColSmolWrapper,
        teacher: TeacherModelWrapper,
        config: Optional[DistillationConfig] = None,
    ):
        self.student = student
        self.teacher = teacher
        self.config = config or DistillationConfig()
        self.optimizer = None
        self.global_step = 0

    def setup(self):
        """Initialize optimizer and scheduler."""
        trainable_params = [
            p for p in self.student.model.parameters() if p.requires_grad
        ]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        print(f"Optimizer initialized with {len(trainable_params)} parameter groups")
        return self

    def train_step(self, batch: dict) -> dict:
        """Execute a single training step.

        Args:
            batch: dict with 'queries', 'positive_images', 'negative_images'

        Returns:
            dict with loss values for logging
        """
        # TODO: Implement full training step
        # This is the skeleton — actual implementation depends on
        # data format from ViDoRe training splits
        raise NotImplementedError(
            "Training step needs implementation once data pipeline is finalized. "
            "See Idea 1 experiment plan in docs/publishable_research_directions.md"
        )

    def train(self, train_dataloader: DataLoader):
        """Full training loop.

        Args:
            train_dataloader: DataLoader yielding training batches
        """
        self.student.model.train()
        self.setup()

        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*50}")

            epoch_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                losses = self.train_step(batch)

                if (self.global_step + 1) % self.config.log_every_n_steps == 0:
                    self._log(losses)

                if (self.global_step + 1) % self.config.save_every_n_steps == 0:
                    self._save_checkpoint()

                self.global_step += 1
                epoch_loss += losses.get("total", 0.0)

            avg_loss = epoch_loss / max(step + 1, 1)
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        self._save_checkpoint(final=True)

    def _log(self, losses: dict):
        """Log training metrics."""
        msg = f"Step {self.global_step}: "
        msg += " | ".join(f"{k}: {v:.4f}" for k, v in losses.items())
        print(msg)

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
        suffix = "final" if final else f"step_{self.global_step}"
        path = os.path.join(self.config.output_dir, suffix)
        self.student.model.save_pretrained(path)
        print(f"Checkpoint saved: {path}")
