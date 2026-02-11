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
            batch: dict with:
                - 'query': list[str]
                - 'image': list[PIL.Image] (Positive documents)
                # TODO: Add negative mining support later
        
        Returns:
            dict with loss values for logging
        """
        queries = batch['query']
        images = batch['image']
        
        device = self.student.config.device
        
        # --- 1. Teacher Forward (Frozen, No Grad) ---
        with torch.no_grad():
            # We use the wrapper methods because they handle processing
            # AND they return CPU tensors which saves GPU memory.
            # We move to GPU only when needed for loss calculation.
            
            # Scores & Attention require (B, T, D) tensors
            t_q_emb = self.teacher.encode_queries(queries).to(device)
            t_d_emb = self.teacher.encode_documents(images).to(device)
            
            # Compute targets
            # Attention: (B, Tq, Td)
            t_attn = self.teacher.get_maxsim_attention(t_q_emb, t_d_emb)
            
            # Scores: (B,)
            # We use the helper from losses.py
            from src.distillation.losses import _maxsim_score
            t_scores = _maxsim_score(t_q_emb, t_d_emb)

        # --- 2. Student Forward (Trainable) ---
        # We must manually process to keep gradients
        s_processor = self.student.processor
        s_model = self.student.model
        
        # Process queries
        s_q_inputs = s_processor.process_queries(queries).to(device)
        s_q_emb = s_model(**s_q_inputs) # (B, Tq, D)
        
        # Process images
        s_d_inputs = s_processor.process_images(images).to(device)
        s_d_emb = s_model(**s_d_inputs) # (B, Td, D)
        
        # --- 3. Compute Losses ---
        # L_contrastive (InfoNCE)
        # We need negatives. For now, we use in-batch negatives.
        # neg_doc_embeddings for contrastive_loss expects (B, N, Td, D).
        # We can reshape s_d_emb to treat other batch items as negatives?
        # Standard contrastive_loss in losses.py takes specific negatives.
        # Let's use a simpler in-batch contrastive loss here directly or adapt.
        
        # Let's use the simple logic from R2R trainer for contrastive part
        # Or adaptation of losses.contrastive_loss
        
        # Calculate student scores for ranking loss
        s_scores = _maxsim_score(s_q_emb, s_d_emb)
        
        # 3a. Contrastive Loss (In-Batch)
        # Using the logic from src/distillation/losses.py would require explicit negatives
        # wrapper. Let's compute it manually here for in-batch SimCLR style.
        # (B, B) similarity matrix
        sim_matrix = torch.einsum("btd,ctd->btc", s_q_emb, s_d_emb) # (B, Tq, B)?? No.
        # We need MaxSim(q_i, d_j) for all i, j.
        # scores[i, j] = sum_t max_p (q_i_t * d_j_p)
        
        # Memory-efficient in-batch calculation
        B = len(queries)
        scores_matrix = torch.zeros(B, B, device=device)
        for i in range(B):
            # q_i: (1, Tq, D)
            q_i = s_q_emb[i].unsqueeze(0)
            # Match against ALL docs: (B, Td, D)
            # e.g. term-wise sim: (B, Tq, Td)
            sim_i = torch.einsum("qtd,bpd->btp", q_i, s_d_emb)
            max_sim_i = sim_i.max(dim=-1).values.sum(dim=-1) # (B,)
            scores_matrix[i] = max_sim_i
            
        labels = torch.arange(B, device=device, dtype=torch.long)
        l_contrastive = torch.nn.functional.cross_entropy(
            scores_matrix / self.config.temperature, labels
        )

        # 3b. Interaction Loss (KL Div)
        l_interaction = interaction_loss(
            s_q_emb, s_d_emb, t_attn, temperature=self.config.temperature
        )
        
        # 3c. Ranking Loss (Margin)
        # We compute pair-wise margins. 
        # We need (B, B) scores for teacher too for full ranking loss?
        # The loss.ranking_loss expects (B, N).
        # Let's iterate over batch to get (B, B) teacher scores too
        with torch.no_grad():
             t_scores_matrix = torch.zeros(B, B, device=device)
             for i in range(B):
                t_q_i = t_q_emb[i].unsqueeze(0)
                t_sim_i = torch.einsum("qtd,bpd->btp", t_q_i, t_d_emb)
                t_max_sim_i = t_sim_i.max(dim=-1).values.sum(dim=-1)
                t_scores_matrix[i] = t_max_sim_i
        
        l_ranking = ranking_loss(scores_matrix, t_scores_matrix)

        # Total Loss
        loss = mid_loss(
            l_contrastive, 
            l_interaction, 
            l_ranking,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma
        )

        # --- 4. Optimization ---
        loss.backward()
        
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(s_model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            "total": loss.item(),
            "cont": l_contrastive.item(),
            "inter": l_interaction.item(),
            "rank": l_ranking.item()
        }

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
