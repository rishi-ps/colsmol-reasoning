"""
Reasoning-to-Retrieve (R2R) Trainer.

Trains ColSmol with reasoning-augmented queries use in-batch negatives.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
import csv
import os

from src.reasoning.augmented_retriever import ReasoningAugmentedRetriever


@dataclass
class R2RTrainerConfig:
    """Configuration for R2R training."""
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Loss temperature
    temperature: float = 0.02

    # Checkpointing
    save_every_n_steps: int = 500
    output_dir: str = "checkpoints/r2r"

    # Logging
    log_every_n_steps: int = 10
    metrics_csv_path: Optional[str] = None
    use_wandb: bool = False
    wandb_project: str = "colsmol-r2r"


class R2RTrainer:
    """Trainer for Reason-to-Retrieve (R2R).

    Fine-tunes the retriever using InfoNCE loss with in-batch negatives.
    Queries are augmented with reasoning traces before encoding.
    """

    def __init__(
        self,
        retriever: ReasoningAugmentedRetriever,
        config: Optional[R2RTrainerConfig] = None,
    ):
        self.retriever = retriever
        self.config = config or R2RTrainerConfig()
        self.optimizer = None
        self.global_step = 0
        self._metrics_initialized = False

    def setup(self):
        """Initialize optimizer."""
        # Train only the LoRA parameters (or full model if not LoRA)
        model = self.retriever.retriever.model
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.optimizer.zero_grad(set_to_none=True)
        self._init_metrics_file()
        print(f"Optimizer initialized with {len(trainable_params)} parameter groups")
        return self

    def compute_loss(
        self, 
        query_embeddings: torch.Tensor, 
        doc_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute In-Batch Contrastive Loss (InfoNCE).
        
        Args:
            query_embeddings: (B, Tq, D)
            doc_embeddings: (B, Td, D)
            
        Returns:
            Scalar loss
        """
        # MaxSim score for all query-doc pairs in the batch
        # query_embeddings: (B, Tq, D)
        # doc_embeddings: (B, Td, D)
        
        # We need (B, B) scores matrix where score[i, j] is sim(q_i, d_j)
        B = query_embeddings.shape[0]
        if B < 2:
            raise ValueError(
                "In-batch InfoNCE requires batch_size >= 2. "
                "Increase training.batch_size or switch to cross-batch negatives."
            )
        
        # Unfortunately, standard batch matrix multiplication doesn't work directly 
        # for (B, Tq, D) x (B, Td, D) -> (B, B) MaxSim
        # We can expand:
        # Q_exp: (B, 1, Tq, D) -> (B, B, Tq, D)
        # D_exp: (1, B, Td, D) -> (B, B, Td, D)
        # But this is O(B^2) memory which is expensive.
        
        # Iterative approach to save memory (since B is small, e.g. 4-8):
        scores = torch.zeros(B, B, device=query_embeddings.device)
        
        for i in range(B):
            # Compute sim(q_i, all_docs)
            # q_i: (1, Tq, D)
            # docs: (B, Td, D)
            # sim: (B, Tq, Td)
            q_i = query_embeddings[i].unsqueeze(0)
            sim_i = torch.einsum("qtd,bpd->btp", q_i, doc_embeddings)
            max_sim_i = sim_i.max(dim=-1).values.sum(dim=-1) # (B,)
            scores[i] = max_sim_i
            
        labels = torch.arange(B, device=query_embeddings.device, dtype=torch.long)
        # Cast to float32 to prevent overflow (exp(50) > 65504 which is max fp16)
        scores = scores.to(torch.float32)
        if torch.isnan(scores).any():
            print(f"NaN in scores! Max score: {scores.max()}, Min score: {scores.min()}")
        return F.cross_entropy(scores / self.config.temperature, labels)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            batch: dict with:
                - 'query': list[str]
                - 'trace': list[str]
                - 'image': list[PIL.Image]

        Returns:
            dict with loss
        """
        queries = batch['query']
        traces = batch['trace']
        images = batch['image']
        
        # Update model to training mode
        self.retriever.retriever.model.train()
        
        # 1. Encode augmented queries
        # (Using internal ColSmol wrapper to handle tokenization + model forward)
        # Note: We need gradients, so we shouldn't use the wrapper's @torch.no_grad methods 
        # if they have them.
        # Let's check src/models/colsmol.py... `encode_queries` has `with torch.no_grad():`!
        # CRITICAL: We need to modify ColSmol wrapper or manually call model here.
        # The wrapper is designed for inference/eval.
        
        # We will manually access processor/model from wrapper to support training
        processor = self.retriever.retriever.processor
        model = self.retriever.retriever.model
        device = self.retriever.retriever.config.device
        
        # Augment queries
        augmented_queries = self.retriever.augment_queries_batch(queries, traces)
        # Process queries with gradients
        q_inputs = processor.process_queries(augmented_queries).to(device)
        q_emb = model(**q_inputs) # (B, Tq, D)
        
        # Process images with gradients
        d_inputs = processor.process_images(images).to(device)
        d_emb = model(**d_inputs) # (B, Td, D)
        
        # Normalize in float32 for numerical stability (fp16 eps underflow can create NaNs).
        q_emb = F.normalize(q_emb.to(torch.float32), p=2, dim=-1, eps=1e-6)
        d_emb = F.normalize(d_emb.to(torch.float32), p=2, dim=-1, eps=1e-6)
        
        if torch.isnan(q_emb).any() or torch.isnan(d_emb).any():
            print(f"NaN in embeddings! Q: {torch.isnan(q_emb).any()}, D: {torch.isnan(d_emb).any()}")

        # Compute loss
        loss = self.compute_loss(q_emb, d_emb)
        
        # Backprop
        (loss / self.config.gradient_accumulation_steps).backward()
        
        # Optimizer step (with simple gradient accumulation handling)
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        return {"loss": loss.item()}

    def train(self, train_dataloader: DataLoader):
        """Full training loop."""
        self.setup()
        self.retriever.retriever.model.train()
        
        print(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = 0.0
            num_steps = 0
            
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(pbar):
                loss_dict = self.train_step(batch)
                
                loss_val = loss_dict["loss"]
                epoch_loss += loss_val
                num_steps += 1
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})
                self._append_metric(epoch + 1, loss_val)

                if self.global_step % self.config.log_every_n_steps == 0:
                    # Log (could add wandb here)
                    pass

                if self.global_step % self.config.save_every_n_steps == 0:
                    self._save_checkpoint()

            if self.global_step % self.config.gradient_accumulation_steps != 0:
                model = self.retriever.retriever.model
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            avg_loss = epoch_loss / max(num_steps, 1)
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        self._save_checkpoint(final=True)

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        suffix = "final" if final else f"step_{self.global_step}"
        path = os.path.join(self.config.output_dir, suffix)
        self.retriever.retriever.model.save_pretrained(path)
        print(f"Checkpoint saved: {path}")

    def _init_metrics_file(self):
        if not self.config.metrics_csv_path or self._metrics_initialized:
            return
        parent = os.path.dirname(self.config.metrics_csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.config.metrics_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["global_step", "epoch", "loss"])
        self._metrics_initialized = True

    def _append_metric(self, epoch: int, loss: float):
        if not self.config.metrics_csv_path:
            return
        with open(self.config.metrics_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.global_step + 1, epoch, float(loss)])
