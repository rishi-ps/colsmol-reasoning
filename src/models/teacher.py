"""
Teacher model wrapper for knowledge distillation (Idea 1).

Supports ColPali (3B) and ColQwen2 (2B) as teachers.
Runs in inference-only mode with optional CPU offloading for 8GB VRAM.
"""

import torch
from typing import Optional
from dataclasses import dataclass
from src.utils import ensure_hf_repo_cached, hf_offline_enabled


@dataclass
class TeacherConfig:
    """Configuration for teacher model loading."""
    model_name: str = "vidore/colpali-v1.3"  # or "vidore/colqwen2-v1.0"
    device: str = "cpu"  # default to CPU to save GPU memory for student
    dtype: torch.dtype = torch.float32


class TeacherModelWrapper:
    """Wrapper for teacher models used in distillation.

    Runs inference-only to generate:
    - Document/query embeddings (for L_contrastive)
    - MaxSim attention maps (for L_interaction)
    - Score orderings (for L_ranking)
    """

    def __init__(self, config: Optional[TeacherConfig] = None):
        self.config = config or TeacherConfig()
        self.model = None
        self.processor = None

    def load(self):
        """Load the teacher model in eval mode."""
        # Auto-detect model type from name
        model_name = self.config.model_name.lower()
        local_only = hf_offline_enabled()
        ensure_hf_repo_cached(
            self.config.model_name,
            required_files=["config.json", "tokenizer_config.json"],
        )

        if "colpali" in model_name:
            from colpali_engine.models import ColPali, ColPaliProcessor
            self.processor = ColPaliProcessor.from_pretrained(
                self.config.model_name,
                local_files_only=local_only,
            )
            self.model = ColPali.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.dtype,
                local_files_only=local_only,
            ).to(self.config.device)
        elif "colqwen" in model_name:
            from colpali_engine.models import ColQwen2, ColQwen2Processor
            self.processor = ColQwen2Processor.from_pretrained(
                self.config.model_name,
                local_files_only=local_only,
            )
            self.model = ColQwen2.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.dtype,
                local_files_only=local_only,
            ).to(self.config.device)
        else:
            raise ValueError(f"Unknown teacher model: {self.config.model_name}")

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"Teacher loaded: {self.config.model_name} on {self.config.device}")
        return self

    @torch.no_grad()
    def encode_queries(self, queries: list[str], batch_size: int = 4) -> torch.Tensor:
        """Encode queries using the teacher model."""
        all_embeddings = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            inputs = self.processor.process_queries(batch).to(self.config.device)
            embeddings = self.model(**inputs)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def encode_documents(self, images, batch_size: int = 2) -> torch.Tensor:
        """Encode document images using the teacher model."""
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor.process_images(batch).to(self.config.device)
            embeddings = self.model(**inputs)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def get_maxsim_attention(
        self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute MaxSim attention maps (which query token matched which doc token).

        Returns:
            attention_maps: (num_queries, num_query_tokens, num_doc_tokens)
                Softmax-normalized similarity showing the teacher's matching pattern.
        """
        # query_embeddings: (B, Tq, D)
        # doc_embeddings:   (B, Td, D)
        # For each sample in the batch, compute token-token similarities.
        sim = torch.einsum("bqd,bkd->bqk", query_embeddings, doc_embeddings)
        attention = torch.softmax(sim / (query_embeddings.shape[-1] ** 0.5), dim=-1)
        return attention
