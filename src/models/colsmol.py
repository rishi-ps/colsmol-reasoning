"""
ColSmol model wrapper for retrieval experiments.

Provides a consistent interface for loading ColSmol (256M) with
optional LoRA adapters for fine-tuning on 8GB VRAM.
"""

import torch
from typing import Optional
from dataclasses import dataclass, field
from src.utils import ensure_hf_repo_cached, hf_offline_enabled


@dataclass
class ColSmolConfig:
    """Configuration for ColSmol model loading."""
    model_name: str = "vidore/colSmol-256M"
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = field(
        default_factory=lambda: torch.float16 if torch.cuda.is_available() else torch.float32
    )
    enable_gradient_checkpointing: bool = False


class ColSmolWrapper:
    """Wrapper around ColSmol for training and inference.

    Handles model loading, LoRA injection, and provides a unified
    interface for encoding queries and documents.
    """

    def __init__(self, config: Optional[ColSmolConfig] = None):
        self.config = config or ColSmolConfig()
        self.model = None
        self.processor = None

    def load(self):
        """Load the ColSmol model and processor."""
        # ColSmol-256M is based on Idefics3 (SmolVLM)
        from colpali_engine.models import ColIdefics3, ColIdefics3Processor
        from peft import PeftModel
        local_only = hf_offline_enabled()
        ensure_hf_repo_cached(
            self.config.model_name,
            required_files=["config.json", "adapter_config.json", "tokenizer_config.json"],
        )

        self.processor = ColIdefics3Processor.from_pretrained(
            self.config.model_name,
            local_files_only=local_only,
        )
        # Load the model - if it's an adapter, from_pretrained will load base + adapter automatically
        # provided the adapter_config is present (which it is for vidore/colSmol-256M)
        self.model = ColIdefics3.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.dtype,
            local_files_only=local_only,
        ).to(self.config.device)

        # Keep training stable by default; checkpointing can be re-enabled explicitly.
        if hasattr(self.model, "gradient_checkpointing_disable"):
            try:
                self.model.gradient_checkpointing_disable()
            except AttributeError:
                # Some wrapped models expose partial checkpointing hooks.
                pass

        # Check if it's already a PEFT model
        is_peft = isinstance(self.model, PeftModel) or hasattr(self.model, "peft_config")

        if self.config.use_lora:
            if is_peft:
                print(f"Model {self.config.model_name} is already a PEFT model. Skipping re-application of LoRA.")
                # Ensure it's trainable if needed
                self.model.train()
                # Enable checkpointing only when explicitly requested.
                if self.config.enable_gradient_checkpointing and str(self.config.device).startswith("cuda"):
                    self.model.gradient_checkpointing_enable()
                for name, param in self.model.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True
                
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                print(f"Existing LoRA stats: {trainable:,} / {total:,} params trainable "
                      f"({100 * trainable / total:.2f}%)")
            else:
                self._apply_lora()

        return self

    def _apply_lora(self):
        """Apply LoRA adapters for parameter-efficient fine-tuning."""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"LoRA applied: {trainable:,} / {total:,} params trainable "
              f"({100 * trainable / total:.2f}%)")

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 8,
        with_grad: bool = False,
    ) -> torch.Tensor:
        """Encode text queries into multi-vector embeddings."""
        all_embeddings = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            inputs = self.processor.process_queries(batch).to(self.config.device)
            if with_grad:
                embeddings = self.model(**inputs)
            else:
                with torch.no_grad():
                    embeddings = self.model(**inputs)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def encode_documents(self, images, batch_size: int = 4, with_grad: bool = False) -> torch.Tensor:
        """Encode document images into multi-vector embeddings."""
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor.process_images(batch).to(self.config.device)
            if with_grad:
                embeddings = self.model(**inputs)
            else:
                with torch.no_grad():
                    embeddings = self.model(**inputs)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)
