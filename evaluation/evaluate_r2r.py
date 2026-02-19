"""
Evaluate R2R (reasoning-augmented) ColSmol LoRA checkpoints on ViDoRe benchmarks.

Supports trace modes:
  - use: augment queries with generated reasoning traces
  - none: use raw queries (no traces)
  - shuffle: augment queries with shuffled traces (control)

Outputs MTEB-style JSON results compatible with compare_ablation_results.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import mteb
from mteb.models.abs_encoder import AbsEncoder
from tqdm.auto import tqdm

from src.reasoning.trace_generator import TraceConfig, TraceGenerator
from src.utils import hf_offline_enabled, ensure_hf_repo_cached, set_global_seed


@dataclass
class EvalConfig:
    base_model: str
    processor_name: str
    adapter_path: Optional[str]
    device: str
    dtype: torch.dtype
    batch_size: int
    trace_mode: str
    trace_separator: str
    trace_cache: Optional[str]
    trace_model: str
    trace_device: str
    trace_max_new_tokens: int
    trace_temperature: float
    trace_batch_size: int
    seed: int


class R2RColSmolWrapper(AbsEncoder):
    """MTEB wrapper for ColSmol with optional R2R query augmentation."""

    def __init__(self, cfg: EvalConfig):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self._batch_counter = 0

        from colpali_engine.models import ColIdefics3, ColIdefics3Processor
        from peft import PeftModel

        local_only = hf_offline_enabled()
        ensure_hf_repo_cached(
            cfg.base_model,
            required_files=["config.json"],
        )
        ensure_hf_repo_cached(
            cfg.processor_name,
            required_files=["config.json", "adapter_config.json", "tokenizer_config.json"],
        )

        self.processor = ColIdefics3Processor.from_pretrained(
            cfg.processor_name,
            local_files_only=local_only,
        )

        self.mdl = ColIdefics3.from_pretrained(
            cfg.base_model,
            torch_dtype=cfg.dtype,
            local_files_only=local_only,
        ).to(cfg.device)

        if cfg.adapter_path:
            adapter_path = Path(cfg.adapter_path)
            if adapter_path.is_dir():
                self.mdl = PeftModel.from_pretrained(self.mdl, str(adapter_path)).to(cfg.device)
            else:
                raise FileNotFoundError(f"Adapter path not found: {cfg.adapter_path}")

        self.mdl.eval()

        self.trace_mode = cfg.trace_mode
        self.trace_separator = cfg.trace_separator
        self.trace_cache_path = Path(cfg.trace_cache) if cfg.trace_cache else None
        self.trace_cache: dict[str, str] = {}
        if self.trace_cache_path and self.trace_cache_path.exists():
            with open(self.trace_cache_path) as f:
                data = json.load(f)
            self.trace_cache = {item["query"]: item["trace"] for item in data}

        self.trace_generator: Optional[TraceGenerator] = None
        if self.trace_mode in {"use", "shuffle"}:
            trace_cfg = TraceConfig(
                model_name=cfg.trace_model,
                device=cfg.trace_device,
                max_new_tokens=cfg.trace_max_new_tokens,
                temperature=cfg.trace_temperature,
                batch_size=cfg.trace_batch_size,
            )
            self.trace_generator = TraceGenerator(trace_cfg).load()

    def encode_input(self, inputs):
        return self.mdl(**inputs)

    def _save_trace_cache(self):
        if not self.trace_cache_path:
            return
        self.trace_cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = [{"query": q, "trace": t} for q, t in self.trace_cache.items()]
        with open(self.trace_cache_path, "w") as f:
            json.dump(data, f, indent=2)

    def _get_traces(self, queries: list[str]) -> list[str]:
        if self.trace_mode == "none":
            return [""] * len(queries)

        missing = [q for q in queries if q not in self.trace_cache]
        if missing:
            if self.trace_generator is None:
                raise RuntimeError("Trace generator not initialized for trace mode")
            torch.manual_seed(self.cfg.seed)
            new_traces = self.trace_generator.generate_traces_batch(missing)
            for q, t in zip(missing, new_traces):
                self.trace_cache[q] = t
            self._save_trace_cache()

        traces = [self.trace_cache.get(q, "") for q in queries]
        if self.trace_mode == "shuffle":
            rng = random.Random(self.cfg.seed + self._batch_counter)
            shuffled = traces.copy()
            rng.shuffle(shuffled)
            traces = shuffled
        return traces

    def _augment_queries(self, queries: list[str]) -> list[str]:
        traces = self._get_traces(queries)
        augmented = []
        for q, t in zip(queries, traces):
            t = (t or "").strip()
            if not t:
                augmented.append(q)
            else:
                augmented.append(f"{q}{self.trace_separator}{t}")
        return augmented

    def get_text_embeddings(self, texts, batch_size: int = 32, **kwargs):
        all_embeds = []
        with torch.no_grad():
            for batch in tqdm(texts, desc="Encoding texts"):
                queries = batch["text"]
                augmented = self._augment_queries(queries)
                inputs = self.processor.process_queries(augmented)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))
                self._batch_counter += 1

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def get_image_embeddings(self, images, batch_size: int = 32, **kwargs):
        import torchvision.transforms.functional as F
        from PIL import Image

        all_embeds = []
        with torch.no_grad():
            for batch in tqdm(images, desc="Encoding images"):
                imgs = [
                    F.to_pil_image(b.to(self.device))
                    if not isinstance(b, Image.Image)
                    else b
                    for b in batch["image"]
                ]
                inputs = self.processor.process_images(imgs)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split,
        hf_subset,
        prompt_type=None,
        **kwargs: Any,
    ):
        text_embeddings = None
        image_embeddings = None
        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(inputs, batch_size=self.cfg.batch_size)
        if "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(inputs, batch_size=self.cfg.batch_size)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError("Text and image embeddings length mismatch")
            return text_embeddings + image_embeddings
        if text_embeddings is not None:
            return text_embeddings
        if image_embeddings is not None:
            return image_embeddings
        raise ValueError("No text or image features found")

    def similarity(self, a, b):
        """CPU chunked MaxSim to avoid OOM."""
        a_cpu = a.cpu().float() if a.is_cuda else a.float()
        b_cpu = b.cpu().float() if b.is_cuda else b.float()

        torch.cuda.empty_cache()
        gc.collect()

        num_queries = a_cpu.shape[0]
        num_docs = b_cpu.shape[0]
        chunk_size = 32
        all_scores = []

        for i in range(0, num_queries, chunk_size):
            q_chunk = a_cpu[i:i + chunk_size]
            row_scores = []
            for j in range(0, num_docs, chunk_size):
                d_chunk = b_cpu[j:j + chunk_size]
                sim = torch.einsum("qnd,psd->qpns", q_chunk, d_chunk)
                scores = sim.max(dim=-1)[0].sum(dim=-1)
                row_scores.append(scores)
            all_scores.append(torch.cat(row_scores, dim=1))

        return torch.cat(all_scores, dim=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate R2R checkpoints on ViDoRe")
    parser.add_argument("--benchmark", default="ViDoRe(v1)", help="Benchmark name")
    parser.add_argument("--output-folder", default="results", help="Output folder for MTEB results")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter folder")
    parser.add_argument("--base-model", default="vidore/ColSmolVLM-Instruct-256M-base")
    parser.add_argument("--processor", default="vidore/colSmol-256M")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--trace-mode", choices=["use", "none", "shuffle"], default="use")
    parser.add_argument("--trace-separator", default=" [SEP] ")
    parser.add_argument("--trace-cache", default=None, help="Optional path to cache traces JSON")
    parser.add_argument("--trace-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trace-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trace-max-new-tokens", type=int, default=100)
    parser.add_argument("--trace-temperature", type=float, default=0.7)
    parser.add_argument("--trace-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = EvalConfig(
        base_model=args.base_model,
        processor_name=args.processor,
        adapter_path=args.adapter,
        device=device,
        dtype=torch.bfloat16,
        batch_size=args.batch_size,
        trace_mode=args.trace_mode,
        trace_separator=args.trace_separator,
        trace_cache=args.trace_cache,
        trace_model=args.trace_model,
        trace_device=args.trace_device,
        trace_max_new_tokens=args.trace_max_new_tokens,
        trace_temperature=args.trace_temperature,
        trace_batch_size=args.trace_batch_size,
        seed=args.seed,
    )

    print(f"Loading model with adapter: {args.adapter}")
    model = R2RColSmolWrapper(cfg)

    print(f"Running benchmark: {args.benchmark}")
    tasks = mteb.get_benchmark(args.benchmark)
    cache = mteb.ResultCache(cache_path=args.output_folder)

    all_results = []
    for task in tasks:
        print(f"\nEvaluating: {task.metadata.name}")
        try:
            result = mteb.evaluate(
                model,
                tasks=task,
                cache=cache,
                encode_kwargs={"batch_size": args.batch_size},
                raise_error=False,
            )
        except Exception as e:
            print(f"Task failed with uncaught exception: {e}")
            result = None
        all_results.append(result)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 60)
    print(f"RESULTS - {args.benchmark}")
    print("=" * 60)
    for task, result in zip(tasks, all_results):
        if result is None:
            print(f"\n{task.metadata.name}:")
            print("  ERROR: task execution failed before returning a ModelResult")
            continue

        task_result = None
        if hasattr(result, "task_results") and result.task_results:
            task_result = result.task_results[0]

        task_name = getattr(task_result, "task_name", None) or task.metadata.name
        print(f"\n{task_name}:")

        if task_result is None:
            exceptions = getattr(result, "exceptions", None) or []
            if exceptions:
                print(f"  ERROR: {exceptions[0].exception}")
            else:
                print("  No task result returned")
            continue

        for split, scores in task_result.scores.items():
            if isinstance(scores, list) and len(scores) > 0:
                if isinstance(scores[0], dict) and "ndcg_at_5" in scores[0]:
                    print(f"  {split}: NDCG@5 = {scores[0]['ndcg_at_5']:.4f}")
                elif isinstance(scores[0], dict) and "main_score" in scores[0]:
                    print(f"  {split}: main_score = {scores[0]['main_score']:.4f}")
                else:
                    print(f"  {split}: {scores}")
            else:
                print(f"  {split}: {scores}")


if __name__ == "__main__":
    main()
