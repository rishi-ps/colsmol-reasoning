"""
Reasoning trace generator for Reason-to-Retrieve (R2R) — Idea 2.

Generates visual grounding reasoning traces for queries using a small LLM.
These traces tell the retriever WHAT visual elements to look for.

Example:
  Query: "What was the Q3 revenue growth?"
  Trace: "I need to find a page with a financial table showing quarterly data.
          Look for bar charts or tables with columns labeled Q1-Q4 and rows
          with revenue figures."
"""

import torch
from typing import Optional
from dataclasses import dataclass


# Default prompt template for generating visual grounding traces
DEFAULT_TRACE_PROMPT = """You are helping a visual document retriever find the right page.

Given a query, describe what visual elements the retriever should look for in
document pages. Focus on:
- What type of document/page layout to expect (table, chart, diagram, text block)
- What visual patterns to match (column headers, axis labels, specific text sections)
- What spatial layout cues are relevant (top of page, sidebar, footer)

Keep your response to 2-3 sentences. Be specific about visual elements.

Query: {query}
Visual search reasoning:"""


@dataclass
class TraceConfig:
    """Configuration for reasoning trace generation."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    max_new_tokens: int = 100
    temperature: float = 0.7
    prompt_template: str = DEFAULT_TRACE_PROMPT
    batch_size: int = 8


class TraceGenerator:
    """Generates reasoning traces for queries using a small LLM.

    Phase 1 (training): Generate traces for all training queries offline.
    Phase 3 (inference): Generate traces on-the-fly with frozen small LLM.
    """

    def __init__(self, config: Optional[TraceConfig] = None):
        self.config = config or TraceConfig()
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load the reasoning LLM."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.dtype,
        ).to(self.config.device)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"Trace generator loaded: {self.config.model_name}")
        return self

    @torch.no_grad()
    def generate_trace(self, query: str) -> str:
        """Generate a reasoning trace for a single query.

        Args:
            query: The search query

        Returns:
            Reasoning trace string
        """
        prompt = self.config.prompt_template.format(query=query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only the generated tokens (skip the prompt)
        generated = outputs[0, inputs["input_ids"].shape[1]:]
        trace = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return trace

    @torch.no_grad()
    def generate_traces_batch(self, queries: list[str]) -> list[str]:
        """Generate reasoning traces for a batch of queries.

        Args:
            queries: List of search queries

        Returns:
            List of reasoning traces
        """
        all_traces = []
        for i in range(0, len(queries), self.config.batch_size):
            batch = queries[i:i + self.config.batch_size]
            prompts = [self.config.prompt_template.format(query=q) for q in batch]

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.config.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            for j, output in enumerate(outputs):
                prompt_len = inputs["input_ids"].shape[1]
                generated = output[prompt_len:]
                trace = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                all_traces.append(trace)

            print(f"Generated {len(all_traces)}/{len(queries)} traces")

        return all_traces

    def save_traces(self, queries: list[str], traces: list[str], output_path: str):
        """Save query-trace pairs to a JSON file."""
        import json
        data = [{"query": q, "trace": t} for q, t in zip(queries, traces)]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} traces to {output_path}")

    @staticmethod
    def load_traces(path: str) -> list[dict]:
        """Load previously generated traces."""
        import json
        with open(path) as f:
            return json.load(f)
