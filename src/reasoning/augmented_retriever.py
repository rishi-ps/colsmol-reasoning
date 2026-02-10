"""
Reasoning-augmented retriever for Reason-to-Retrieve (R2R) — Idea 2.

Augments queries with reasoning traces before encoding:
  Input: [query] + [SEP] + [reasoning_trace]
  → Additional embedding vectors from reasoning tokens
  → MaxSim matches reasoning tokens to doc patches
  → e.g., "bar chart", "Q1-Q4 columns" directly match relevant visual patches
"""

import torch
from typing import Optional
from dataclasses import dataclass

from src.models.colsmol import ColSmolWrapper
from src.reasoning.trace_generator import TraceGenerator


@dataclass
class R2RConfig:
    """Configuration for reasoning-augmented retrieval."""
    separator: str = " [SEP] "
    max_query_length: int = 512  # Max tokens for augmented query


class ReasoningAugmentedRetriever:
    """Augments queries with reasoning traces before ColSmol encoding.

    This implements Phase 2 of R2R: the retriever processes
    [query] + [SEP] + [reasoning_trace] as a single input,
    generating additional embedding vectors from the reasoning tokens.
    """

    def __init__(
        self,
        retriever: ColSmolWrapper,
        trace_generator: Optional[TraceGenerator] = None,
        config: Optional[R2RConfig] = None,
    ):
        self.retriever = retriever
        self.trace_generator = trace_generator
        self.config = config or R2RConfig()

    def augment_query(self, query: str, trace: str) -> str:
        """Combine query with reasoning trace.

        Args:
            query: Original search query
            trace: Reasoning trace from TraceGenerator

        Returns:
            Augmented query string
        """
        return f"{query}{self.config.separator}{trace}"

    def augment_queries_batch(
        self, queries: list[str], traces: Optional[list[str]] = None
    ) -> list[str]:
        """Augment a batch of queries with reasoning traces.

        If traces are not provided and a trace_generator is available,
        generates them on-the-fly (Phase 3 inference mode).

        Args:
            queries: List of search queries
            traces: Optional pre-computed traces

        Returns:
            List of augmented query strings
        """
        if traces is None:
            if self.trace_generator is None:
                raise ValueError(
                    "No traces provided and no trace_generator available. "
                    "Either provide traces or initialize with a TraceGenerator."
                )
            traces = self.trace_generator.generate_traces_batch(queries)

        return [self.augment_query(q, t) for q, t in zip(queries, traces)]

    def encode_augmented_queries(
        self,
        queries: list[str],
        traces: Optional[list[str]] = None,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Encode augmented queries into multi-vector embeddings.

        Args:
            queries: List of search queries
            traces: Optional pre-computed traces
            batch_size: Encoding batch size

        Returns:
            (N, T, D) tensor of augmented query embeddings
        """
        augmented = self.augment_queries_batch(queries, traces)
        return self.retriever.encode_queries(augmented, batch_size=batch_size)

    def retrieve(
        self,
        queries: list[str],
        doc_embeddings: torch.Tensor,
        traces: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> list[list[int]]:
        """Retrieve top-k documents for augmented queries.

        Args:
            queries: Search queries
            doc_embeddings: (M, Td, D) pre-computed document embeddings
            traces: Optional pre-computed reasoning traces
            top_k: Number of documents to retrieve

        Returns:
            List of lists of document indices
        """
        query_embeddings = self.encode_augmented_queries(queries, traces)

        results = []
        for q_emb in query_embeddings:
            # MaxSim: for each query token, max similarity over doc tokens
            # q_emb: (Tq, D), doc_embeddings: (M, Td, D)
            sim = torch.einsum("qd,mtd->mqt", q_emb, doc_embeddings)  # (M, Tq, Td)  # noqa: E501
            maxsim = sim.max(dim=-1).values.sum(dim=-1)  # (M,)
            top_indices = maxsim.topk(min(top_k, len(maxsim))).indices.tolist()
            results.append(top_indices)

        return results
