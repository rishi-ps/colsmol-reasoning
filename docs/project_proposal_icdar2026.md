# Project Proposal: Reasoning-Enhanced Distillation for Compact Visual Document Retrieval

## Working Title

**Bridging the Gap: Reasoning-to-Retrieve with Interaction Distillation for Compact Visual Document Retrievers**

## Venue and Timeline

- Target venue: **ICDAR 2026**
- Abstract deadline: **February 13, 2026, 23:59 AOE**
- Full paper deadline: **February 27, 2026, 23:59 AOE**

## Problem Statement

Compact visual retrievers are attractive for deployment because they are cheaper and faster, but they still underperform larger models on multi-domain document retrieval benchmarks. The key challenge is to improve small-model retrieval quality without losing efficiency.

## Proposed Approach

We propose a two-part method that addresses complementary weaknesses in compact models:

1. **Reason-to-Retrieve (R2R)** for better query understanding  
   We augment each query with a short reasoning trace that describes what visual evidence should be matched in documents.

2. **MaxSim Interaction Distillation (MID)** for better token-level matching  
   We distill from a larger teacher by transferring interaction patterns that matter for late-interaction scoring, instead of relying only on standard embedding-level supervision.

## Core Research Questions

1. Can reasoning-augmented queries improve retrieval for compact visual retrievers without increasing model size?
2. Does trace content quality matter beyond added text length?
3. Can interaction-level distillation transfer useful matching behavior from larger visual retrievers to compact models?
4. Are R2R and MID complementary when combined?

## Technical Plan

### A. R2R Track

- Generate concise reasoning traces for queries.
- Train and evaluate with three controlled modes:
  - `use`: query + aligned trace
  - `none`: query only
  - `shuffle`: query + mismatched trace
- Interpret gains with `use > none` and `use > shuffle`.

### B. MID Track

- Teacher: larger visual retriever (inference-only during distillation).
- Student: compact retriever (256M scale).
- Distillation objective combines:
  - retrieval/contrastive signal
  - interaction-level supervision (matching behavior)
  - optional ranking consistency
- Evaluate whether MID reduces the gap to stronger teacher-scale behavior.

### C. Combined Track (R2R + MID)

- Train/evaluate a combined model using reasoning augmentation and interaction distillation.
- Test whether gains are additive or partially overlapping.

## Experimental Design

### Benchmarks

- ViDoRe(v1), ViDoRe(v2), ViDoRe(v3)

### Main Comparisons

- Baseline compact retriever
- +R2R (`use`)
- +MID
- +R2R+MID
- Controls: `none` and `shuffle`

### Metrics

- Primary: nDCG@5
- Secondary: per-task deltas and benchmark-wise averages

### Analysis

- Task-level where reasoning helps most vs least
- Task-level where MID helps most vs least
- Combined model vs individual components
- Failure cases and residual gap to larger models

## Expected Contribution

This project aims to show that compact visual retrievers can be improved in a principled way by combining:

- explicit query-side reasoning (R2R), and
- teacher-guided interaction transfer (MID),

while preserving deployment-friendly model size and inference cost.
