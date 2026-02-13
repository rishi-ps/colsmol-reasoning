# R2R + MID: Nitty-Gritty Technical Explanation

This document explains, in practical detail, how our two methods work:

1. `Reason-to-Retrieve (R2R)` for query understanding
2. `MaxSim Interaction Distillation (MID)` for token-level matching transfer

It is written against the current codebase, not just the conceptual idea.

---

## 1) Why this is needed in our setup

Our retriever is a late-interaction model (`ColSmol`) that encodes:

- a query into token vectors: `Q = [q_1, ..., q_Tq]`
- a document page into token/patch vectors: `D = [d_1, ..., d_Td]`

Scoring is MaxSim-style:

`score(Q, D) = sum_i max_j (q_i dot d_j)`

This gives strong retrieval behavior but depends heavily on:

- whether query tokens represent the *right intent* (query understanding),
- whether token-token interactions are aligned with good matching behavior.

R2R targets the first problem. MID targets the second.

---

## 2) R2R in detail (query-side reasoning augmentation)

## 2.1 Core idea

Instead of feeding only raw query text, we feed:

`[query] + [SEP] + [reasoning_trace]`

The trace describes what visual evidence should be matched:

- layout cues (table/chart/section)
- lexical cues (headers/keywords)
- spatial cues (top, sidebar, footer)

This creates extra query tokens that can participate in MaxSim matching.

Implementation entry points:

- `src/reasoning/trace_generator.py`
- `src/reasoning/augmented_retriever.py`
- `src/reasoning/trainer.py`
- `scripts/train_r2r.py`
- `evaluation/evaluate_r2r.py`

## 2.2 Trace generation (what exactly is generated)

`TraceGenerator` uses `Qwen/Qwen2.5-0.5B-Instruct` with a prompt that asks for visual grounding hints.

Relevant details:

- prompt template: `DEFAULT_TRACE_PROMPT` in `src/reasoning/trace_generator.py`
- generation settings:
  - `max_new_tokens=100`
  - `temperature=0.7`
  - batched generation support
- model is frozen (`requires_grad=False`)

The generator is used in two modes:

1. Offline/precompute traces for training (Phase 1)
2. On-the-fly trace generation at evaluation/inference for `trace_mode=use` or `shuffle` (Phase 3 path in `evaluation/evaluate_r2r.py`)

## 2.3 How queries are augmented

`ReasoningAugmentedRetriever.augment_query()` in `src/reasoning/augmented_retriever.py` does:

- if trace exists: `query + " [SEP] " + trace`
- if empty trace: fallback to query only

Batch augmentation:

- `augment_queries_batch(queries, traces)`
- if `traces` is not provided and generator exists, it auto-generates traces

## 2.4 R2R training objective (actual loss in code)

Training is in `R2RTrainer` (`src/reasoning/trainer.py`).

For each batch of `(query, trace, image)`:

1. Build augmented query text
2. Encode queries and images with ColSmol
3. L2-normalize embeddings in float32 (`eps=1e-6`) for numerical stability
4. Build in-batch score matrix `S` where:
   - `S[i,j] = MaxSim(query_i, doc_j)`
5. Apply InfoNCE-style cross-entropy:
   - labels are diagonal (`i` should match `i`)
   - `loss = CE(S / temperature, labels)`

Key implementation notes:

- in-batch negatives only
- requires `batch_size >= 2`
- uses iterative score matrix construction (memory-safe vs fully vectorized `B x B`)
- gradient accumulation + clipping are enabled

## 2.5 Why `use/none/shuffle` matters

In `scripts/train_r2r.py`, `R2RDataset` supports:

- `use`: aligned trace map
- `none`: trace is empty string
- `shuffle`: traces are deterministically shuffled across queries

Interpretation logic:

- `use > none` means adding reasoning helps.
- `use > shuffle` means trace *content alignment* matters, not just longer text.

## 2.6 Evaluation path for R2R

`evaluation/evaluate_r2r.py` wraps ColSmol for MTEB benchmarks.

Important mechanics:

- trace cache support (`--trace-cache`) to avoid regenerating traces repeatedly
- same augmentation logic used at eval-time (`trace_mode`)
- per-task JSON outputs are written in MTEB format
- a compatibility fix was added for result field naming (`task_name` vs `task_names`)

---

## 3) MID in detail (interaction-level distillation)

## 3.1 Core idea

Classic distillation often focuses on embedding regression or final score matching only.
For late interaction, we care about *which query tokens match which doc tokens*.

MID therefore distills interaction structure, not just final outputs.

Implementation entry points:

- `src/models/teacher.py`
- `src/distillation/losses.py`
- `src/distillation/trainer.py`
- `scripts/train_distillation.py`

## 3.2 Teacher and student roles

Student:

- `ColSmolWrapper` (`src/models/colsmol.py`)
- trainable, often with LoRA

Teacher:

- `TeacherModelWrapper` (`src/models/teacher.py`)
- supports `colpali` or `colqwen`
- loaded frozen (`requires_grad=False`)
- can run on CPU to save GPU memory

## 3.3 What signals are distilled

MID uses three loss components from `src/distillation/losses.py`:

1. `L_contrastive`
2. `L_interaction`
3. `L_ranking`

Total:

`L_total = alpha * L_contrastive + beta * L_interaction + gamma * L_ranking`

where `(alpha, beta, gamma)` are in `DistillationConfig`.

## 3.4 Loss components in plain language + formula

### A) Contrastive loss (`L_contrastive`)

Goal:

- positive pair `(q_i, d_i)` should score higher than negatives.

In trainer implementation, we currently use in-batch negatives and build a `B x B` score matrix (same style as R2R trainer), then CE over diagonal targets.

### B) Interaction loss (`L_interaction`)

Goal:

- match teacher token-level attention over doc tokens for each query token.

Teacher attention source:

- `TeacherModelWrapper.get_maxsim_attention()`
- computes token similarity and softmax over doc-token dimension

Student attention:

- `student_sim = student_q @ student_d^T`
- `log_softmax(student_sim / T)`

Teacher target:

- soft target distribution with temperature

Loss:

- `KL(student || teacher) * T^2`

This is the core "interaction distillation" piece.

### C) Ranking loss (`L_ranking`)

Goal:

- if teacher prefers document `i` over `j`, student should preserve this ordering with margin.

Pairwise hinge style:

- `max(0, margin - (s_i - s_j))` for teacher-preferred pairs

This distills relative ordering behavior, not only absolute scores.

## 3.5 Distillation training loop behavior

`DistillationTrainer.train_step()` does:

1. Teacher forward pass (no grad) to get:
   - query/doc embeddings
   - teacher attention maps
   - teacher scores / score matrix
2. Student forward pass (with grad) on same batch
3. Compute:
   - in-batch contrastive
   - interaction KL
   - ranking loss
4. Weighted sum -> backward -> optimizer step with accumulation/clipping

Important practical detail:

- the trainer builds score matrices iteratively to reduce memory pressure.

---

## 4) How R2R and MID complement each other

R2R improves *what the query asks the retriever to look for*.

- better query token content
- better grounding hints

MID improves *how the student matches tokens once encoded*.

- better token-token interaction pattern
- teacher-guided matching behavior

Intuition:

- R2R is mostly query-side guidance.
- MID is mostly interaction/representation transfer.
- Combined system can improve both intent grounding and matching quality.

---

## 5) Current code-level status (important for interpretation)

What is fully operational:

- R2R training and ablations (`use/none/shuffle`)
- multi-benchmark R2R eval pipeline
- per-benchmark comparison tooling
- cross-benchmark summary tooling

What exists and is trainable for MID:

- teacher wrapper
- student wrapper with LoRA
- MID loss components
- distillation trainer + script

What to keep in mind when reporting:

- Distillation trainer currently uses in-batch score matrices; explicit mined negatives are not yet implemented.
- The project narrative should separate:
  - completed R2R evidence
  - MID as complementary ongoing/extended evidence (unless full MID runs are completed).

---

## 6) End-to-end data flow summary

### R2R flow

1. Query -> trace generation (or cached trace)
2. Query augmentation: `query [SEP] trace`
3. ColSmol encodes augmented query and document image
4. MaxSim scoring + contrastive training
5. Evaluate on ViDoRe with `use/none/shuffle`
6. Compare deltas per task and benchmark

### MID flow

1. Same query/document batch sent to teacher and student
2. Teacher provides:
   - embeddings
   - interaction attention map
   - score preferences
3. Student provides embeddings/scores
4. Compute weighted MID objective
5. Update student only

---

## 7) Practical interpretation guide for writing

When explaining results:

- If `use > none`: reasoning traces add useful grounding information.
- If `use > shuffle`: gains depend on semantic alignment of traces, not length.
- If MID improves over base student: teacher interaction priors transfer.
- If R2R+MID improves over either alone: components are complementary.

When being conservative:

- avoid claiming MID final effectiveness until full runs and ablations are complete.
- keep claims tied to the exact runs that are finished and reproducible.
