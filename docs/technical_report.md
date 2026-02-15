# ColSmol-Reasoning — Technical Audit

15 Feb 2026. Full codebase review covering architecture, training, losses, data, and experimental state.

---

## 1. Architecture

### 1.1 Backbone

Student is **ColSmol-256M** — a compact SmolVLM (Idefics3) that pairs a SigLIP vision encoder with a small LM backbone. Loaded from `vidore/colSmol-256M` via `colpali_engine` (`ColIdefics3`).

SigLIP produces patch-level embeddings from the image. These get fed into the LM where they're contextualised with any text tokens. Output is one embedding per token/patch — the multi-vector representation needed for late interaction.

Teachers for distillation: **ColPali-v1.3** (3B) and **ColQwen2-v1.0** (2B). Always frozen, default to CPU to leave GPU free for the student.

### 1.2 Projection

Single linear layer maps LM hidden dim → **D = 128** (standard ColBERT dimensionality). Inherited from the pretrained checkpoint, not re-implemented. No MLP, no nonlinearity — just a linear projection.

### 1.3 MaxSim (Late Interaction)

$$\text{score}(Q, D) = \sum_{i} \max_{j} \langle q_i, d_j \rangle$$

Implemented in multiple places. Training code builds the $B \times B$ score matrix iteratively (one query at a time) to keep memory in check — the full 4D tensor doesn't fit at useful batch sizes. Eval code chunks queries/docs into groups of 32 on CPU for the same reason.

### 1.4 LoRA Setup

`vidore/colSmol-256M` already ships as a PEFT model. The wrapper detects this and reuses the existing adapter.

- Targets: `q_proj`, `v_proj`
- Rank 32, alpha 32, dropout 0.05
- Gradient checkpointing off (deliberate — stability)

---

## 2. The Two Ideas

### 2.1 MID — MaxSim Interaction Distillation

Three-part loss:

$$L = \alpha \cdot L_{\text{contrastive}} + \beta \cdot L_{\text{interaction}} + \gamma \cdot L_{\text{ranking}}$$

- **Contrastive:** InfoNCE over in-batch MaxSim scores.
- **Interaction:** KL-div between teacher and student token-level attention maps — which query token should match which doc patch. Scaled by $T^2$.
- **Ranking:** Pairwise hinge loss. If teacher prefers doc $i$ over $j$, student must too, by at least margin 0.1.

Weights: $\alpha=1.0$, $\beta=1.0$, $\gamma=0.5$.

No architectural changes to the student. The novelty is entirely in the loss signal from the teacher.

**Status: fully coded, never run on real data.**

### 2.2 R2R — Reason-to-Retrieve

Queries get augmented with a reasoning trace before encoding:

`[query] [SEP] [reasoning_trace]`

The trace describes what visual evidence to look for ("bar chart", "Q1–Q4 columns", etc.). These extra tokens produce additional embeddings that participate in MaxSim, giving the model explicit grounding hints it otherwise can't infer at 256M scale.

**Trace generation:** Frozen Qwen2.5-0.5B-Instruct, prompted to output 2–3 sentences of visual grounding cues. ~7.8k traces pre-computed for training; cached on-the-fly at eval.

**Training loss:** Plain InfoNCE with in-batch negatives on augmented queries. Temperature 0.1. Embeddings L2-normalised in fp32 (stability fix for bf16 overflow).

**Ablation controls:**
- `use` — aligned traces (experimental condition)
- `none` — query only (baseline)
- `shuffle` — mismatched traces (length control)

If `use > none`: reasoning helps. If `use > shuffle`: it's the content, not just extra tokens.

### 2.3 Not Implemented

- No hard negative mining — in-batch negatives only.
- No "thinking tokens" or architectural modifications.
- No auxiliary reasoning losses.
- **No combined R2R+MID trainer** — the two ideas live in separate pipelines despite the ICDAR proposal framing their combination as the main contribution.

---

## 3. Training Config

### R2R (primary track)

| Param | Value |
|---|---|
| Model | ColSmol-256M + LoRA (r=32) |
| LR | 5e-6 |
| Batch / accum / effective | 2 / 8 / 16 |
| Epochs | 3 |
| Temperature | 0.1 |
| Optimizer | AdamW (wd=0.01) |
| Precision | bf16 |
| Training data | `vidore/colpali_train_set` (streaming, ~118k pairs) |
| Traces | ~7.8k from Qwen2.5-0.5B |

### MID (not yet run)

| Param | Value |
|---|---|
| Teacher | ColPali-v1.3 (3B, CPU) |
| Student | ColSmol-256M + LoRA (r=32, GPU) |
| LR | 2e-5 |
| Batch / accum / effective | 2 / 8 / 16 |
| Epochs | 1 |
| KD temperature | 2.0 |
| Weights (α, β, γ) | 1.0, 1.0, 0.5 |

### Evaluation benchmarks

- **ViDoRe v1:** 10 subtasks (ArxivQA, DocVQA, InfoVQA, TabFQuAD, TATDQA, ShiftProject, 4× SyntheticDocQA)
- **ViDoRe v2:** 4 subtasks (ESG HL, ESG Reports, Economics, BioMedical)
- **ViDoRe v3:** Referenced in scripts and the ICDAR proposal but no definition in the data loader — relies on MTEB resolving the name at runtime

---

## 4. Results

### 4.1 Baseline (ViDoRe v1, nDCG@5 × 100)

| Task | ColSmol | ColPali (3B) | ColQwen2 (2B) |
|---|---:|---:|---:|
| ArxivQA | 72.0 | 79.1 | 86.4 |
| DocVQA | 55.7 | 54.4 | 56.2 |
| InfoVQA | 82.5 | 81.8 | 89.8 |
| TabFQuAD | 62.1 | 83.9 | 88.7 |
| TATDQA | 74.5 | 65.8 | 75.2 |
| ShiftProject | 56.2 | 73.2 | 85.7 |
| AI | 94.9 | 96.2 | 98.8 |
| Energy | 91.8 | 91.0 | 94.8 |
| Gov. | 92.6 | 92.7 | 93.6 |
| Health. | 95.1 | 94.4 | 97.3 |
| **Average** | **77.7** | **81.3** | **86.6** |

Gap to ColPali is 3.6 pts average, but much worse on hard tasks: TabFQuAD (−21.8), ShiftProject (−17.0), ArxivQA (−7.1).

### 4.2 Baseline (ViDoRe v2 English)

| Task | nDCG@5 |
|---|---:|
| ESG Reports (HL) | 0.418 |
| ESG Reports | 0.456 |
| Economics | 0.532 |
| BioMedical | 0.504 |
| **Average** | **0.477** |

Much harder than v1. ColSmol barely clears 48%.

### 4.3 R2R Experiment — Training Done, Results Disappointing

All three ablation runs trained to completion (seed 42):
- `r2r_use_seed42` — done
- `r2r_none_seed42` — done
- `r2r_shuffle_seed42` — done

The 9-run evaluation matrix (3 modes × 3 benchmarks) was launched but had not fully completed at audit time. From partial evaluation outputs and the experiment log, the picture is not encouraging: **the R2R ablation has not produced the clear `use > none > shuffle` ordering we expected.** The deltas between trace modes are small and inconsistent across tasks, making it hard to claim that reasoning traces meaningfully help retrieval at this scale. The experiment log's quantitative summary table remains filled with "TBD" entries, and no strong per-task wins have been reported.

This is the central negative result of the project so far. The hypothesis — that externalised reasoning traces would let a small model punch above its weight — has not been validated with the current setup.

### 4.4 Loss Curves

No training logs or metrics CSVs are committed. The framework writes `train_metrics.csv` per run, but these sit in git-ignored checkpoint directories. No loss curve visualisation exists.

### 4.5 Latency / Memory

No benchmarks implemented. The project targets 8GB VRAM. R2R inference fits in ~6GB (0.5B trace LLM + 256M retriever + headroom). MaxSim similarity is offloaded to CPU in chunks during eval to avoid OOM.

---

## 5. What's Wrong

### 5.1 Bottlenecks

**Only 1 negative per forward pass.** Physical batch is 2. Gradient accumulation gets the effective batch to 16 for optimizer steps, but it does not increase the number of negatives visible to the contrastive loss. Each forward pass contrasts against exactly 1 in-batch negative. This is extremely weak supervision for InfoNCE.

**Teacher-student sequence length mismatch (MID).** ColPali and ColSmol produce different numbers of tokens. The interaction loss (KL-div over attention maps) assumes matching $T_q$ and $T_d$ between teacher and student. It will break or produce garbage if they differ. Not handled anywhere.

**Memory-bound MaxSim.** Score matrix is built with a Python loop over batch items. Correct but slow, and limits practical batch scaling.

### 5.2 Dead Code and Missing Pieces

1. **MID — fully coded, zero runs.** Teacher wrapper, 3 loss components, trainer, training script — all there, never executed beyond unit tests.
2. **Hard negative support in loss function** — the contrastive loss accepts explicit negatives as $(B, N, T_d, D)$. Never called with real negatives.
3. **Combined R2R+MID** — the ICDAR proposal pitches this as the main result. No combined trainer exists.
4. **Warmup scheduler** — config says `warmup_steps=100`, neither trainer implements a scheduler. Training jumps to full LR from step 0.
5. **WandB** — config fields exist, no integration code.
6. **Gradient checkpointing** — available but force-disabled.

### 5.3 Bugs / Concerns

- **NaN fix is R2R-only.** The fp32 normalisation + epsilon guard was added to the R2R trainer after hitting NaN issues. The distillation trainer has neither safeguard. MID will likely NaN in bf16.
- **Double softmax in interaction loss.** The teacher's `get_maxsim_attention` already produces a softmax'd distribution. The loss function then does `log → / T → softmax` on it again. This flattens the teacher signal.
- **Temperature overload in MID.** The KD temperature (2.0) is also used for the contrastive loss. InfoNCE and KD temperatures serve different roles and shouldn't share a value.

---

## 6. Tried vs. Untested

| What | Code | Run |
|---|---|---|
| Baseline eval (v1, v2) | Done | Done — numbers in repo |
| R2R trace generation | Done | Done — ~7.8k traces |
| R2R training (3 ablations) | Done | Done — all converged |
| R2R eval | Done | Partial — results weak so far |
| MID losses | Done | Unit tests only |
| MID training | Done | Never run |
| Combined R2R+MID | Not built | — |
| Hard negatives | Half-built | Never used |
| LR warmup | Not built | — |

---

## Bottom Line

The project set out to close the gap between ColSmol (256M) and ColPali (3B) through two ideas: reasoning-augmented queries (R2R) and interaction-level distillation (MID).

R2R was the focus. Three controlled training runs completed; evaluation was in progress. The early signal is discouraging — trace augmentation hasn't produced clear gains over the query-only baseline, undermining the core hypothesis. The main suspects: a batch size of 2 starves the contrastive loss of negatives, and a 0.5B trace generator may not produce high-enough quality reasoning to actually help.

MID exists in code but has never been trained. It also has a latent dimension-mismatch bug in the interaction loss and missing numerical stability safeguards.

The combined experiment (R2R+MID) that the ICDAR proposal positions as the headline result hasn't been started. With the R2R track underperforming and MID untested, the project is behind where it needs to be for the Feb 27 paper deadline.
