# ColSmol-Reasoning: Technical Audit Report

**Date:** 15 February 2026
**Scope:** Full codebase audit of the ColSmol-Reasoning project — architecture, training, losses, data, evaluation, and experimental status.

---

## 1. System Architecture (Reverse Engineered)

### 1.1 Backbone Analysis

The student retriever is **ColSmol-256M**, loaded via the `colpali_engine` library as a `ColIdefics3` model. This is a compact variant of **SmolVLM** (Idefics3 architecture), which pairs a **SigLIP vision encoder** with a small language model backbone. The pretrained checkpoint used is `vidore/colSmol-256M`.

Image inputs are processed by the SigLIP-based vision encoder, which produces patch-level embeddings. These patch embeddings are fed into the language model backbone, where they are contextualised alongside any text tokens (e.g., query text). The model outputs one embedding vector per token/patch in the sequence, preserving the multi-vector (late interaction) structure required for ColBERT-style retrieval.

The teacher models supported for distillation are **ColPali-v1.3** (3B, PaliGemma-based) and **ColQwen2-v1.0** (2B, Qwen2-VL-based), both loaded through `colpali_engine` as `ColPali`/`ColQwen2` classes respectively. The teacher is always frozen and runs inference-only, with CPU offloading as the default to preserve GPU VRAM for the student.

### 1.2 Projection Layer

The `ColIdefics3` model, as inherited from the `colpali_engine` library, contains an internal linear projection head that maps each output token embedding from the language model's hidden dimension down to **D = 128**, consistent with the original ColBERT dimensionality. This projection is not re-implemented in the project codebase; it is inherited from the pretrained `vidore/colSmol-256M` checkpoint and the `colpali_engine` model class.

No MLP, non-linearity, or additional learned layers are added on top of this projection in the project code. The mapping from hidden states to the 128-dimensional retrieval space is a single linear layer.

### 1.3 Late Interaction Mechanism (MaxSim)

The late interaction scoring is implemented explicitly in several locations across the codebase, all following the same formulation:

**MaxSim Score:**
$$\text{score}(Q, D) = \sum_{i=1}^{T_q} \max_{j=1}^{T_d} \langle q_i, d_j \rangle$$

where $Q = [q_1, \dots, q_{T_q}]$ are query token embeddings and $D = [d_1, \dots, d_{T_d}]$ are document patch embeddings, both in $\mathbb{R}^{128}$.

In the training code, the score matrix for a batch of $B$ queries and $B$ documents is computed iteratively (one query at a time against all documents) to control GPU memory usage, since the full $B \times B \times T_q \times T_d$ tensor would be expensive. The einsum pattern used is `"qtd,bpd->btp"` followed by max over the doc-token dimension and sum over the query-token dimension.

In the evaluation MTEB wrapper, MaxSim is computed with a 4D einsum `"qnd,psd->qpns"`, then max over the last dimension (doc-patches) and sum over query-tokens, processed in chunks of 32 queries and 32 documents to remain within CPU/GPU memory limits.

### 1.4 Model Loading and Adapter Strategy

LoRA (Low-Rank Adaptation) is the default fine-tuning strategy. The `ColSmolWrapper` detects whether the loaded model is already a PEFT (Parameter-Efficient Fine-Tuning) model — which `vidore/colSmol-256M` is, since it ships with `adapter_config.json` — and either reuses the existing adapter or applies a new one.

LoRA configuration:
- **Target modules:** `q_proj`, `v_proj` (query and value projections in attention)
- **Rank:** 32 (production), 16 (debug/test), 8 (CPU debug)
- **Alpha:** 32
- **Dropout:** 0.05
- **Bias:** None
- **Task type:** `FEATURE_EXTRACTION`

Gradient checkpointing is explicitly disabled by default and only re-enabled on request, a deliberate choice to keep training stable.

---

## 2. The "Reasoning" Component (Novelty Analysis)

The project implements two distinct research ideas. Idea 1 (MID) is a distillation approach; Idea 2 (R2R) introduces reasoning-augmented retrieval. Both are fully coded, but they are at different stages of empirical validation.

### 2.1 Idea 1 — MaxSim Interaction Distillation (MID)

#### Training Objective

The MID loss is a weighted combination of three components:

$$L_{\text{total}} = \alpha \cdot L_{\text{contrastive}} + \beta \cdot L_{\text{interaction}} + \gamma \cdot L_{\text{ranking}}$$

**A. Contrastive Loss ($L_{\text{contrastive}}$):** An InfoNCE-style cross-entropy over a $B \times B$ MaxSim score matrix using in-batch negatives. The positive is the diagonal entry $(q_i, d_i)$; all off-diagonal entries are treated as negatives. Temperature-scaled softmax is applied before cross-entropy.

**B. Interaction Loss ($L_{\text{interaction}}$):** KL-divergence between the teacher's and student's token-level attention maps. The teacher's attention is computed as $\text{softmax}(\frac{q \cdot d^T}{\sqrt{D}})$ over the doc-token dimension. The student's log-softmax of its raw similarity matrix (scaled by temperature $T$) is compared to the teacher's softened distribution. The loss is scaled by $T^2$ following standard knowledge distillation convention.

**C. Ranking Loss ($L_{\text{ranking}}$):** A pairwise margin-based hinge loss. For every document pair $(d_i, d_j)$ where the teacher prefers $d_i$ over $d_j$, the student must also respect this ordering by at least a margin (default 0.1). The loss is $\max(0, \text{margin} - (s_i - s_j))$ averaged over valid teacher-preferred pairs.

Default weights: $\alpha = 1.0$, $\beta = 1.0$, $\gamma = 0.5$.

#### Architecture Tweaks for MID

No extra tokens, attention masks, or adapter layers are added to the student for MID. The distillation is purely a training objective change: the student's forward pass is identical to the baseline ColSmol forward pass. The novelty is entirely in what signals the loss function extracts from the teacher.

### 2.2 Idea 2 — Reason-to-Retrieve (R2R)

#### Core Mechanism

Queries are augmented before encoding by concatenating a reasoning trace:

`[query] + " [SEP] " + [reasoning_trace]`

This augmented string is tokenised and passed through ColSmol's text encoder. The reasoning tokens produce additional embedding vectors that participate in MaxSim scoring, allowing the model to match descriptive visual reasoning terms (e.g., "bar chart", "Q1–Q4 columns") directly against relevant document patches.

#### Trace Generation

Traces are generated by a frozen **Qwen/Qwen2.5-0.5B-Instruct** language model. A carefully designed prompt asks the model to describe what visual elements and layout cues the retriever should look for, constrained to 2–3 sentences. Generation parameters: `max_new_tokens=100`, `temperature=0.7`, `do_sample=True`.

Traces are generated in two modes:
- **Offline (Phase 1):** Pre-computed for all training queries and saved as JSON. The main trace file is approximately 7,800 traces.
- **Online (Phase 3):** Generated on-the-fly at inference time using the frozen 0.5B LLM, with trace caching to avoid redundant generation.

#### Training Objective for R2R

R2R training uses a single **InfoNCE loss with in-batch negatives**, identical to the contrastive component of MID but applied to augmented queries. Temperature is set to 0.1 (from the R2R config, reduced from the initial 0.02 to prevent NaN issues). Embeddings are L2-normalised in float32 with an epsilon guard of $1 \times 10^{-6}$ — a numerical stability fix applied specifically to address fp16/bf16 overflow in the softmax of the score matrix.

No auxiliary reasoning losses, thinking tokens, or additional adapter layers are introduced. The reasoning component is purely a **data-level augmentation** of the query input; the model architecture is unchanged.

#### Ablation Controls

The training script supports three trace modes for controlled comparison:
- **`use`:** Aligned query-trace pairs (the experimental condition)
- **`none`:** Empty trace; query-only baseline
- **`shuffle`:** Traces are deterministically shuffled across queries (fixed by seed), serving as a length-control to test whether trace *content* matters or just additional tokens

### 2.3 What is NOT Implemented

- No explicit hard negative mining. Both MID and R2R rely entirely on in-batch negatives.
- No "thinking tokens" or special architectural modifications (e.g., extra learnable tokens prepended to the sequence).
- No auxiliary reasoning-specific loss (e.g., trace reconstruction loss, reasoning consistency loss).
- No combined R2R+MID training script exists. The two ideas are implemented in separate training pipelines.

---

## 3. Experimental Setup and Hyperparameters

### 3.1 R2R Training Configuration (Primary Experiment Track)

| Parameter | Value |
|---|---|
| Base model | `vidore/colSmol-256M` (ColIdefics3) |
| LoRA rank / alpha | 32 / 32 |
| Learning rate | $5 \times 10^{-6}$ |
| Batch size | 2 |
| Gradient accumulation steps | 8 |
| Effective batch size | 16 |
| Epochs | 3 |
| Warmup steps | 100 |
| Temperature (InfoNCE) | 0.1 |
| Optimizer | AdamW (weight decay 0.01) |
| Max gradient norm | 1.0 |
| Precision | bf16 (auto-detected) |
| Seed | 42 |
| Training data | `vidore/colpali_train_set` (streaming) |
| Traces | `data/traces/train_traces_full.json` (~7.8k pairs) |
| Trace LLM | Qwen/Qwen2.5-0.5B-Instruct |

### 3.2 MID Distillation Configuration

| Parameter | Value |
|---|---|
| Teacher | `vidore/colpali-v1.3` (3B, CPU) |
| Student | `vidore/colSmol-256M` (GPU) |
| LoRA rank / alpha | 32 / 32 |
| Learning rate | $2 \times 10^{-5}$ |
| Batch size | 2 |
| Gradient accumulation steps | 8 |
| Effective batch size | 16 |
| Epochs | 1 |
| Warmup steps | 100 |
| KD temperature | 2.0 |
| Loss weights ($\alpha$, $\beta$, $\gamma$) | 1.0, 1.0, 0.5 |
| Training data | `vidore/colpali_train_set` (streaming) |

### 3.3 Datasets

**Training:** `vidore/colpali_train_set` — the same 118,695 query-page pair dataset released with the original ColPali paper, loaded in streaming mode.

**Evaluation benchmarks:**
- **ViDoRe v1:** 10 subtasks — ArxivQA, DocVQA, InfoVQA, TabFQuAD, TATDQA, ShiftProject, SyntheticDocQA (AI, Energy, Gov., Healthcare)
- **ViDoRe v2 (English):** BioMedicalLectures
- **ViDoRe v2 (All):** BioMedicalLectures, ESG Reports, ESG Reports HL, EconomicsReports
- **ViDoRe v3:** Referenced in experiment plans and evaluation scripts but no result JSONs are present in the repository (evaluations were in-progress at time of audit)

Evaluation is performed via the MTEB library, using a custom `R2RColSmolWrapper` (which extends `AbsEncoder`) that handles trace augmentation, LoRA adapter loading, and chunked CPU-based MaxSim similarity computation.

---

## 4. Results and Metrics

### 4.1 Baseline ColSmol-256M Performance (ViDoRe v1)

Complete baseline results are available from the MTEB JSON outputs. All scores are nDCG@5 (×100):

| Task | ColSmol-256M | ColPali (3B) | ColQwen2 (2B) |
|---|---:|---:|---:|
| ArxivQA | 72.0 | 79.1 | 86.4 |
| DocVQA | 55.7 | 54.4 | 56.2 |
| InfoVQA | 82.5 | 81.8 | 89.8 |
| TabFQuAD | 62.1 | 83.9 | 88.7 |
| TATDQA | 74.5 | 65.8 | 75.2 |
| ShiftProject | 56.2 | 73.2 | 85.7 |
| SynDocQA-AI | 94.9 | 96.2 | 98.8 |
| SynDocQA-Energy | 91.8 | 91.0 | 94.8 |
| SynDocQA-Gov. | 92.6 | 92.7 | 93.6 |
| SynDocQA-Health. | 95.1 | 94.4 | 97.3 |
| **Average** | **77.7** | **81.3** | **86.6** |

The ColSmol-to-ColPali gap is 3.6 points on average, but widens significantly on visually complex tasks: TabFQuAD (−21.8), ShiftProject (−17.0), and ArxivQA (−7.1).

### 4.2 Baseline ColSmol-256M Performance (ViDoRe v2 English)

| Task | nDCG@5 |
|---|---:|
| ESG Reports (Manual/HL) | 0.4175 |
| ESG Reports | 0.4561 |
| Economics | 0.5320 |
| BioMedical | 0.5035 |
| **Average** | **0.4773** |

The v2 scores are substantially lower than v1, reflecting harder multilingual and domain-specific tasks.

### 4.3 R2R Ablation Training Status

Per the experiment log and task tracker:
- **`r2r_use_seed42`:** Training completed. Final checkpoint saved.
- **`r2r_none_seed42`:** Training completed. Final checkpoint saved.
- **`r2r_shuffle_seed42`:** Training completed. Final checkpoint saved.

Training metrics CSV files (`train_metrics.csv`) and run metadata JSON files were written for each run, but these artifacts are not present in the current repository snapshot (likely in a `checkpoints/` directory that is git-ignored).

### 4.4 R2R Evaluation Status

The full evaluation matrix (3 trace modes × 3 benchmarks = 9 evaluations) was launched via `scripts/run_r2r_evals.sh` and logged as in-progress. Expected output directories:
- `results/r2r_{use,none,shuffle}_seed42_vidorev{1,2,3}/`

No final ablation comparison CSVs or summary tables are present in the repository at audit time. The comparison and aggregation scripts (`compare_ablation_results.py`, `aggregate_ablation_summary.py`) are fully implemented and ready to run once the evaluation outputs land.

### 4.5 Loss Curve and Training Trajectory

No training logs, wandb data, or metrics CSV files are committed to the repository. The training framework writes per-step loss values to `train_metrics.csv` (columns: `global_step`, `epoch`, `loss`), but these files reside outside the committed tree. No loss curve visualisation is available for audit.

### 4.6 Latency and Memory

No explicit latency or memory benchmarks are implemented in the repository. The evaluation scripts use chunked CPU-based MaxSim (chunk size 32) to avoid GPU OOM during the similarity matrix computation, which indicates that the full similarity tensor does not fit in 8GB VRAM for typical benchmark corpus sizes. The project targets 8GB VRAM for training (LoRA + small batch + gradient accumulation). The R2R inference pipeline fits within approximately 6GB: 0.5B reasoning LLM (~1GB) + 256M retriever (~1GB) + LoRA + working memory.

---

## 5. Critical Analysis

### 5.1 Primary Bottlenecks

**Batch size constraint:** The in-batch negative contrastive loss requires $B \geq 2$ and becomes more effective with larger $B$. The physical batch size is limited to 2 by VRAM constraints, with gradient accumulation to reach an effective batch of 16. However, gradient accumulation does not increase the number of in-batch negatives seen per forward pass — each step still only contrasts against 1 negative. This is a fundamental quality bottleneck for the contrastive signal.

**Teacher-student dimension mismatch for MID:** The teacher (ColPali/ColQwen) and student (ColSmol) produce different numbers of tokens ($T_q$, $T_d$) due to different vision encoders and resolutions. The interaction loss assumes the teacher and student have the same sequence lengths for the attention map KL-divergence. If they differ, the loss computation will fail or produce misaligned gradients. This is not handled in the current code and represents a latent bug in the MID pipeline.

**Memory pressure during MaxSim:** The iterative score matrix construction (looping over batch items) trades compute efficiency for memory safety but becomes slow for larger batches.

### 5.2 Implemented but Unused or Incomplete Features

1. **MID distillation (full pipeline):** The teacher wrapper, all three loss components, the distillation trainer, and the training script are fully implemented. However, based on the experiment log, no MID training runs have been executed. All experimental effort has been directed at R2R.

2. **Explicit negative mining in contrastive loss:** The `contrastive_loss` function in `losses.py` accepts a dedicated `neg_doc_embeddings` tensor of shape $(B, N, T_d, D)$ for explicit hard negatives. This is never used — both the distillation trainer and R2R trainer construct in-batch score matrices instead.

3. **Combined R2R+MID training:** Identified in the project proposal as the highest-impact experiment ("Combined Track"), but no combined trainer or script exists.

4. **Warmup scheduler:** Both trainer configs specify `warmup_steps=100`, but neither trainer (`DistillationTrainer` or `R2RTrainer`) implements a learning rate scheduler. The warmup parameter is set but ignored; training runs from step 0 at the full learning rate.

5. **WandB logging:** Both trainer configs include `use_wandb` and `wandb_project` fields, but no WandB integration code exists in either trainer. All logging is print-to-console only, with optional CSV metric logging in the R2R trainer.

6. **Gradient checkpointing:** Explicitly disabled by default. The R2R config has `enable_gradient_checkpointing: false`. The model wrapper forcefully disables it after loading. It is available as an option but was deliberately turned off, likely to avoid training instability.

7. **ViDoRe v3 evaluation:** Referenced in experiment plans, evaluation scripts, and the ICDAR proposal, but no v3 dataset definitions exist in the data loader (`vidore.py` only defines v1, v2, and train). The evaluation shell script references `ViDoRe(v3)` as an MTEB benchmark name, suggesting it may be resolved at runtime by the MTEB library.

### 5.3 Code Quality Observations

- **Numerical stability fix is partial:** The R2R trainer normalises embeddings and casts scores to float32, but the distillation trainer does not apply the same safeguards. If MID training is attempted in fp16/bf16, the same NaN issues that were fixed in R2R may recur.

- **Gradient accumulation edge case:** Both trainers flush any remaining accumulated gradients at the end of each epoch, which is correct. However, the `global_step` counter increments before the flush check, meaning the final accumulated step within an epoch may clip gradients using parameters that have already been partially updated.

- **Score matrix temperature difference:** The distillation trainer uses `self.config.temperature` (default 2.0, KD temperature) for the in-batch contrastive loss, while the R2R trainer uses its own temperature (0.1). The MID contrastive loss should arguably use a separate, lower temperature independent of the KD temperature, since InfoNCE and knowledge distillation temperatures serve different purposes.

- **Teacher attention computation correctness:** The teacher's `get_maxsim_attention` divides by $\sqrt{D}$ (scaled dot-product), but the interaction loss then re-softmaxes the teacher attention through `log → divide by T → softmax`. This double-softmax (once in the teacher, once in the loss) may dilute the teacher signal. The intent appears to be temperature-softened KD, but the teacher output is already a probability distribution.

### 5.4 What Has Been Tried vs. What Is Theoretically Implemented

| Component | Implementation Status | Experimental Status |
|---|---|---|
| ColSmol baseline evaluation (v1) | Complete | Validated — results in repo |
| ColSmol baseline evaluation (v2) | Complete | Validated — results in repo |
| R2R trace generation | Complete | Executed — ~7.8k traces generated |
| R2R training (use/none/shuffle) | Complete | All 3 ablations trained to completion |
| R2R evaluation pipeline | Complete | In-progress at time of audit (9 runs launched) |
| R2R ablation comparison tooling | Complete | Awaiting evaluation results |
| MID loss functions | Complete | Unit-tested (loss shapes, batch constraints) |
| MID distillation trainer | Complete | Never executed on real data |
| MID training script | Complete | Never executed |
| Teacher model wrapper | Complete | Unit-tested (attention shape check) |
| Combined R2R+MID | Not implemented | Not attempted |
| Hard negative mining | Partially implemented (loss accepts negatives) | Never used |
| Learning rate scheduler | Not implemented | N/A |
| WandB logging | Not implemented | N/A |
| Latency/memory benchmarks | Not implemented | N/A |

---

## Summary

The ColSmol-Reasoning project is a well-structured research codebase targeting two complementary approaches to improve a compact 256M-parameter visual document retriever: reasoning-augmented querying (R2R) and interaction-level knowledge distillation (MID).

The R2R track is substantially further along: trace generation, training with ablation controls, and the full evaluation pipeline are operational. Three training runs (use, none, shuffle) have completed, and the evaluation matrix was running at audit time. The MID track is fully coded and unit-tested but has not been executed on real data.

The core architectural design is sound — the ColBERT-style late interaction with MaxSim scoring is correctly implemented, LoRA adaptation targets the attention projections, and the 8GB VRAM constraint is respected through aggressive memory management (small batch, CPU teacher, gradient accumulation, chunked similarity).

The most significant gaps are: (1) the absence of a learning rate warmup scheduler despite the config field, (2) the in-batch-only negative strategy limiting contrastive quality at small batch sizes, (3) the untested MID pipeline and latent dimension-mismatch risk in the interaction loss, and (4) the missing combined R2R+MID experiment that the ICDAR proposal identifies as the primary contribution.
