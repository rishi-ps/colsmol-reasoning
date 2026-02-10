# Project Status & Update Sheet

Use this document to quickly restore context and understand the current state of the project.

## 📅 Session: 2026-02-11 (Idea 2 Implementation)

**Summarized Status:**
We successfully implemented the core components for **Idea 2: Reason-to-Retrieve (R2R)**. The pipeline is validated end-to-end with a small subset of data.

### ✅ What We Have Done

1.  **Project Structure**:
    - Organized codebase into `src/`, `configs/`, `scripts/`, `docs/`.
    - `src.models`: Implemented `ColSmolWrapper` (using `ColIdefics3` architecture) and `TeacherModelWrapper`.
    - `src.data`: Standardized ViDoRe dataset loading with `load_vidore_dataset`.
    - `src.reasoning`: Implemented `TraceGenerator` (Qwen2.5-0.5B) and `ReasoningAugmentedRetriever`.

2.  **R2R Implementation**:
    - **Trace Generation (`scripts/generate_traces.py`)**:
      - Implemented streaming pipeline to process `vidore/colpali_train_set` without full download.
      - Added deterministic shuffling (`seed=42`) to ensure consistent data order.
      - **Validated**: Generated 100 sample traces in `data/traces/train_traces.json`.
    - **R2R Training (`scripts/train_r2r.py`)**:
      - Implemented `R2RTrainer` with in-batch contrastive loss.
      - Created `R2RDataset` that synchronizes streaming data with generated traces.
      - **Validated**: Validated training loop for 3 epochs on the 100-sample subset. It successfully aligns traces and computes loss.

3.  **Distillation (Idea 1) Foundation**:
    - Implemented MaxSim Interaction Distillation (MID) losses in `src/distillation/losses.py`.
    - Created trainer skeleton in `src/distillation/trainer.py`.

### ⏭️ What To Do Next

1.  **Run Full Trace Generation**:
    - Run `scripts/generate_traces.py` on the full training set (or a large subset, e.g., 20k-50k samples).
    - _Command_: `python scripts/generate_traces.py --config configs/r2r/base.yaml --version train --output data/traces/train_traces_full.json` (remove `--limit`).

2.  **Run Full R2R Training**:
    - Launch full fine-tuning of ColSmol using the generated traces.
    - _Command_: `python scripts/train_r2r.py --config configs/r2r/base.yaml --traces data/traces/train_traces_full.json`

3.  **Evaluate R2R**:
    - Run evaluation on ViDoRe v1 benchmark to measure performance improvement.

4.  **Implement Idea 1 (Distillation)**:
    - Finalize `scripts/train_distillation.py` loop (similar to R2R but with teacher scores).
    - Run distillation experiments.
