# Experiment Log

## 2026-02-10: Project Setup

- Structured project for Ideas 1 (Distillation) + 2 (Reason-to-Retrieve)
- Baseline ColSmol evaluation results available in `results/`
- ViDoRe v1, v2, v3 benchmarks accessible

### Baseline Performance (ColSmol 256M)

| Benchmark       | Average NDCG@5 |
| --------------- | -------------- |
| ViDoRe v1       | ~81.2          |
| ViDoRe v2 (EN)  | TBD            |
| ViDoRe v2 (All) | TBD            |
| ViDoRe v3 (EN)  | TBD            |

---

## 2026-02-13: R2R Ablation Plan (Use/None/Shuffle)

### Objective

- Validate Idea 2 claim with controlled ablations:
  - `use > none` to show reasoning augmentation helps retrieval.
  - `use > shuffle` to show trace content quality matters (not only longer input).

### Fixed Experimental Setup

- Model: `ColSmol 256M` + LoRA adapters
- Seed: `42`
- Trace modes: `use`, `none`, `shuffle`
- Training traces: `data/traces/train_traces_full.json`
- Benchmarks: `ViDoRe(v1)`, `ViDoRe(v2)`, `ViDoRe(v3)`

### Training Completion Status

- `r2r_use_seed42`: complete
- `r2r_none_seed42`: complete
- `r2r_shuffle_seed42`: complete

### Evaluation Status

- Runner: `scripts/run_r2r_evals.sh`
- Logs:
  - `logs/live/eval_all_runner.log`
  - `logs/live/eval_r2r_use_seed42_vidorev1.log` (and corresponding per-run logs)

### Result Artifacts (Expected)

- Per benchmark comparison CSVs:
  - `results/r2r_ablation_comparison_vidorev1.csv`
  - `results/r2r_ablation_comparison_vidorev2.csv`
  - `results/r2r_ablation_comparison_vidorev3.csv`
- Cross-benchmark summary:
  - `results/r2r_ablation_summary.csv`

### Quantitative Summary Template

Fill after eval + comparison scripts complete.

| Benchmark | None Avg nDCG@5 | Use Avg nDCG@5 | Shuffle Avg nDCG@5 | Use-None | Use-Shuffle |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViDoRe(v1) | TBD | TBD | TBD | TBD | TBD |
| ViDoRe(v2) | TBD | TBD | TBD | TBD | TBD |
| ViDoRe(v3) | TBD | TBD | TBD | TBD | TBD |
| Overall | TBD | TBD | TBD | TBD | TBD |

### Interpretation Template

- `use > none`:
  - Summary: TBD
  - Strongest tasks: TBD
  - Weak/neutral tasks: TBD

- `use > shuffle`:
  - Summary: TBD
  - Evidence that trace quality matters: TBD
  - Any counterexamples: TBD

- Hard-vs-easy task behavior:
  - Hard tasks expected to benefit more: TBD
  - Easy tasks with smaller gains: TBD

### Risks / Caveats

- Single-seed result so far (`seed=42`) may overfit conclusions.
- Runtime is long; partial reads before full completion can be misleading.
- Need to separate benchmark effects from trace-mode effects in final claims.
