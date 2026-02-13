# Project Task List (R2R / Idea 2)

Last updated: `2026-02-13`  
Resume instruction: `Open this file and continue from the first [IN PROGRESS] or [NEXT] item.`

## Goal

- Produce publishable evidence for `Idea 2 (Reason-to-Retrieve)` in `docs/publishable_research_directions.md`:
  - stable training
  - `use/none/shuffle` ablations
  - clear comparison and interpretation.

## Milestones Achieved

- [DONE] Core R2R pipeline implemented (trace generation + training flow).
- [DONE] Reproducibility plumbing:
  - `src/utils/experiment.py`
  - run metadata + metrics + per-run folders.
- [DONE] Training controls added:
  - `--trace-mode use|none|shuffle`
  - `--dtype auto|fp16|bf16|fp32`.
- [DONE] Numerical stability fix in `src/reasoning/trainer.py`:
  - normalize embeddings in float32 with epsilon guard (fixes fp16/bf16 NaN risk).
- [DONE] Script correctness fix in `scripts/train_r2r.py`:
  - removed trailing code that caused post-run `NameError: args`.
- [DONE] Ablation comparison script added:
  - `evaluation/compare_ablation_results.py`.
- [DONE] Smoke trace generation completed:
  - `data/traces/train_traces_smoke_200.json` (+ metadata).
- [DONE] Main trace file currently available:
  - `data/traces/train_traces_full.json` (~7.8k traces).

## Experiment Status (Live)

- [DONE] `r2r_use_seed42` finished successfully.
  - artifacts: `checkpoints/r2r/runs/r2r_use_seed42/checkpoints/final/adapter_model.safetensors`
  - metrics: `checkpoints/r2r/runs/r2r_use_seed42/train_metrics.csv`.
- [DONE] `r2r_none_seed42` finished successfully.
  - artifacts: `checkpoints/r2r/runs/r2r_none_seed42/checkpoints/final/adapter_model.safetensors`
  - metrics: `checkpoints/r2r/runs/r2r_none_seed42/train_metrics.csv`.
- [DONE] `r2r_shuffle_seed42` finished successfully.
  - artifacts: `checkpoints/r2r/runs/r2r_shuffle_seed42/checkpoints/final/adapter_model.safetensors`
  - metrics: `checkpoints/r2r/runs/r2r_shuffle_seed42/train_metrics.csv`.
- [IN PROGRESS] Full evaluation matrix is running via `scripts/run_r2r_evals.sh` in tmux session `r2r_eval`.
  - runner log: `logs/live/eval_all_runner.log`
  - current per-run log example: `logs/live/eval_r2r_use_seed42_vidorev1.log`

## Task List

- [DONE] Lock experiment settings:
  - config `configs/r2r/base.yaml`
  - seed `42`
  - trace modes `use, none, shuffle`.
- [DONE] Validate training health with bf16 smoke runs.
- [DONE] Complete ablation training matrix on `train_traces_full.json`.
  - `use`: done
  - `none`: done
  - `shuffle`: done
- [DONE] Added one-command runner:
  - `scripts/run_ablation.sh` (runs `none` then `shuffle`, skips already-complete modes, writes logs to `logs/live/`).
- [DONE] Started evaluation for completed checkpoints.
  - command used: `scripts/run_r2r_evals.sh`
  - outputs currently written per benchmark:
    - `results/r2r_use_seed42_vidorev1`, `results/r2r_use_seed42_vidorev2`, `results/r2r_use_seed42_vidorev3`
    - `results/r2r_none_seed42_vidorev1`, `results/r2r_none_seed42_vidorev2`, `results/r2r_none_seed42_vidorev3`
    - `results/r2r_shuffle_seed42_vidorev1`, `results/r2r_shuffle_seed42_vidorev2`, `results/r2r_shuffle_seed42_vidorev3`
- [NEXT] Wait for `scripts/run_r2r_evals.sh` to finish all 9 runs (3 modes x 3 benchmarks).
- [NEXT] Run ablation comparison per benchmark:
```bash
./venv/bin/python evaluation/compare_ablation_results.py \
  --run use=results/r2r_use_seed42_vidorev1 \
  --run none=results/r2r_none_seed42_vidorev1 \
  --run shuffle=results/r2r_shuffle_seed42_vidorev1 \
  --output-csv results/r2r_ablation_comparison_vidorev1.csv
```
```bash
./venv/bin/python evaluation/compare_ablation_results.py \
  --run use=results/r2r_use_seed42_vidorev2 \
  --run none=results/r2r_none_seed42_vidorev2 \
  --run shuffle=results/r2r_shuffle_seed42_vidorev2 \
  --output-csv results/r2r_ablation_comparison_vidorev2.csv
```
```bash
./venv/bin/python evaluation/compare_ablation_results.py \
  --run use=results/r2r_use_seed42_vidorev3 \
  --run none=results/r2r_none_seed42_vidorev3 \
  --run shuffle=results/r2r_shuffle_seed42_vidorev3 \
  --output-csv results/r2r_ablation_comparison_vidorev3.csv
```
- [NEXT] Build a single summary table from the three benchmark CSVs:
```bash
./venv/bin/python evaluation/aggregate_ablation_summary.py \
  --csv vidorev1=results/r2r_ablation_comparison_vidorev1.csv \
  --csv vidorev2=results/r2r_ablation_comparison_vidorev2.csv \
  --csv vidorev3=results/r2r_ablation_comparison_vidorev3.csv \
  --output-csv results/r2r_ablation_summary.csv
```
- [NEXT] Write interpretation in `docs/experiment_log.md`:
  - where reasoning helps (`use > none`)
  - whether trace quality matters (`use > shuffle`)
  - task-level wins/failures tied to publishable claim.

## Notes For Next Session

- If a run seems stalled, check:
  - `ps -ef | rg train_r2r.py`
  - `tail -c 6000 logs/live/<run_name>.log | tr '\r' '\n' | tail -n 30`.
- Use sequential foreground runs with `tee` for reliable live logs; previous `nohup` parallel attempts exited early.
- To survive disconnects, run in tmux:
```bash
tmux new -s r2r_ablation
cd /home/bs_thesis/colsmol-reasoning
scripts/run_ablation.sh
# detach: Ctrl+b then d
```
