#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<EOF
Usage:
  scripts/run_ablation.sh

Environment overrides:
  CONFIG=<path>   (default: configs/r2r/base.yaml)
  TRACES=<path>   (default: data/traces/train_traces_full.json)
  DTYPE=<dtype>   (default: bf16)
  SEED=<int>      (default: 42)
  MODES=<modes>   (default: "none shuffle")

Example:
  MODES="none shuffle" DTYPE=bf16 SEED=42 scripts/run_ablation.sh
EOF
  exit 0
fi

CONFIG="${CONFIG:-configs/r2r/base.yaml}"
TRACES="${TRACES:-data/traces/train_traces_full.json}"
DTYPE="${DTYPE:-bf16}"
SEED="${SEED:-42}"
MODES="${MODES:-none shuffle}"

if [[ ! -x "./venv/bin/python" ]]; then
  echo "Missing virtualenv python at ./venv/bin/python"
  exit 1
fi

mkdir -p logs/live

echo "Starting ablation runner"
echo "  config: $CONFIG"
echo "  traces: $TRACES"
echo "  dtype:  $DTYPE"
echo "  seed:   $SEED"
echo "  modes:  $MODES"

for mode in $MODES; do
  run_name="r2r_${mode}_seed${SEED}"
  run_dir="checkpoints/r2r/runs/${run_name}"
  final_ckpt="${run_dir}/checkpoints/final/adapter_model.safetensors"
  log_file="logs/live/${run_name}.log"

  if [[ -f "$final_ckpt" ]]; then
    echo "[skip] ${run_name} already complete: ${final_ckpt}"
    continue
  fi

  echo "[run] ${run_name}"
  env PYTORCH_ALLOC_CONF=expandable_segments:True \
    ./venv/bin/python scripts/train_r2r.py \
      --config "$CONFIG" \
      --traces "$TRACES" \
      --trace-mode "$mode" \
      --dtype "$DTYPE" \
      --seed "$SEED" \
      --run-name "$run_name" \
      2>&1 | tee "$log_file"
done

echo "Ablation runner finished."
