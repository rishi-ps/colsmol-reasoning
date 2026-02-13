#!/usr/bin/env bash
set -euo pipefail

# Full R2R evaluation runner:
# - runs use/none/shuffle
# - runs ViDoRe(v1), ViDoRe(v2), ViDoRe(v3)
# - writes logs and outputs under results/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-./venv/bin/python}"
LOG_DIR="${LOG_DIR:-logs/live}"
mkdir -p "$LOG_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

declare -a MODES=("use" "none" "shuffle")
declare -a BENCHMARKS=("ViDoRe(v1)" "ViDoRe(v2)" "ViDoRe(v3)")
declare -A RUN_NAMES=(
  ["use"]="r2r_use_seed42"
  ["none"]="r2r_none_seed42"
  ["shuffle"]="r2r_shuffle_seed42"
)

for mode in "${MODES[@]}"; do
  run_name="${RUN_NAMES[$mode]}"
  adapter="checkpoints/r2r/runs/${run_name}/checkpoints/final"
  if [[ ! -d "$adapter" ]]; then
    echo "Missing adapter dir for ${mode}: ${adapter}"
    exit 1
  fi

  for benchmark in "${BENCHMARKS[@]}"; do
    suffix="$(echo "$benchmark" | tr -d '()' | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
    # Example: ViDoRe(v1) -> vidorev1
    output_dir="results/${run_name}_${suffix}"
    trace_cache="data/traces/eval_${suffix}.json"
    log_file="${LOG_DIR}/eval_${run_name}_${suffix}.log"

    if find "$output_dir" -name "*.json" -type f 2>/dev/null | grep -q .; then
      echo "Skipping ${mode} ${benchmark}; JSON results already exist in ${output_dir}"
      continue
    fi

    echo ""
    echo "============================================================"
    echo "Running mode=${mode}, benchmark=${benchmark}"
    echo "adapter=${adapter}"
    echo "output=${output_dir}"
    echo "log=${log_file}"
    echo "============================================================"

    "$PY" evaluation/evaluate_r2r.py \
      --benchmark "$benchmark" \
      --output-folder "$output_dir" \
      --adapter "$adapter" \
      --trace-mode "$mode" \
      --trace-cache "$trace_cache" \
      2>&1 | tee "$log_file"
  done
done

echo ""
echo "All evaluations completed."
