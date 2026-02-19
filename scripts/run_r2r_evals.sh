#!/usr/bin/env bash
set -uo pipefail

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
MAX_RETRIES="${MAX_RETRIES:-2}"

declare -a MODES=("use" "none" "shuffle")
declare -a BENCHMARKS=("ViDoRe(v1)" "ViDoRe(v2)" "ViDoRe(v3)")
declare -A RUN_NAMES=(
  ["use"]="r2r_use_seed42"
  ["none"]="r2r_none_seed42"
  ["shuffle"]="r2r_shuffle_seed42"
)

failed_runs=()

run_eval() {
  local mode="$1"
  local benchmark="$2"
  local adapter="$3"
  local output_dir="$4"
  local trace_cache="$5"
  local log_file="$6"

  local attempt=1
  local force_online=0

  : > "$log_file"
  while (( attempt <= MAX_RETRIES )); do
    echo "[attempt ${attempt}/${MAX_RETRIES}] mode=${mode} benchmark=${benchmark}" | tee -a "$log_file"
    if (( force_online == 1 )); then
      echo "Detected offline dataset miss previously; retrying with online HF flags." | tee -a "$log_file"
      env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
        "$PY" evaluation/evaluate_r2r.py \
          --benchmark "$benchmark" \
          --output-folder "$output_dir" \
          --adapter "$adapter" \
          --trace-mode "$mode" \
          --trace-cache "$trace_cache" \
          2>&1 | tee -a "$log_file"
    else
      "$PY" evaluation/evaluate_r2r.py \
        --benchmark "$benchmark" \
        --output-folder "$output_dir" \
        --adapter "$adapter" \
        --trace-mode "$mode" \
        --trace-cache "$trace_cache" \
        2>&1 | tee -a "$log_file"
    fi

    local status="${PIPESTATUS[0]}"
    if [[ "$status" -eq 0 ]]; then
      return 0
    fi

    if grep -q "OfflineModeIsEnabled" "$log_file" && (( force_online == 0 )); then
      force_online=1
    fi

    attempt=$((attempt + 1))
  done

  return 1
}

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

    echo ""
    echo "============================================================"
    echo "Running mode=${mode}, benchmark=${benchmark}"
    echo "adapter=${adapter}"
    echo "output=${output_dir}"
    echo "log=${log_file}"
    echo "============================================================"

    if ! run_eval "$mode" "$benchmark" "$adapter" "$output_dir" "$trace_cache" "$log_file"; then
      echo "FAILED mode=${mode}, benchmark=${benchmark}. Continuing with next run."
      failed_runs+=("${mode}|${benchmark}|${log_file}")
    fi
  done
done

echo ""
if [[ "${#failed_runs[@]}" -eq 0 ]]; then
  echo "All evaluations completed."
else
  echo "Evaluations completed with failures:"
  for item in "${failed_runs[@]}"; do
    IFS='|' read -r mode benchmark log_file <<< "$item"
    echo "  - mode=${mode}, benchmark=${benchmark}, log=${log_file}"
  done
  exit 1
fi
