#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Multi-hop pipeline — identical to run_pipeline.sh, different data.
#
# Steps:
#   1  Baseline eval (all prompt × reasoning modes)
#   2  Extract features (hidden states + labels)
#   3  Train probe
#   4  Transfer: single-step probe → multi-hop prediction
#   5  Probe-guided eval (sweep thresholds)
#   6  Plot tradeoff
#
# Usage:
#   ./run_multihop_pipeline.sh
#   ./run_multihop_pipeline.sh --steps 2,3 qwen2.5-7b
#
# Options:
#   --steps S    Run only specified steps
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STEPS=""
MODELS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) STEPS="$2"; shift 2 ;;
    *) MODELS+=("$1"); shift ;;
  esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=("qwen3-1.7b" "qwen3-4b-instruct" "qwen3-14b" "qwen3-32b" "llama3.1-8b" "llama3.3-70b")
fi

should_run() {
  local step=$1
  if [[ -z "$STEPS" ]]; then return 0; fi
  IFS=',' read -ra parts <<< "$STEPS"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then
      local lo=${part%-*} hi=${part#*-}
      [[ $step -ge $lo && $step -le $hi ]] && return 0
    elif [[ "$step" -eq "$part" ]]; then
      return 0
    fi
  done
  return 1
}

REG=10000
THRESHOLDS="0.1 0.3 0.5 0.7 0.9"
TEMPERATURE=2.0
N_RUNS=3

DATA_TRAIN="./data/tasks_v1_multihop_train.json"
DATA_TEST="./data/tasks_v1_multihop_test.json"

for MODEL in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL")"
  OUT_DIR="./outputs/${MODEL_NAME}_multihop"
  PROBE_DIR="./probe_data/${MODEL_NAME}_multihop"
  SINGLE_PROBE_DIR="./probe_data/${MODEL_NAME}"

  echo "========================================================================"
  echo "  MULTI-HOP: ${MODEL_NAME}  STEPS: ${STEPS:-1-6}"
  echo "========================================================================"

  # Step 1: Baseline eval (using multihop data)
  if should_run 1; then
    echo "[Step 1/6] Baseline evaluation..."
    mkdir -p "$OUT_DIR"
    ./run_all_settings.sh "$MODEL" all all "" "$N_RUNS" "$DATA_TEST" "$OUT_DIR"
  fi

  # Step 2: Extract features
  if should_run 2; then
    echo "[Step 2/6] Extracting features..."
    mkdir -p "$PROBE_DIR"
    python src/extract_features.py \
      --model_path "$MODEL" \
      --output_dir "$PROBE_DIR" \
      --data_path "$DATA_TRAIN" \
      --data_path_test "$DATA_TEST"
  fi

  # Step 3: Train probe
  if should_run 3; then
    echo "[Step 3/6] Training probe..."
    python src/train_probe.py \
      --data_dir "$PROBE_DIR" \
      --mode no_reasoning \
      --reg $REG --all_layers
  fi

  # Step 4: Transfer — apply single-hop probe to multi-hop data
  if should_run 4; then
    echo "[Step 4/6] Transfer: single-hop probe → multi-hop..."
    SINGLE_PROBE="${SINGLE_PROBE_DIR}/probe_no_reasoning.pt"
    if [[ ! -f "$SINGLE_PROBE" ]]; then
      echo "  ERROR: Run single-hop pipeline first: $SINGLE_PROBE not found"
      exit 1
    fi
    python src/eval_probe_transfer.py \
      --source_probe "$SINGLE_PROBE" \
      --target_hidden_dir "$PROBE_DIR" \
      --mode no_reasoning \
      --output_dir "$OUT_DIR"
  fi

  # Step 5: Probe-guided eval (sweep thresholds)
  if should_run 5; then
    echo "[Step 5/6] Probe-guided eval..."
    for t in $THRESHOLDS; do
      python src/run_probe_eval.py \
        --model_path "$MODEL" \
        --probe_dir "$PROBE_DIR" \
        --data_path "$DATA_TEST" \
        --threshold "$t" --temperature "$TEMPERATURE" --prefill_mode soft \
        --output_dir "$OUT_DIR" \
        --n_runs "$N_RUNS"
    done
  fi

  # Step 6: Plot
  if should_run 6; then
    echo "[Step 6/6] Plotting..."
    python src/plot_figures.py \
      --models "${MODEL_NAME}_multihop" \
      --data_path "$DATA_TEST" \
      --output "./figures/${MODEL_NAME}_multihop_tradeoff.pdf"
  fi

  echo "  Done: ${MODEL_NAME} (multi-hop)"
  echo ""
done

echo "Multi-hop pipeline complete."
