#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Full pipeline: eval → extract features → train probe → probe eval → plot
# Repeats for each model specified.
#
# Usage:
#   ./run_pipeline.sh                                  # all 3 models, all steps
#   ./run_pipeline.sh --steps 4,5 qwen2.5-7b           # only steps 4 & 5
#   ./run_pipeline.sh --steps 1-3 qwen2.5-7b qwen3-4b-instruct
#
# Options:
#   --steps S    Run only specified steps (e.g., "1-3", "4,5")
#
# Steps:
#   1  Run evaluation (all prompt modes × reasoning modes)
#   2  Extract features (hidden states + no-tool labels)
#   3  Train probe (all layers)
#   4  Probe-guided evaluation (soft prefill, sweep thresholds)
#   5  Plot tradeoff (baselines + probe, overall + per-difficulty)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse flags
STEPS=""
MODELS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      STEPS="$2"
      shift 2
      ;;
    *)
      MODELS+=("$1")
      shift
      ;;
  esac
done

DATA_TRAIN="./data/tasks_v1_train.json"
DATA_TEST="./data/tasks_v1_test.json"

# Defaults
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=("qwen3-1.7b" "qwen3-4b-instruct" "qwen3-14b" "qwen3-32b" "llama3.1-8b" "llama3.3-70b")
fi

# Parse step spec into a set (e.g. "4,5" or "1-3" or "1,3-5")
should_run() {
  local step=$1
  if [[ -z "$STEPS" ]]; then
    return 0  # no filter = run all
  fi
  IFS=',' read -ra parts <<< "$STEPS"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then
      local lo=${part%-*}
      local hi=${part#*-}
      if [[ $step -ge $lo && $step -le $hi ]]; then
        return 0
      fi
    elif [[ "$step" -eq "$part" ]]; then
      return 0
    fi
  done
  return 1
}

# Probe settings
REG=10000
THRESHOLDS="0.1 0.3 0.5 0.7 0.9"
TEMPERATURE=2.0
N_RUNS=3

for MODEL in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL")"
  OUT_DIR="./outputs/${MODEL_NAME}"
  PROBE_DIR="./probe_data/${MODEL_NAME}"

  echo "========================================================================"
  echo "  MODEL: ${MODEL_NAME}  STEPS: ${STEPS:-1-5}"
  echo "========================================================================"

  # ------------------------------------------------------------------
  # Step 1: Run evaluation (all prompt modes × reasoning modes)
  # ------------------------------------------------------------------
  if should_run 1; then
    echo "[Step 1/5] Running evaluation..."
    mkdir -p "$OUT_DIR"
    ./run_all_settings.sh "$MODEL" all all "" "$N_RUNS"
  fi

  # ------------------------------------------------------------------
  # Step 2: Extract features (no-reasoning mode)
  # ------------------------------------------------------------------
  if should_run 2; then
    echo "[Step 2/5] Extracting features..."
    mkdir -p "$PROBE_DIR"
    python extract_features.py \
      --model_path "$MODEL" \
      --output_dir "$PROBE_DIR" \
      --data_path "$DATA_TRAIN" \
      --data_path_test "$DATA_TEST"
  fi

  # ------------------------------------------------------------------
  # Step 3: Train probe (all layers)
  # ------------------------------------------------------------------
  if should_run 3; then
    echo "[Step 3/5] Training probe..."
    python train_probe.py \
      --data_dir "$PROBE_DIR" \
      --mode no_reasoning \
      --reg $REG --all_layers
  fi

  # ------------------------------------------------------------------
  # Step 4: Probe-guided evaluation (sweep thresholds)
  # ------------------------------------------------------------------
  if should_run 4; then
    echo "[Step 4/5] Running probe-guided evaluation (soft prefill)..."
    for t in $THRESHOLDS; do
      python run_probe_eval.py \
        --model_path "$MODEL" \
        --probe_dir "$PROBE_DIR" \
        --data_path "$DATA_TEST" \
        --threshold "$t" --temperature "$TEMPERATURE" --prefill_mode soft \
        --output_dir "$OUT_DIR" \
        --n_runs "$N_RUNS"
    done
    # Hard prefill for Llama models (soft prefill is partially ignored)
    if echo "$MODEL" | grep -qi "llama"; then
      echo "[Step 4/5] Running probe-guided evaluation (hard prefill)..."
      for t in $THRESHOLDS; do
        python run_probe_eval.py \
          --model_path "$MODEL" \
          --probe_dir "$PROBE_DIR" \
          --data_path "$DATA_TEST" \
          --threshold "$t" --temperature "$TEMPERATURE" --prefill_mode hard \
          --output_dir "$OUT_DIR" \
          --n_runs "$N_RUNS"
      done
    fi
  fi

  # ------------------------------------------------------------------
  # Step 5: Plot tradeoff (baselines + probe, overall + per-difficulty)
  # ------------------------------------------------------------------
  if should_run 5; then
    echo "[Step 5/5] Plotting tradeoff..."
    python plot_figures.py \
      --models "$MODEL_NAME" \
      --data_path "$DATA_TEST" \
      --output "./figures/${MODEL_NAME}_tradeoff.pdf"
  fi

  echo ""
  echo "  Done: ${MODEL_NAME}"
  echo "    Eval results:  ${OUT_DIR}/"
  echo "    Probe data:    ${PROBE_DIR}/"
  echo ""

done

echo "========================================================================"
echo "  All models complete."
echo "========================================================================"
