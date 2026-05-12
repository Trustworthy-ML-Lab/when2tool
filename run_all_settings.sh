#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_path> [prompt_mode] [reasoning_mode] [envs] [n_runs] [data_path] [output_dir]"
  echo "Example: $0 qwen2.5-7b sparse_tool reasoning"
  echo "Example: $0 qwen2.5-7b all all CodeExecutorEnv,RegexMatchEnv 3"
  echo "prompt_mode: current|necessary_tool|sparse_tool|force_tool|no_tool|all (default: all)"
  echo "reasoning_mode: reasoning|no_reasoning|all (default: all)"
  echo "envs: comma-separated env filter (default: all envs)"
  echo "n_runs: number of evaluation runs per setting (default: 3)"
  echo "data_path: path to test data (default: ./data/tasks_v1_test.json)"
  echo "output_dir: override output directory (default: ./outputs/<model_name>)"
  exit 1
fi

MODEL_PATH="$1"
PROMPT_MODE="${2:-all}"
REASONING_MODE="${3:-all}"
ENVS="${4:-}"
N_RUNS="${5:-3}"
DATA_PATH="${6:-./data/tasks_v1_test.json}"
OUTPUT_DIR_OVERRIDE="${7:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="$(basename "$MODEL_PATH")"
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
  OUT_DIR="$OUTPUT_DIR_OVERRIDE"
else
  OUT_DIR="./outputs/${MODEL_NAME}"
fi
mkdir -p "$OUT_DIR"

ENVS_ARG=""
if [[ -n "$ENVS" ]]; then
  ENVS_ARG="--envs $ENVS"
fi

python run_eval.py \
  --model_path "$MODEL_PATH" \
  --backend vllm \
  --data_path "$DATA_PATH" \
  --record_mode lite \
  --prompt_mode "$PROMPT_MODE" \
  --reasoning_mode "$REASONING_MODE" \
  --output_path "${OUT_DIR}/" \
  --tensor_parallel_size 4 \
  --n_runs "$N_RUNS" \
  $ENVS_ARG

echo "Done."
echo "Output directory: ${OUT_DIR}"
echo "prompt_mode=${PROMPT_MODE}, reasoning_mode=${REASONING_MODE}"
echo "Per-setting outputs were written by run_eval."
echo "To build comparison tables, run:"
echo "python summarize_settings.py --glob \"${OUT_DIR}/*_settings_table.json\" --output_prefix \"${OUT_DIR}/summary\""
