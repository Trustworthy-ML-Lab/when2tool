#!/bin/bash
# Generate all benchmark data from scratch.
# Running this script produces per-env JSONs in data/task_sets/ and
# combined tasks_v1_{train,test}.json in data/.
#
# Usage: bash generate_data.sh  (from repo root)

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
GEN_DIR="$REPO_ROOT/data_generators"
TASK_SETS_DIR="$REPO_ROOT/data/task_sets"

mkdir -p "$TASK_SETS_DIR"

echo "=== Step 1: Generate per-environment single-hop tasks ==="
cd "$GEN_DIR"
python3 gen_calculator_20.py
python3 gen_statistics_20.py
python3 gen_counting_20.py
python3 gen_matrix_20.py
python3 gen_prime_20.py
python3 gen_retriever_20.py
python3 gen_historical_year_20.py
python3 gen_game_rule_20.py
python3 gen_hash_20.py
python3 gen_decoding_20.py
python3 gen_list_20.py
python3 gen_datetime_20.py
python3 gen_code_executor_20.py
python3 gen_schedule_20.py
python3 gen_regex_match_20.py

echo ""
echo "=== Step 2: Generate per-environment multi-hop tasks ==="
python3 gen_calculator_multihop.py
python3 gen_code_executor_multihop.py
python3 gen_retriever_multihop.py

echo ""
echo "=== Step 3: Assign IDs and combine single-hop tasks ==="
python3 fix_ids.py

echo ""
echo "=== Step 4: Assign IDs and combine multi-hop tasks ==="
python3 combine_multihop.py

echo ""
echo "=== Done ==="
echo "Generated files:"
ls -lh "$REPO_ROOT/data/tasks_v1_"*.json
