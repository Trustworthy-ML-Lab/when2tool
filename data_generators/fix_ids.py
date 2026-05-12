"""Fix IDs: assign clean, deterministic IDs to all task files.

ID scheme:
  Train: ID = <env_number> * 10000 + <difficulty_code> * 1000 + <task_index>
  Test:  ID = <env_number> * 10000 + <difficulty_code> * 1000 + 500 + <task_index>

  env_number:    1-15 (order matches ENV_PLAN.md)
  difficulty:    1=easy, 2=medium, 3=hard
  task_index:    1-20 (train) or 1-50 (test)

Examples:
  11001 = env 1 (calculator), easy, train task 01
  11020 = env 1 (calculator), easy, train task 20
  11501 = env 1 (calculator), easy, test task 01
  11550 = env 1 (calculator), easy, test task 50
  12001 = env 1 (calculator), medium, train task 01
  153050 = env 15 (regex_match), hard, test task 50
"""
import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent / ".." / "data" / "task_sets"
BASE = BASE.resolve()

# Env order (matches ENV_PLAN.md)
ENVS = [
    # A: Scale
    "calculator", "statistics", "counting", "matrix", "prime",
    # B: Missing Knowledge
    "retriever", "historical_year", "game_rule", "hash", "decoding",
    # C: Reliable Execution
    "list", "datetime", "code_executor", "schedule", "regex_match",
]

DIFF_CODE = {"easy": 1, "medium": 2, "hard": 3}
SPLIT_OFFSET = {"train": 0, "test": 500}

print(f"{'Env':<20} {'#':>3}  {'Easy Train':>14} {'Easy Test':>14} {'Med Train':>14} {'Med Test':>14} {'Hard Train':>14} {'Hard Test':>14}")
print("-" * 110)

for env_idx, env in enumerate(ENVS):
    env_num = env_idx + 1  # 1 to 15
    ranges = {}
    for diff in ["easy", "medium", "hard"]:
        for split in ["train", "test"]:
            f = BASE / f"{env}_single_{diff}_{split}.json"
            if not f.exists():
                print(f"  MISSING: {f.name}")
                continue
            tasks = json.loads(f.read_text())
            base_id = env_num * 10000 + DIFF_CODE[diff] * 1000 + SPLIT_OFFSET[split]
            for i, t in enumerate(tasks):
                t["id"] = base_id + (i + 1)
            f.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")

            first_id = base_id + 1
            last_id = base_id + len(tasks)
            ranges[f"{diff}_{split}"] = f"{first_id}-{last_id}"

    print(f"{env:<20} {env_num:>3}  "
          f"{ranges.get('easy_train','?'):>14} {ranges.get('easy_test','?'):>14} "
          f"{ranges.get('medium_train','?'):>14} {ranges.get('medium_test','?'):>14} "
          f"{ranges.get('hard_train','?'):>14} {ranges.get('hard_test','?'):>14}")

# Rebuild combined files
print()
train_all, test_all = [], []
for env in ENVS:
    for diff in ["easy", "medium", "hard"]:
        for split, bucket in [("train", train_all), ("test", test_all)]:
            f = BASE / f"{env}_single_{diff}_{split}.json"
            bucket.extend(json.loads(f.read_text()))

train_ids = [t["id"] for t in train_all]
test_ids = [t["id"] for t in test_all]
dupes_train = [k for k, v in Counter(train_ids).items() if v > 1]
dupes_test = [k for k, v in Counter(test_ids).items() if v > 1]
assert not dupes_train, f"Duplicate train IDs: {dupes_train[:10]}"
assert not dupes_test, f"Duplicate test IDs: {dupes_test[:10]}"

# No overlap between train and test
overlap = set(train_ids) & set(test_ids)
assert not overlap, f"Train/test ID overlap: {list(overlap)[:10]}"

out = BASE.parent
(out / "tasks_v1_train.json").write_text(json.dumps(train_all, ensure_ascii=False, indent=2) + "\n")
(out / "tasks_v1_test.json").write_text(json.dumps(test_all, ensure_ascii=False, indent=2) + "\n")
print(f"tasks_v1_train.json: {len(train_all)} tasks, {len(set(train_ids))} unique IDs")
print(f"tasks_v1_test.json: {len(test_all)} tasks, {len(set(test_ids))} unique IDs")
