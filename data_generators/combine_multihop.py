"""Combine per-env multihop task files into tasks_v1_multihop_{train,test}.json.

Expects files like: calculator_multihop_easy_train.json, retriever_multihop_hard_test.json, etc.
Assigns deterministic IDs using env_number * 10000 + difficulty_code * 1000 + offset.
"""
import json
from pathlib import Path
from collections import Counter

BASE = (Path(__file__).resolve().parent / ".." / "data" / "task_sets").resolve()

ENVS = ["calculator", "retriever", "code_executor"]
DIFF_CODE = {"easy": 1, "medium": 2, "hard": 3}
SPLIT_OFFSET = {"train": 0, "test": 500}

train_all, test_all = [], []

for env_idx, env in enumerate(ENVS):
    env_num = 16 + env_idx  # 16, 17, 18 (after single-hop 1-15)
    for diff in ["easy", "medium", "hard"]:
        for split, bucket in [("train", train_all), ("test", test_all)]:
            f = BASE / f"{env}_multihop_{diff}_{split}.json"
            if not f.exists():
                print(f"  MISSING: {f.name}")
                continue
            tasks = json.loads(f.read_text())
            base_id = env_num * 10000 + DIFF_CODE[diff] * 1000 + SPLIT_OFFSET[split]
            for i, t in enumerate(tasks):
                t["id"] = base_id + (i + 1)
            f.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
            bucket.extend(tasks)

train_ids = [t["id"] for t in train_all]
test_ids = [t["id"] for t in test_all]
assert len(train_ids) == len(set(train_ids)), "Duplicate train IDs"
assert len(test_ids) == len(set(test_ids)), "Duplicate test IDs"
assert not (set(train_ids) & set(test_ids)), "Train/test ID overlap"

out = BASE.parent
(out / "tasks_v1_multihop_train.json").write_text(json.dumps(train_all, ensure_ascii=False, indent=2) + "\n")
(out / "tasks_v1_multihop_test.json").write_text(json.dumps(test_all, ensure_ascii=False, indent=2) + "\n")
print(f"tasks_v1_multihop_train.json: {len(train_all)} tasks")
print(f"tasks_v1_multihop_test.json: {len(test_all)} tasks")
