import json
import random
import re
from copy import deepcopy
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "data" / "task_sets"
OUT.mkdir(parents=True, exist_ok=True)
N_TOTAL = 70
N_SPLIT = 20
SEED = 20260304

LIST_TOOLS = ["append", "remove", "insert", "sort", "reverse"]
LIST_TOOLS_EASY = ["append", "remove", "insert", "reverse"]

RANGES = {
    "easy": (1, 40),
    "medium": (40, 260),
    "hard": (300, 5000),
}


def list_env(difficulty: str):
    tools = LIST_TOOLS_EASY if difficulty == "easy" else LIST_TOOLS
    return [{"name": "ListManipulationEnv", "tools": tools, "parameters": {}}]


def _weighted_choice(rng: random.Random, weighted_ops):
    ops = [x[0] for x in weighted_ops]
    weights = [x[1] for x in weighted_ops]
    return rng.choices(ops, weights=weights, k=1)[0]


def _sample_unique(rng: random.Random, lo: int, hi: int, k: int, forbid=None):
    forbid = set(forbid or [])
    pool = [x for x in range(lo, hi + 1) if x not in forbid]
    if len(pool) < k:
        raise ValueError(f"not enough unique values in range [{lo}, {hi}] for k={k}")
    return rng.sample(pool, k)


def make_1d(rng: random.Random, difficulty: str):
    lo, hi = RANGES[difficulty]
    if difficulty == "easy":
        n = rng.randint(3, 4)
    elif difficulty == "medium":
        n = rng.randint(6, 8)
    else:
        n = rng.randint(10, 13)
    return _sample_unique(rng, lo, hi, n)


def make_2d_hard(rng: random.Random):
    lo, hi = RANGES["hard"]
    rows = rng.randint(10, 13)
    cols = rng.randint(10, 13)
    vals = _sample_unique(rng, lo, hi, rows * cols)
    arr = []
    k = 0
    for _ in range(rows):
        arr.append(vals[k : k + cols])
        k += cols
    return arr


def apply_1d(arr, op, value=None, index=None):
    arr = deepcopy(arr)
    if op == "append":
        arr.append(value)
    elif op == "insert":
        arr.insert(index, value)
    elif op == "remove":
        arr.remove(value)
    elif op == "sort":
        arr.sort()
    elif op == "reverse":
        arr.reverse()
    else:
        raise ValueError(op)
    return arr


def apply_2d(arr, op, axis, value=None, index=None):
    arr = deepcopy(arr)
    if op == "sort":
        if axis == 0:
            if len(arr) > 0 and len(arr[0]) > 0:
                n_rows = len(arr)
                n_cols = len(arr[0])
                for c in range(n_cols):
                    col_sorted = sorted(arr[r][c] for r in range(n_rows))
                    for r in range(n_rows):
                        arr[r][c] = col_sorted[r]
        else:
            for row in arr:
                row.sort()
    elif op == "reverse":
        if axis == 0:
            arr.reverse()
        else:
            for row in arr:
                row.reverse()
    elif op == "remove":
        if axis == 0:
            arr.pop(index)
        else:
            for row in arr:
                row.pop(index)
    elif op == "insert":
        if axis == 0:
            arr.insert(index, deepcopy(value))
        else:
            for row in arr:
                row.insert(index, value)
    elif op == "append":
        if axis == 0:
            arr.append(deepcopy(value))
        else:
            for row in arr:
                row.append(value)
    else:
        raise ValueError(op)
    return arr


def op_text_1d(op, value=None, index=None):
    if op == "append":
        return f"append({value})"
    if op == "insert":
        return f"insert(index={index},value={value})"
    if op == "remove":
        return f"remove({value})"
    if op == "sort":
        return "sort()"
    if op == "reverse":
        return "reverse()"
    raise ValueError(op)


def op_text_2d(op, axis, value=None, index=None):
    if op == "append":
        if axis == 0:
            return f"append(axis=0,value={value})"
        return f"append(axis=1,value={value})"
    if op == "insert":
        if axis == 0:
            return f"insert(axis=0,index={index},value={value})"
        return f"insert(axis=1,index={index},value={value})"
    if op == "remove":
        return f"remove(axis={axis},index={index})"
    if op == "sort":
        return f"sort(axis={axis})"
    if op == "reverse":
        return f"reverse(axis={axis})"
    raise ValueError(op)


def task_single(task_id, difficulty, instruction, answer):
    return {
        "id": task_id,
        "difficulty": difficulty,
        "multi_step": False,
        "instruction": instruction,
        "environments": list_env(difficulty),
        "expected": {"answer": str(answer)},
        "tags": ["list", "single_round"],
    }


def gen_single_easy(start_id):
    rng = random.Random(SEED + 11)
    weighted = [("remove", 4), ("insert", 4), ("append", 3), ("reverse", 2)]
    out = []
    seen = set()
    while len(out) < N_TOTAL:
        init = make_1d(rng, "easy")
        op = _weighted_choice(rng, weighted)
        lo, hi = RANGES["easy"]
        if op == "remove":
            value = rng.choice(init)
            final = apply_1d(init, op, value=value)
            op_txt = op_text_1d(op, value=value)
        elif op == "insert":
            value = _sample_unique(rng, lo, hi, 1, forbid=init)[0]
            idx = rng.randint(0, len(init))
            final = apply_1d(init, op, value=value, index=idx)
            op_txt = op_text_1d(op, value=value, index=idx)
        elif op == "append":
            value = _sample_unique(rng, lo, hi, 1, forbid=init)[0]
            final = apply_1d(init, op, value=value)
            op_txt = op_text_1d(op, value=value)
        elif op == "sort":
            final = apply_1d(init, op)
            op_txt = op_text_1d(op)
        else:
            final = apply_1d(init, op)
            op_txt = op_text_1d(op)
        instruction = f"Single operation only. Initial {init}. Apply {op_txt}. Return final list."
        if instruction in seen:
            continue
        seen.add(instruction)
        out.append(task_single(start_id + len(out), "easy", instruction, final))
    return out


def gen_single_medium(start_id):
    rng = random.Random(SEED + 22)
    weighted = [("remove", 2), ("insert", 2), ("append", 2), ("sort", 2), ("reverse", 2)]
    out = []
    seen = set()
    while len(out) < N_TOTAL:
        init = make_1d(rng, "medium")
        op = _weighted_choice(rng, weighted)
        lo, hi = RANGES["medium"]
        if op == "remove":
            value = rng.choice(init)
            final = apply_1d(init, op, value=value)
            op_txt = op_text_1d(op, value=value)
        elif op == "insert":
            value = _sample_unique(rng, lo, hi, 1, forbid=init)[0]
            idx = rng.randint(0, len(init))
            final = apply_1d(init, op, value=value, index=idx)
            op_txt = op_text_1d(op, value=value, index=idx)
        elif op == "append":
            value = _sample_unique(rng, lo, hi, 1, forbid=init)[0]
            final = apply_1d(init, op, value=value)
            op_txt = op_text_1d(op, value=value)
        elif op == "sort":
            final = apply_1d(init, op)
            op_txt = op_text_1d(op)
        else:
            final = apply_1d(init, op)
            op_txt = op_text_1d(op)
        instruction = f"Single operation only. Initial {init}. Apply {op_txt}. Return final list."
        if instruction in seen:
            continue
        seen.add(instruction)
        out.append(task_single(start_id + len(out), "medium", instruction, final))
    return out


def gen_single_hard(start_id):
    rng = random.Random(SEED + 33)
    weighted = [("sort", 4), ("reverse", 4), ("insert", 2), ("remove", 2), ("append", 1)]
    out = []
    seen = set()
    while len(out) < N_TOTAL:
        init = make_2d_hard(rng)
        rows, cols = len(init), len(init[0])
        lo, hi = RANGES["hard"]
        op = _weighted_choice(rng, weighted)
        axis = rng.choice([0, 1])
        if op == "sort":
            final = apply_2d(init, op, axis=axis)
            op_txt = op_text_2d(op, axis=axis)
        elif op == "reverse":
            final = apply_2d(init, op, axis=axis)
            op_txt = op_text_2d(op, axis=axis)
        elif op == "remove":
            if axis == 0:
                idx = rng.randint(0, rows - 1)
            else:
                idx = rng.randint(0, cols - 1)
            final = apply_2d(init, op, axis=axis, index=idx)
            op_txt = op_text_2d(op, axis=axis, index=idx)
        elif op == "insert":
            if axis == 0:
                idx = rng.randint(0, rows)
                row = _sample_unique(rng, lo, hi, cols)
                final = apply_2d(init, op, axis=axis, index=idx, value=row)
                op_txt = op_text_2d(op, axis=axis, index=idx, value=row)
            else:
                idx = rng.randint(0, cols)
                value = _sample_unique(rng, lo, hi, 1)[0]
                final = apply_2d(init, op, axis=axis, index=idx, value=value)
                op_txt = op_text_2d(op, axis=axis, index=idx, value=value)
        else:  # append
            if axis == 0:
                row = _sample_unique(rng, lo, hi, cols)
                final = apply_2d(init, op, axis=axis, value=row)
                op_txt = op_text_2d(op, axis=axis, value=row)
            else:
                value = _sample_unique(rng, lo, hi, 1)[0]
                final = apply_2d(init, op, axis=axis, value=value)
                op_txt = op_text_2d(op, axis=axis, value=value)
        instruction = f"Single operation only on 2D list. Initial {init}. Apply {op_txt}. Return final 2D list."
        if instruction in seen:
            continue
        seen.add(instruction)
        out.append(task_single(start_id + len(out), "hard", instruction, final))
    return out


def validate(tasks, label, expected_n=N_TOTAL):
    if len(tasks) != expected_n:
        raise ValueError(f"{label}: expected {expected_n}, got {len(tasks)}")
    inst = [re.sub(r"\\s+", " ", t["instruction"].strip().lower()) for t in tasks]
    if len(inst) != len(set(inst)):
        raise ValueError(f"{label}: duplicate instructions")


def split_train_test(tasks):
    return tasks[:N_SPLIT], tasks[N_SPLIT:]


def main():
    generators = {
        "list_single_easy": gen_single_easy(2000),
        "list_single_medium": gen_single_medium(2100),
        "list_single_hard": gen_single_hard(2200),
    }
    for name, tasks in generators.items():
        validate(tasks, name)
        train, test = split_train_test(tasks)
        for split_name, split_tasks in [("train", train), ("test", test)]:
            fname = f"{name}_{split_name}.json"
            (OUT / fname).write_text(json.dumps(split_tasks, ensure_ascii=False, indent=2) + "\n")
            print(f"wrote {fname}: {len(split_tasks)}")


if __name__ == "__main__":
    main()
