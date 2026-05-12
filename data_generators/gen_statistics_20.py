"""Generate single-step Statistics tasks (easy / medium / hard)."""
import json, re
import math
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "data" / "task_sets"
OUT.mkdir(parents=True, exist_ok=True)
N_TOTAL = 70
N_SPLIT = 20
SEED = 20260304


def _rng(offset=0):
    return random.Random(SEED + offset)


def stats_env():
    return [
        {
            "name": "StatisticsEnv",
            "tools": ["compute_stat", "describe"],
            "parameters": {},
        }
    ]


def _mean(data):
    return sum(data) / len(data)


def _median(data):
    s = sorted(data)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _std(data):
    m = _mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / len(data))


def _correlation(x, y):
    n = len(x)
    mx, my = _mean(x), _mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x) / n)
    sy = math.sqrt(sum((b - my) ** 2 for b in y) / n)
    return cov / (sx * sy)


def task_single(tid, difficulty, instruction, answer):
    return {
        "id": tid,
        "difficulty": difficulty,
        "multi_step": False,
        "instruction": instruction,
        "environments": stats_env(),
        "expected": {"answer": str(answer)},
        "tags": ["statistics", "single_round"],
    }


# ---- Easy: mean/median of 3-5 small integers ----
def gen_single_easy(start_id):
    rng = _rng(100)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = len(out) % 4
        n = rng.randint(3, 5)
        data = [rng.randint(1, 20) for _ in range(n)]
        data_str = str(data)
        if data_str in seen:
            continue
        seen.add(data_str)
        if p == 0:
            ans = _mean(data)
            ans_str = f"{ans:.2f}" if ans != int(ans) else str(int(ans))
            instr = f"What is the mean of {data_str}? Answer with only the number."
        elif p == 1:
            ans = _median(data)
            ans_str = f"{ans:.1f}" if ans != int(ans) else str(int(ans))
            instr = f"What is the median of {data_str}? Answer with only the number."
        elif p == 2:
            ans = sum(data)
            ans_str = str(ans)
            instr = f"What is the sum of {data_str}? Answer with only the number."
        else:
            ans = max(data) - min(data)
            ans_str = str(ans)
            instr = f"What is the range (max minus min) of {data_str}? Answer with only the number."
        out.append(task_single(start_id + len(out), "easy", instr, ans_str))
    return out


# ---- Medium: std dev, variance, percentile of 8-15 numbers ----
def gen_single_medium(start_id):
    rng = _rng(200)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = len(out) % 4
        n = rng.randint(8, 15)
        data = [rng.randint(1, 100) for _ in range(n)]
        data_str = str(data)
        if data_str in seen:
            continue
        seen.add(data_str)
        if p == 0:
            ans = _std(data)
            ans_str = f"{ans:.2f}"
            instr = (f"What is the population standard deviation of {data_str}? "
                     f"Round to 2 decimal places. Answer with only the number.")
        elif p == 1:
            m = _mean(data)
            ans_str = f"{m:.2f}"
            instr = (f"What is the mean of {data_str}? "
                     f"Round to 2 decimal places. Answer with only the number.")
        elif p == 2:
            var = sum((x - _mean(data)) ** 2 for x in data) / len(data)
            ans_str = f"{var:.2f}"
            instr = (f"What is the population variance of {data_str}? "
                     f"Round to 2 decimal places. Answer with only the number.")
        else:
            ans = _median(data)
            ans_str = f"{ans:.1f}" if ans != int(ans) else str(int(ans))
            instr = (f"What is the median of {data_str}? "
                     f"Answer with only the number.")
        out.append(task_single(start_id + len(out), "medium", instr, ans_str))
    return out


# ---- Hard: correlation / std of 20-30 numbers ----
def gen_single_hard(start_id):
    rng = _rng(300)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = len(out) % 3
        if p == 0:
            # Correlation between two lists
            n = rng.randint(15, 25)
            x = [rng.randint(1, 200) for _ in range(n)]
            slope = rng.uniform(0.5, 3.0)
            noise = rng.uniform(5, 30)
            y = [round(slope * xi + rng.gauss(0, noise)) for xi in x]
            key = (str(x), str(y))
            if key in seen:
                continue
            seen.add(key)
            ans = _correlation(x, y)
            ans_str = f"{ans:.4f}"
            instr = (f"What is the Pearson correlation coefficient between "
                     f"X={x} and Y={y}? Round to 4 decimal places. Answer with only the number.")
        elif p == 1:
            # Std of large dataset
            n = rng.randint(20, 30)
            data = [rng.randint(1, 500) for _ in range(n)]
            key = str(data)
            if key in seen:
                continue
            seen.add(key)
            ans = _std(data)
            ans_str = f"{ans:.2f}"
            instr = (f"What is the population standard deviation of {data}? "
                     f"Round to 2 decimal places. Answer with only the number.")
        else:
            # Variance of large dataset
            n = rng.randint(20, 30)
            data = [rng.randint(1, 500) for _ in range(n)]
            key = str(data)
            if key in seen:
                continue
            seen.add(key)
            var = sum((x - _mean(data)) ** 2 for x in data) / len(data)
            ans_str = f"{var:.2f}"
            instr = (f"What is the population variance of {data}? "
                     f"Round to 2 decimal places. Answer with only the number.")
        out.append(task_single(start_id + len(out), "hard", instr, ans_str))
    return out


def validate(tasks, label, expected_n=N_TOTAL):
    if len(tasks) != expected_n:
        raise ValueError(f"{label}: expected {expected_n}, got {len(tasks)}")
    inst = [re.sub(r"\s+", " ", t["instruction"].strip().lower()) for t in tasks]
    if len(set(inst)) != len(inst):
        raise ValueError(f"{label}: duplicate instruction found")



def split_train_test(tasks):
    return tasks[:N_SPLIT], tasks[N_SPLIT:]


def main():
    generators = {
        "statistics_single_easy": gen_single_easy(2000),
        "statistics_single_medium": gen_single_medium(2100),
        "statistics_single_hard": gen_single_hard(2200),
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
