"""Generate single-step Counting tasks (easy / medium / hard)."""
import json, re, math, random
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "data" / "task_sets"
OUT.mkdir(parents=True, exist_ok=True)
N_TOTAL = 70; N_SPLIT = 20; SEED = 20260304

def _rng(offset=0): return random.Random(SEED + offset)

def counting_env():
    return [{"name": "CountingEnv", "tools": ["combination", "permutation", "factorial"], "parameters": {}}]

def task(tid, diff, instr, ans):
    return {"id": tid, "difficulty": diff, "multi_step": False, "instruction": instr,
            "environments": counting_env(), "expected": {"answer": str(ans)}, "tags": ["counting", "single_round"]}

def gen_single_easy(start_id):
    rng = _rng(110)
    out, seen = [], set()
    attempts = 0
    while len(out) < N_TOTAL and attempts < 500:
        attempts += 1
        p = rng.randint(0, 3)
        if p == 0:
            n = rng.randint(4, 12); k = rng.randint(1, n)
            ans = math.comb(n, k); instr = f"How many ways can you choose {k} items from {n}? Answer with only the number."
        elif p == 1:
            n = rng.randint(3, 8)
            ans = math.factorial(n); instr = f"What is {n}!? Answer with only the number."
        elif p == 2:
            n = rng.randint(3, 10); k = rng.randint(2, min(4, n))
            ans = math.perm(n, k); instr = f"How many ways to arrange {k} items from {n}? Answer with only the number."
        else:
            n = rng.randint(4, 12); k = rng.randint(1, n - 1)
            ans = math.comb(n, k); instr = f"Compute C({n},{k}). Answer with only the number."
        if instr in seen: continue
        seen.add(instr)
        out.append(task(start_id + len(out), "easy", instr, ans))
    return out

def gen_single_medium(start_id):
    rng = _rng(220)
    out, seen = [], set()
    attempts = 0
    while len(out) < N_TOTAL and attempts < 500:
        attempts += 1
        p = rng.randint(0, 3)
        if p == 0:
            n = rng.randint(15, 30); k = rng.randint(3, n // 2)
            ans = math.comb(n, k); instr = f"Compute C({n},{k}). Answer with only the number."
        elif p == 1:
            n = rng.randint(10, 16)
            ans = math.factorial(n); instr = f"What is {n}!? Answer with only the number."
        elif p == 2:
            n = rng.randint(10, 20); k = rng.randint(3, 7)
            ans = math.perm(n, k); instr = f"Compute P({n},{k}). Answer with only the number."
        else:
            n = rng.randint(12, 25); k = rng.randint(3, n // 2)
            ans = math.comb(n, k); instr = f"How many ways can you choose {k} items from {n}? Answer with only the number."
        if instr in seen: continue
        seen.add(instr)
        out.append(task(start_id + len(out), "medium", instr, ans))
    return out

def gen_single_hard(start_id):
    rng = _rng(330)
    out, seen = [], set()
    attempts = 0
    while len(out) < N_TOTAL and attempts < 500:
        attempts += 1
        p = rng.randint(0, 3)
        if p == 0:
            n = rng.randint(40, 100); k = rng.randint(10, n // 2)
            ans = math.comb(n, k); instr = f"Compute C({n},{k}). Answer with only the number."
        elif p == 1:
            n = rng.randint(18, 30)
            ans = math.factorial(n); instr = f"What is {n}!? Answer with only the number."
        elif p == 2:
            n = rng.randint(30, 80); k = rng.randint(5, 12)
            ans = math.perm(n, k); instr = f"Compute P({n},{k}). Answer with only the number."
        else:
            n = rng.randint(50, 120); k = rng.randint(10, n // 2)
            ans = math.comb(n, k); instr = f"How many ways can you choose {k} items from {n}? Answer with only the number."
        if instr in seen: continue
        seen.add(instr)
        out.append(task(start_id + len(out), "hard", instr, ans))
    return out

def validate(tasks, label, expected_n=N_TOTAL):
    if len(tasks) != expected_n:
        raise ValueError(f"{label}: expected {expected_n}, got {len(tasks)}")
    inst = [re.sub(r"\s+", " ", t["instruction"].strip().lower()) for t in tasks]
    if len(set(inst)) != len(inst):
        raise ValueError(f"{label}: duplicate instruction found")


def split_train_test(tasks): return tasks[:N_SPLIT], tasks[N_SPLIT:]

def main():
    gens = {"counting_single_easy": gen_single_easy(2300), "counting_single_medium": gen_single_medium(2400),
            "counting_single_hard": gen_single_hard(2500)}
    for name, tasks in gens.items():
        validate(tasks, name)
        train, test = split_train_test(tasks)
        for sn, st in [("train", train), ("test", test)]:
            fname = f"{name}_{sn}.json"
            (OUT / fname).write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n")
            print(f"wrote {fname}: {len(st)}")

if __name__ == "__main__": main()
