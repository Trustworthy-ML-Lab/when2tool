"""Generate single-step Matrix tasks (easy / medium / hard)."""
import json, re, random
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "data" / "task_sets"
OUT.mkdir(parents=True, exist_ok=True)
N_TOTAL = 70; N_SPLIT = 20; SEED = 20260304

def _rng(offset=0): return random.Random(SEED + offset)

def matrix_env():
    return [{"name": "MatrixEnv", "tools": ["matrix_determinant", "matrix_multiply", "matrix_trace"], "parameters": {}}]

def _det(m):
    n = len(m)
    if n == 1: return m[0][0]
    if n == 2: return m[0][0]*m[1][1] - m[0][1]*m[1][0]
    d = 0
    for c in range(n):
        sub = [[m[r][j] for j in range(n) if j != c] for r in range(1, n)]
        d += ((-1)**c) * m[0][c] * _det(sub)
    return d

def task(tid, diff, instr, ans):
    return {"id": tid, "difficulty": diff, "multi_step": False, "instruction": instr,
            "environments": matrix_env(), "expected": {"answer": str(ans)}, "tags": ["matrix", "single_round"]}

def _mat_str(m):
    return "[" + ", ".join("[" + ", ".join(str(x) for x in row) + "]" for row in m) + "]"

def gen_single_easy(start_id):
    rng = _rng(140)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = len(out) % 3
        if p == 0:  # 2x2 determinant
            m = [[rng.randint(-5, 10) for _ in range(2)] for _ in range(2)]
            ans = _det(m)
            instr = f"What is the determinant of {_mat_str(m)}? Answer with only the number."
        elif p == 1:  # 2x2 trace
            m = [[rng.randint(-5, 10) for _ in range(2)] for _ in range(2)]
            ans = m[0][0] + m[1][1]
            instr = f"What is the trace of {_mat_str(m)}? Answer with only the number."
        else:  # 2x2 determinant different range
            m = [[rng.randint(1, 15) for _ in range(2)] for _ in range(2)]
            ans = _det(m)
            instr = f"Compute the determinant of the matrix {_mat_str(m)}. Answer with only the number."
        key = _mat_str(m)
        if key in seen: continue
        seen.add(key)
        out.append(task(start_id + len(out), "easy", instr, ans))
    return out

def gen_single_medium(start_id):
    rng = _rng(250)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = len(out) % 3
        if p == 0:  # 3x3 determinant
            m = [[rng.randint(-10, 15) for _ in range(3)] for _ in range(3)]
            ans = _det(m)
            instr = f"What is the determinant of {_mat_str(m)}? Answer with only the number."
        elif p == 1:  # 3x3 trace
            m = [[rng.randint(-10, 15) for _ in range(3)] for _ in range(3)]
            ans = sum(m[i][i] for i in range(3))
            instr = f"What is the trace of {_mat_str(m)}? Answer with only the number."
        else:  # 3x3 det larger values
            m = [[rng.randint(-20, 30) for _ in range(3)] for _ in range(3)]
            ans = _det(m)
            instr = f"Compute the determinant of {_mat_str(m)}. Answer with only the number."
        key = _mat_str(m)
        if key in seen: continue
        seen.add(key)
        out.append(task(start_id + len(out), "medium", instr, ans))
    return out

def gen_single_hard(start_id):
    rng = _rng(360)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = len(out) % 3
        if p == 0:  # 4x4 determinant
            m = [[rng.randint(-20, 30) for _ in range(4)] for _ in range(4)]
        elif p == 1:  # 5x5 determinant
            m = [[rng.randint(-10, 15) for _ in range(5)] for _ in range(5)]
        else:  # 4x4 large values
            m = [[rng.randint(-50, 50) for _ in range(4)] for _ in range(4)]
        ans = _det(m)
        key = _mat_str(m)
        if key in seen: continue
        seen.add(key)
        n = len(m)
        instr = f"What is the determinant of the {n}x{n} matrix {_mat_str(m)}? Answer with only the number."
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
    gens = {"matrix_single_easy": gen_single_easy(2600), "matrix_single_medium": gen_single_medium(2700),
            "matrix_single_hard": gen_single_hard(2800)}
    for name, tasks in gens.items():
        validate(tasks, name)
        train, test = split_train_test(tasks)
        for sn, st in [("train", train), ("test", test)]:
            fname = f"{name}_{sn}.json"
            (OUT / fname).write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n")
            print(f"wrote {fname}: {len(st)}")

if __name__ == "__main__": main()
