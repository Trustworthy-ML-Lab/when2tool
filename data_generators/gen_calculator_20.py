import json
import random
import re
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "data" / "task_sets"
OUT.mkdir(parents=True, exist_ok=True)
N_TOTAL = 70
N_SPLIT = 20
SEED = 20260304

# Difficulty-aware numeric ranges.
RANGES = {
    "easy": {
        "small": (2, 40),
        "mul": (2, 14),
    },
    "medium": {
        "a": (80, 900),
        "b": (7, 140),
        "c": (10, 300),
        "d": (2, 25),
        "mod": (17, 211),
    },
    "hard": {
        "big_a": (10**9, 10**11),
        "big_b": (10**10, 10**12),
        "big_c": (10**7, 10**9),
        "big_d": (10**5, 10**7),
        "square": (10**7, 10**9),
        "mod": (10**6, 10**8),
    },
}


def _rng():
    return random.Random(SEED)


def _ri(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


def calc_env():
    return [
        {
            "name": "CalculatorEnv",
            "tools": ["evaluate_expression", "get_last_result", "clear_last_result"],
            "parameters": {},
        }
    ]


def eval_int(expr: str) -> int:
    v = eval(expr, {"__builtins__": {}}, {})
    if isinstance(v, bool):
        raise ValueError("bool result not allowed")
    if isinstance(v, float):
        if not v.is_integer():
            raise ValueError(f"non-integer result for expression: {expr} -> {v}")
        return int(v)
    return int(v)


def task_single(task_id: int, difficulty: str, expr: str):
    if difficulty == "easy":
        instruction = f"Answer with only the number: {expr}"
    else:
        instruction = f"Compute exactly and return only number: {expr}"
    return {
        "id": task_id,
        "difficulty": difficulty,
        "multi_step": False,
        "instruction": instruction,
        "environments": calc_env(),
        "expected": {"answer": str(eval_int(expr))},
        "tags": ["calculator", "single_round"],
    }


def gen_single_easy(start_id: int):
    rng = _rng()
    out = []
    seen = set()
    while len(out) < N_TOTAL:
        p = len(out) % 5
        s_lo, s_hi = RANGES["easy"]["small"]
        m_lo, m_hi = RANGES["easy"]["mul"]
        if p == 0:
            a, b = _ri(rng, s_lo, s_hi), _ri(rng, s_lo, s_hi)
            expr = f"{a}+{b}"
        elif p == 1:
            a, b = _ri(rng, s_lo + 10, s_hi + 20), _ri(rng, s_lo, s_hi)
            if a < b:
                a, b = b + 1, a
            expr = f"{a}-{b}"
        elif p == 2:
            a, b = _ri(rng, m_lo, m_hi), _ri(rng, m_lo, m_hi)
            expr = f"{a}*{b}"
        elif p == 3:
            a, b, c = _ri(rng, s_lo, s_hi), _ri(rng, s_lo, s_hi), _ri(rng, 1, 20)
            expr = f"({a}+{b})-{c}"
        else:
            a, b, c = _ri(rng, m_lo, m_hi), _ri(rng, m_lo, m_hi), _ri(rng, 1, 30)
            expr = f"({a}*{b})+{c}"
        if expr in seen:
            continue
        seen.add(expr)
        out.append(task_single(start_id + len(out), "easy", expr))
    return out


def gen_single_medium(start_id: int):
    rng = _rng()
    out = []
    seen = set()
    while len(out) < N_TOTAL:
        p = len(out) % 5
        a_lo, a_hi = RANGES["medium"]["a"]
        b_lo, b_hi = RANGES["medium"]["b"]
        c_lo, c_hi = RANGES["medium"]["c"]
        d_lo, d_hi = RANGES["medium"]["d"]
        m_lo, m_hi = RANGES["medium"]["mod"]
        if p == 0:
            a, b = _ri(rng, a_lo, a_hi), _ri(rng, b_lo, b_hi)
            c, d = _ri(rng, c_lo, c_hi), _ri(rng, c_lo, c_hi)
            expr = f"({a}*{b})-{c}+{d}"
        elif p == 1:
            a, b = _ri(rng, a_lo, a_hi), _ri(rng, b_lo, b_hi)
            c, d = _ri(rng, d_lo, d_hi), _ri(rng, c_lo, c_hi)
            expr = f"(({a}+{b})*{c})-{d}"
        elif p == 2:
            a, b = _ri(rng, a_lo * 3, a_hi * 4), _ri(rng, d_lo + 2, d_hi + 7)
            c, d = _ri(rng, c_lo, c_hi), _ri(rng, b_lo, b_hi)
            e = _ri(rng, c_lo, c_hi)
            expr = f"({a}//{b})+({c}*{d})-{e}"
        elif p == 3:
            a, m = _ri(rng, a_lo * 5, a_hi * 8), _ri(rng, m_lo, m_hi)
            b, c = _ri(rng, d_lo + 3, d_hi + 9), _ri(rng, c_lo, c_hi)
            expr = f"({a}%{m})*{b}+{c}"
        else:
            a, b = _ri(rng, a_lo, a_hi), _ri(rng, a_lo, a_hi)
            c, d = _ri(rng, c_lo, c_hi), _ri(rng, d_lo + 1, d_hi + 5)
            expr = f"(({a}*{b})+{c})//{d}"
        if expr in seen:
            continue
        seen.add(expr)
        out.append(task_single(start_id + len(out), "medium", expr))
    return out


def gen_single_hard(start_id: int):
    rng = _rng()
    out = []
    seen = set()
    while len(out) < N_TOTAL:
        p = len(out) % 5
        ra = RANGES["hard"]["big_a"]
        rb = RANGES["hard"]["big_b"]
        rc = RANGES["hard"]["big_c"]
        rd = RANGES["hard"]["big_d"]
        rs = RANGES["hard"]["square"]
        rm = RANGES["hard"]["mod"]
        if p == 0:
            a, b, c = _ri(rng, *ra), _ri(rng, *rb), _ri(rng, *rc)
            expr = f"({a}*{b})-{c}"
        elif p == 1:
            a, b = _ri(rng, *rs), _ri(rng, *ra)
            expr = f"({a}**2)+{b}"
        elif p == 2:
            a, b = _ri(rng, *rb), _ri(rng, *rb)
            d, e = _ri(rng, 7, 19), _ri(rng, *rc)
            expr = f"(({a}*{b})//{d})-{e}"
        elif p == 3:
            a, b, c = _ri(rng, *rb), _ri(rng, *ra), _ri(rng, *ra)
            m = _ri(rng, *rm)
            expr = f"(({a}+{b})*{c})%{m}"
        else:
            a, b = _ri(rng, *ra), _ri(rng, *rb)
            c, d, e = _ri(rng, *rd), _ri(rng, *rd), _ri(rng, *rc)
            expr = f"(({a}*{b})+({c}*{d})-{e})"
        if expr in seen:
            continue
        seen.add(expr)
        out.append(task_single(start_id + len(out), "hard", expr))
    return out


def validate(tasks, label: str, expected_n: int = N_TOTAL):
    if len(tasks) != expected_n:
        raise ValueError(f"{label}: expected {expected_n}, got {len(tasks)}")
    inst = [re.sub(r"\s+", " ", t["instruction"].strip().lower()) for t in tasks]
    if len(set(inst)) != len(inst):
        raise ValueError(f"{label}: duplicate instruction found")


def split_train_test(tasks):
    return tasks[:N_SPLIT], tasks[N_SPLIT:]


def main():
    generators = {
        "calculator_single_easy": gen_single_easy(1000),
        "calculator_single_medium": gen_single_medium(1100),
        "calculator_single_hard": gen_single_hard(1200),
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
