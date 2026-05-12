"""Generate multi-hop (3-step chained) Calculator tasks (Env 16: ChainedCalculatorEnv).

Each task requires 3 sequential tool calls where each depends on the previous result.
Pattern: compute x = expr1, then y = expr2(x), then z = expr3(y). Return z.
"""
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


def _rng():
    return random.Random(SEED + 160)


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
            raise ValueError(f"non-integer result: {expr} -> {v}")
        return int(v)
    return int(v)


def _make_task(task_id, difficulty, expr1, expr2_template, expr3_template):
    """Build a 3-step chained calculator task.

    expr1: literal expression string (no dependencies)
    expr2_template: expression string with '{x}' placeholder
    expr3_template: expression string with '{y}' placeholder
    """
    x = eval_int(expr1)
    expr2 = expr2_template.format(x=x)
    y = eval_int(expr2)
    expr3 = expr3_template.format(y=y)
    z = eval_int(expr3)

    instruction = (
        f"Three-step computation: First compute x = {expr1}. "
        f"Then compute y = {expr2_template.replace('{x}', 'x')}. "
        f"Finally compute z = {expr3_template.replace('{y}', 'y')}. "
        f"Return only z."
    )

    return {
        "id": task_id,
        "difficulty": difficulty,
        "multi_step": True,
        "instruction": instruction,
        "environments": calc_env(),
        "expected": {
            "answer": str(z),
            "steps": [str(x), str(y), str(z)],
        },
        "tags": ["calculator", "multihop_3step"],
    }


def gen_easy(start_id):
    """Easy: small numbers (2-40), simple ops (+, -, *)."""
    rng = _rng()
    out = []
    seen = set()

    while len(out) < N_TOTAL:
        p = len(out) % 7
        if p == 0:
            a, b = _ri(rng, 2, 20), _ri(rng, 2, 20)
            c = _ri(rng, 2, 10)
            d = _ri(rng, 2, 15)
            expr1 = f"{a}+{b}"
            expr2_t = f"{{x}}*{c}"
            expr3_t = f"{{y}}-{d}"
        elif p == 1:
            a, b = _ri(rng, 3, 12), _ri(rng, 3, 12)
            c = _ri(rng, 2, 8)
            d = _ri(rng, 5, 30)
            expr1 = f"{a}*{b}"
            expr2_t = f"{{x}}+{c}"
            expr3_t = f"{{y}}*{d}"
        elif p == 2:
            a = _ri(rng, 20, 40)
            b = _ri(rng, 2, 15)
            c = _ri(rng, 3, 10)
            d = _ri(rng, 2, 8)
            expr1 = f"{a}-{b}"
            expr2_t = f"{{x}}*{c}"
            expr3_t = f"{{y}}+{d}"
        elif p == 3:
            a, b = _ri(rng, 2, 10), _ri(rng, 2, 10)
            c = _ri(rng, 5, 20)
            d = _ri(rng, 2, 6)
            expr1 = f"{a}+{b}"
            expr2_t = f"{{x}}+{c}"
            expr3_t = f"{{y}}*{d}"
        elif p == 4:
            a, b = _ri(rng, 3, 15), _ri(rng, 3, 15)
            c = _ri(rng, 2, 5)
            d = _ri(rng, 10, 30)
            expr1 = f"{a}*{b}"
            expr2_t = f"{{x}}-{c}"
            expr3_t = f"{{y}}+{d}"
        elif p == 5:
            a, b = _ri(rng, 5, 25), _ri(rng, 2, 10)
            c = _ri(rng, 2, 8)
            d = _ri(rng, 2, 5)
            expr1 = f"{a}+{b}"
            expr2_t = f"{{x}}-{c}"
            expr3_t = f"{{y}}*{d}"
        else:
            a = _ri(rng, 10, 40)
            b = _ri(rng, 2, 10)
            c = _ri(rng, 3, 12)
            d = _ri(rng, 2, 20)
            expr1 = f"{a}-{b}"
            expr2_t = f"{{x}}+{c}"
            expr3_t = f"{{y}}-{d}"

        # Build instruction for dedup check
        x = eval_int(expr1)
        expr2 = expr2_t.format(x=x)
        y = eval_int(expr2)
        expr3 = expr3_t.format(y=y)
        key = (expr1, expr2_t, expr3_t)
        if key in seen:
            continue
        seen.add(key)
        out.append(_make_task(start_id + len(out), "easy", expr1, expr2_t, expr3_t))

    return out


def gen_medium(start_id):
    """Medium: 3-digit numbers (80-900), multiplication/division/modulo."""
    rng = _rng()
    out = []
    seen = set()

    while len(out) < N_TOTAL:
        p = len(out) % 7
        if p == 0:
            a, b = _ri(rng, 80, 400), _ri(rng, 2, 25)
            c = _ri(rng, 10, 100)
            d = _ri(rng, 17, 211)
            expr1 = f"{a}*{b}"
            expr2_t = f"{{x}}+{c}"
            expr3_t = f"{{y}}%{d}"
        elif p == 1:
            a = _ri(rng, 200, 900)
            b = _ri(rng, 80, 300)
            c = _ri(rng, 3, 20)
            d = _ri(rng, 50, 300)
            expr1 = f"{a}+{b}"
            expr2_t = f"{{x}}*{c}"
            expr3_t = f"{{y}}//{d}"
        elif p == 2:
            a, b = _ri(rng, 100, 500), _ri(rng, 2, 15)
            c = _ri(rng, 80, 300)
            d = _ri(rng, 3, 12)
            expr1 = f"{a}*{b}"
            expr2_t = f"{{x}}-{c}"
            expr3_t = f"{{y}}*{d}"
        elif p == 3:
            a = _ri(rng, 300, 900)
            b = _ri(rng, 100, 400)
            c = _ri(rng, 17, 100)
            d = _ri(rng, 2, 15)
            expr1 = f"{a}-{b}"
            expr2_t = f"{{x}}%{c}"
            expr3_t = f"{{y}}*{d}"
        elif p == 4:
            a, b = _ri(rng, 80, 500), _ri(rng, 10, 50)
            c = _ri(rng, 2, 8)
            d = _ri(rng, 100, 400)
            expr1 = f"{a}*{b}"
            expr2_t = f"{{x}}//{c}"
            expr3_t = f"{{y}}+{d}"
        elif p == 5:
            a = _ri(rng, 100, 900)
            b = _ri(rng, 80, 300)
            c = _ri(rng, 5, 25)
            d = _ri(rng, 17, 150)
            expr1 = f"{a}+{b}"
            expr2_t = f"{{x}}*{c}"
            expr3_t = f"{{y}}%{d}"
        else:
            a, b = _ri(rng, 100, 400), _ri(rng, 3, 20)
            c = _ri(rng, 2, 10)
            d = _ri(rng, 3, 15)
            expr1 = f"{a}*{b}"
            expr2_t = f"{{x}}%{c}"
            expr3_t = f"{{y}}*{d}"

        key = (expr1, expr2_t, expr3_t)
        if key in seen:
            continue
        seen.add(key)
        out.append(_make_task(start_id + len(out), "medium", expr1, expr2_t, expr3_t))

    return out


def gen_hard(start_id):
    """Hard: numbers in 10^9-10^12 range."""
    rng = _rng()
    out = []
    seen = set()

    while len(out) < N_TOTAL:
        p = len(out) % 7
        if p == 0:
            a = _ri(rng, 10**9, 10**11)
            b = _ri(rng, 10**9, 10**10)
            c = _ri(rng, 10**6, 10**8)
            d = _ri(rng, 7, 19)
            expr1 = f"{a}+{b}"
            expr2_t = f"{{x}}*{c}"
            expr3_t = f"{{y}}//{d}"
        elif p == 1:
            a = _ri(rng, 10**9, 10**11)
            b = _ri(rng, 10**10, 10**12)
            c = _ri(rng, 10**6, 10**8)
            d = _ri(rng, 10**7, 10**9)
            expr1 = f"{a}*{b}"
            expr2_t = f"{{x}}%{c}"
            expr3_t = f"{{y}}+{d}"
        elif p == 2:
            a = _ri(rng, 10**7, 10**9)
            b = _ri(rng, 10**7, 10**9)
            c = _ri(rng, 10**5, 10**7)
            d = _ri(rng, 10**6, 10**8)
            expr1 = f"({a}**2)"
            expr2_t = f"{{x}}-{c}"
            expr3_t = f"{{y}}%{d}"
        elif p == 3:
            a = _ri(rng, 10**10, 10**12)
            b = _ri(rng, 10**9, 10**11)
            c = _ri(rng, 10**5, 10**7)
            d = _ri(rng, 10**6, 10**8)
            expr1 = f"{a}-{b}"
            expr2_t = f"{{x}}*{c}"
            expr3_t = f"{{y}}%{d}"
        elif p == 4:
            a = _ri(rng, 10**9, 10**11)
            b = _ri(rng, 10**10, 10**12)
            c = _ri(rng, 7, 19)
            d = _ri(rng, 10**7, 10**9)
            expr1 = f"{a}*{b}"
            expr2_t = f"{{x}}//{c}"
            expr3_t = f"{{y}}-{d}"
        elif p == 5:
            a = _ri(rng, 10**9, 10**11)
            b = _ri(rng, 10**9, 10**10)
            c = _ri(rng, 10**6, 10**8)
            d = _ri(rng, 10**5, 10**7)
            expr1 = f"{a}+{b}"
            expr2_t = f"{{x}}%{c}"
            expr3_t = f"{{y}}*{d}"
        else:
            a = _ri(rng, 10**10, 10**12)
            b = _ri(rng, 10**9, 10**10)
            c = _ri(rng, 10**9, 10**11)
            d = _ri(rng, 10**6, 10**8)
            expr1 = f"{a}-{b}"
            expr2_t = f"{{x}}+{c}"
            expr3_t = f"{{y}}%{d}"

        key = (expr1, expr2_t, expr3_t)
        if key in seen:
            continue
        seen.add(key)
        out.append(_make_task(start_id + len(out), "hard", expr1, expr2_t, expr3_t))

    return out


def _normalize_inst(x):
    return re.sub(r"\s+", " ", x.strip().lower())


def validate(tasks, label, expected_n=N_TOTAL):
    if len(tasks) != expected_n:
        raise ValueError(f"{label}: expected {expected_n}, got {len(tasks)}")
    inst = [_normalize_inst(t["instruction"]) for t in tasks]
    if len(set(inst)) != len(inst):
        raise ValueError(f"{label}: duplicate instruction found")
    for t in tasks:
        steps = t["expected"]["steps"]
        if len(steps) != 3:
            raise ValueError(f"{label}: expected 3 steps, got {len(steps)}")
        if t["expected"]["answer"] != steps[-1]:
            raise ValueError(f"{label}: answer != last step")


def split_train_test(tasks):
    return tasks[:N_SPLIT], tasks[N_SPLIT:]


def main():
    generators = {
        "calculator_multihop_easy": gen_easy(16000),
        "calculator_multihop_medium": gen_medium(16100),
        "calculator_multihop_hard": gen_hard(16200),
    }

    for name, tasks in generators.items():
        validate(tasks, name)
        train, test = split_train_test(tasks)
        for split_name, split_tasks in [("train", train), ("test", test)]:
            fname = f"{name}_{split_name}.json"
            (OUT / fname).write_text(
                json.dumps(split_tasks, ensure_ascii=False, indent=2) + "\n"
            )
            print(f"wrote {fname}: {len(split_tasks)}")


if __name__ == "__main__":
    main()
