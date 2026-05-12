"""Generate single-step Code Executor tasks (easy / medium / hard)."""
import io
import contextlib
import json, re
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


def code_env():
    return [{"name": "CodeExecutorEnv", "tools": ["run_code"], "parameters": {}}]


def _run(code):
    """Execute code and capture stdout, matching CodeExecutorEnv.run_code behavior."""
    stdout = io.StringIO()
    globs = {"__builtins__": __builtins__}
    with contextlib.redirect_stdout(stdout):
        exec(code, globs, globs)
    return stdout.getvalue().rstrip("\n")


def task_single(tid, difficulty, instruction, answer):
    return {
        "id": tid, "difficulty": difficulty, "multi_step": False,
        "instruction": instruction, "environments": code_env(),
        "expected": {"answer": str(answer)}, "tags": ["code_executor", "single_round"],
    }


# ---- Easy: simple one-line expressions and basic operations ----
EASY_SNIPPETS = [
    "print(2 + 3)",
    "print(10 * 7)",
    "print(100 - 37)",
    "print(len('hello'))",
    "print('hello' + ' ' + 'world')",
    "print(max(3, 7, 1))",
    "print(min(5, 2, 8))",
    "print(abs(-42))",
    "print(2 ** 10)",
    "print(17 % 5)",
    "print(17 // 3)",
    "print(sorted([3, 1, 2]))",
    "print(list(range(5)))",
    "print('abc'.upper())",
    "print('HELLO'.lower())",
    "print(len([1, 2, 3, 4, 5]))",
    "print(sum([1, 2, 3, 4, 5]))",
    "print(type(42).__name__)",
    "print(bool(0))",
    "print(bool(1))",
    "print(3 in [1, 2, 3])",
    "print('a' in 'cat')",
    "print(round(3.14159, 2))",
    "print(divmod(17, 5))",
    "x = 3; y = x + 2; print(y)",
    "x = 10; y = x * 2; print(y - 5)",
    "print(7 + 8)",
    "print(50 // 7)",
    "print(2 ** 8)",
    "print(99 - 42)",
    "print(len('python'))",
    "print('ab' * 3)",
    "print(int('42'))",
    "print(str(123))",
    "print(float(7))",
    "print(list('abc'))",
    "print(tuple([1,2,3]))",
    "print(set([1,2,2,3]))",
    "print(9 ** 0.5)",
    "print(pow(2, 5))",
    "print(chr(65))",
    "print(ord('A'))",
    "print(hex(255))",
    "print(bin(10))",
    # Additional easy snippets
    "print(oct(8))",
    "print(3 == 3)",
    "print(3 != 4)",
    "print(5 > 3)",
    "print(2 < 1)",
    "print(10 >= 10)",
    "print(not True)",
    "print(True and False)",
    "print(True or False)",
    "print(len({}))",
    "print(len(set()))",
    "print('hello'[1])",
    "print('hello'[-1])",
    "print('hello'[1:3])",
    "print([1,2,3] + [4,5])",
    "print([0] * 5)",
    "print(max('apple', 'banana'))",
    "print(min('cat', 'bat'))",
    "print('hello world'.split())",
    "print('-'.join(['a','b','c']))",
    "print('  spaces  '.strip())",
    "print('hello'.startswith('he'))",
    "print('hello'.endswith('lo'))",
    "print('hello'.count('l'))",
    "print('hello'.find('ll'))",
    "print('hello'.replace('l', 'r'))",
    "print('hello'.index('e'))",
    "print(isinstance(42, int))",
    "print(isinstance('hi', str))",
    "print(all([True, True, False]))",
    "print(any([False, False, True]))",
]

# ---- Medium: loops, comprehensions, string operations ----
MEDIUM_SNIPPETS = [
    "print([x**2 for x in range(6)])",
    "print(sum(x for x in range(1, 11) if x % 2 == 0))",
    "print(''.join(reversed('python')))",
    "print({x: x**2 for x in range(5)})",
    "print(sorted('hello world'.split()))",
    "print(list(zip([1,2,3], ['a','b','c'])))",
    "print([i for i in range(20) if i % 3 == 0])",
    "s = 'abcdef'; print(s[::2])",
    "print(len(set([1, 2, 2, 3, 3, 3])))",
    "x = [1, 2, 3]; x.append(4); print(x)",
    "print(' '.join(['hello', 'world', 'foo']))",
    "print(list(filter(lambda x: x > 3, [1, 5, 2, 7, 3])))",
    "print(list(map(lambda x: x*2, [1, 2, 3, 4])))",
    "d = {'a': 1, 'b': 2, 'c': 3}; print(sorted(d.keys()))",
    "print([x for x in range(2, 20) if all(x % d != 0 for d in range(2, x))])",
    "print('hello world'.count('l'))",
    "print('hello world'.replace('world', 'python'))",
    "print(bin(42))",
    "print(hex(1023))",
    "print(oct(511))",
    "a, b = 3, 5; a, b = b, a; print(a, b)",
    "print(sum(1 for c in 'Hello World' if c.isupper()))",
    "s = 0\nfor i in range(8):\n    s += i * i\nprint(s)",
    "s = 1\nfor i in range(1, 7):\n    s *= i\nprint(s)",
    "result = []\nfor i in range(10):\n    if i % 2 == 0:\n        result.append(i)\nprint(result)",
    "s = ''\nfor c in 'hello':\n    s = c + s\nprint(s)",
    "n = 100\ns = 0\nfor i in range(1, n+1):\n    if i % 3 == 0 or i % 5 == 0:\n        s += i\nprint(s)",
    "x = 1\nfor _ in range(10):\n    x = x * 2\nprint(x)",
    "print(sorted([5,3,8,1,9,2,7,4,6], reverse=True))",
    "print('_'.join(str(i) for i in range(5)))",
    "print(list(enumerate('abc')))",
    "nums = [1,2,3,4,5]\nprint(nums[::-1])",
    "print({i: i**3 for i in range(1, 6)})",
    "print([c for c in 'Hello World' if c.islower()])",
    "print(max([3, 1, 4, 1, 5, 9, 2, 6]))",
    "print(min(abs(x) for x in [-3, 1, -4, 2]))",
    "print(''.join(sorted('python')))",
    "print(len([x for x in range(1, 51) if x % 7 == 0]))",
    "d = dict(zip('abc', [1,2,3])); print(d)",
    "print(sum(range(1, 21)))",
    "print([x*x for x in range(1, 8)])",
    "print('abcabc'.count('abc'))",
    # Additional medium snippets
    "print([i for i in range(2, 30) if all(i % j != 0 for j in range(2, int(i**0.5)+1))])",
    "print(list(zip([1,2,3], [4,5,6])))",
    "print(list(zip(*[(1,4),(2,5),(3,6)])))",
    "d = {}; [d.update({c: d.get(c,0)+1}) for c in 'mississippi']; print(d)",
    "print(sorted('banana', key=lambda c: (-'banana'.count(c), c)))",
    "print([x for x in range(100) if x == sum(int(d)**3 for d in str(x))])",
    "print(list(range(10, 0, -2)))",
    "print([bin(i).count('1') for i in range(8)])",
    "a = [1,2,3,4,5]; a.insert(2, 99); print(a)",
    "a = [1,2,3,4,5]; a.pop(2); print(a)",
    "s = set([1,2,3]); s.add(4); s.discard(2); print(sorted(s))",
    "print({k:v for k,v in enumerate('hello')})",
    "print('hello world'.title())",
    "print('Hello World'.swapcase())",
    "print('abc' * 3 + 'def')",
    "print([i*j for i in range(1,4) for j in range(1,4)])",
    "x = [[1,2],[3,4]]; print([item for row in x for item in row])",
    "print(len(set('mississippi')))",
    "print(sorted(set('mississippi')))",
    "print(''.join(c for c in 'Hello World 123' if c.isalpha()))",
    "print(sum(int(d) for d in '12345'))",
    "a = list(range(10)); a[::2] = [0]*5; print(a)",
    "print([x for x in range(1, 20) if x % 2 != 0 and x % 3 != 0])",
    "d = {'a':1,'b':2,'c':3}; print(list(d.values()))",
    "d = {'a':1,'b':2,'c':3}; print(list(d.items()))",
    "print(max(enumerate([4,2,7,1,9]), key=lambda x: x[1]))",
    "s = 'hello'; print(s.center(11, '*'))",
    "print('42'.zfill(6))",
    "print(list(reversed(range(5))))",
    "print([chr(i) for i in range(97, 104)])",
]

# ---- Hard: recursion, DP, nested loops requiring many steps to trace ----
def _gen_hard_snippets():
    """Generate hard code snippets using templates with varying parameters."""
    snippets = []

    # Recursion: factorial with large n (9 variants)
    for n in [12, 15, 13, 14, 11, 16, 17, 10, 18]:
        snippets.append(f"def f(n):\n    return 1 if n <= 1 else n * f(n-1)\nprint(f({n}))")

    # Fibonacci with memo (DP top-down) (9 variants)
    for n in [15, 20, 18, 22, 25, 16, 23, 19, 21]:
        snippets.append(
            f"memo = {{}}\ndef fib(n):\n    if n in memo: return memo[n]\n"
            f"    if n < 2: return n\n    memo[n] = fib(n-1) + fib(n-2)\n"
            f"    return memo[n]\nprint(fib({n}))")

    # DP bottom-up: climbing stairs (9 variants)
    for n in [15, 20, 12, 18, 25, 22, 16, 14, 23]:
        snippets.append(
            f"dp = [0] * ({n}+1)\ndp[0] = dp[1] = 1\n"
            f"for i in range(2, {n}+1):\n    dp[i] = dp[i-1] + dp[i-2]\nprint(dp[{n}])")

    # Triple nested loop: matrix-like accumulation (9 variants)
    for n in [5, 6, 7, 4, 8, 9, 10, 3, 11]:
        snippets.append(
            f"s = 0\nfor i in range({n}):\n    for j in range(i, {n}):\n"
            f"        for k in range(j, {n}):\n            s += 1\nprint(s)")

    # DP: coin change (min coins) (9 variants)
    for amount, coins in [(11, [1,5,6]), (15, [1,3,7]), (23, [1,5,10]), (19, [1,4,6]),
                          (27, [1,5,11]), (31, [1,7,10]), (17, [1,3,5]), (29, [1,6,13]),
                          (22, [1,4,9])]:
        snippets.append(
            f"coins = {coins}\namount = {amount}\n"
            f"dp = [float('inf')] * (amount+1)\ndp[0] = 0\n"
            f"for i in range(1, amount+1):\n    for c in coins:\n"
            f"        if c <= i and dp[i-c]+1 < dp[i]:\n            dp[i] = dp[i-c]+1\n"
            f"print(dp[amount])")

    # Accumulator with conditional: collatz-like sequence length (9 variants)
    for start in [27, 19, 31, 42, 15, 25, 37, 47, 63]:
        snippets.append(
            f"n = {start}\nsteps = 0\nwhile n != 1:\n"
            f"    n = n // 2 if n % 2 == 0 else 3 * n + 1\n    steps += 1\nprint(steps)")

    # Double loop: build and sum a 2D table (9 variants)
    for r, c in [(6, 8), (7, 5), (5, 9), (8, 6), (4, 10), (9, 4), (3, 12), (10, 3), (6, 7)]:
        snippets.append(
            f"t = [[0]*{c} for _ in range({r})]\n"
            f"for i in range({r}):\n    for j in range({c}):\n"
            f"        t[i][j] = i * j + i + j\n"
            f"s = sum(t[i][j] for i in range({r}) for j in range({c}))\nprint(s)")

    # DP: longest increasing subsequence length (9 variants)
    for arr in [[3,1,4,1,5,9,2,6], [10,9,2,5,3,7,101,18], [5,1,4,2,3,8,6,7],
                [7,2,8,1,3,5,9,4], [1,3,6,2,8,4,5,7], [4,10,4,3,8,9],
                [2,6,3,4,1,7], [9,1,3,7,2,8,4], [6,3,5,10,2,8,1]]:
        snippets.append(
            f"a = {arr}\nn = len(a)\ndp = [1]*n\n"
            f"for i in range(1, n):\n    for j in range(i):\n"
            f"        if a[j] < a[i] and dp[j]+1 > dp[i]:\n            dp[i] = dp[j]+1\n"
            f"print(max(dp))")

    return snippets

HARD_SNIPPETS = _gen_hard_snippets()


def gen_single_easy(start_id):
    rng = _rng(210)
    pool = list(EASY_SNIPPETS)
    rng.shuffle(pool)
    out = []
    for i, code in enumerate(pool[:N_TOTAL]):
        ans = _run(code)
        instr = f"What is the exact output of this Python code?\n```\n{code}\n```\nAnswer with only the output, nothing else."
        out.append(task_single(start_id + i, "easy", instr, ans))
    return out


def gen_single_medium(start_id):
    rng = _rng(320)
    pool = list(MEDIUM_SNIPPETS)
    rng.shuffle(pool)
    out = []
    for i, code in enumerate(pool[:N_TOTAL]):
        ans = _run(code)
        instr = f"What is the exact output of this Python code?\n```\n{code}\n```\nAnswer with only the output, nothing else."
        out.append(task_single(start_id + i, "medium", instr, ans))
    return out


def gen_single_hard(start_id):
    rng = _rng(430)
    pool = list(HARD_SNIPPETS)
    rng.shuffle(pool)
    out = []
    for i, code in enumerate(pool[:N_TOTAL]):
        ans = _run(code)
        instr = f"What is the exact output of this Python code?\n```\n{code}\n```\nAnswer with only the output, nothing else."
        out.append(task_single(start_id + i, "hard", instr, ans))
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
        "code_executor_single_easy": gen_single_easy(4700),
        "code_executor_single_medium": gen_single_medium(4800),
        "code_executor_single_hard": gen_single_hard(4900),
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
