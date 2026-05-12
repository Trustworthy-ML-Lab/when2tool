"""Generate multi-hop (3-step chained) Code Executor tasks (Env 18: ChainedCodeExecutorEnv).

Each task requires 3 sequential code executions where the output of code1
feeds into code2, and the output of code2 feeds into code3.
Pattern: run code1 -> get x, run code2(x) -> get y, run code3(y) -> get z. Return z.
"""
import io
import contextlib
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
    return random.Random(SEED + 180)


def code_env():
    return [{"name": "CodeExecutorEnv", "tools": ["run_code"], "parameters": {}}]


def _run(code):
    """Execute code and capture stdout."""
    stdout = io.StringIO()
    globs = {"__builtins__": __builtins__}
    with contextlib.redirect_stdout(stdout):
        exec(code, globs, globs)
    return stdout.getvalue().rstrip("\n")


def _normalize_inst(x):
    return re.sub(r"\s+", " ", x.strip().lower())


def _make_task(task_id, difficulty, code1, code2_template, code3_template,
               code2_desc, code3_desc):
    """Build a 3-step chained code executor task.

    code1: literal code string (no dependencies)
    code2_template: code string with '{x}' placeholder for output of code1
    code3_template: code string with '{y}' placeholder for output of code2
    code2_desc / code3_desc: human-readable description of what code2/code3 do
    """
    x = _run(code1)
    code2 = code2_template.replace("{x}", x)
    y = _run(code2)
    code3 = code3_template.replace("{y}", y)
    z = _run(code3)

    instruction = (
        f"Three-step code execution: "
        f"Step 1: Run the following code. Call the result x.\n```\n{code1}\n```\n"
        f"Step 2: Using x, {code2_desc}. Call the result y.\n"
        f"Step 3: Using y, {code3_desc}. Call the result z.\n"
        f"Return only z."
    )

    return {
        "id": task_id,
        "difficulty": difficulty,
        "multi_step": True,
        "instruction": instruction,
        "environments": code_env(),
        "expected": {
            "answer": str(z),
            "steps": [str(x), str(y), str(z)],
        },
        "tags": ["code_executor", "multihop_3step"],
    }


# =========================================================================
# EASY: simple arithmetic chains
# =========================================================================
def gen_easy(start_id):
    """Easy: simple arithmetic (print(a+b) -> x, print(x*c) -> y, print(y-d) -> z)."""
    rng = _rng()
    out = []
    seen = set()

    while len(out) < N_TOTAL:
        p = len(out) % 7
        if p == 0:
            a, b = rng.randint(2, 20), rng.randint(2, 20)
            c = rng.randint(2, 8)
            d = rng.randint(1, 15)
            code1 = f"print({a} + {b})"
            code2_t = "print({x} * " + str(c) + ")"
            code3_t = "print({y} - " + str(d) + ")"
            desc2 = f"run: print(x * {c})"
            desc3 = f"run: print(y - {d})"
        elif p == 1:
            a, b = rng.randint(3, 12), rng.randint(3, 12)
            c = rng.randint(2, 10)
            d = rng.randint(5, 25)
            code1 = f"print({a} * {b})"
            code2_t = "print({x} + " + str(c) + ")"
            code3_t = "print({y} * " + str(d) + ")"
            desc2 = f"run: print(x + {c})"
            desc3 = f"run: print(y * {d})"
        elif p == 2:
            a = rng.randint(20, 40)
            b = rng.randint(2, 10)
            c = rng.randint(2, 6)
            d = rng.randint(2, 8)
            code1 = f"print({a} - {b})"
            code2_t = "print({x} * " + str(c) + ")"
            code3_t = "print({y} + " + str(d) + ")"
            desc2 = f"run: print(x * {c})"
            desc3 = f"run: print(y + {d})"
        elif p == 3:
            a, b = rng.randint(2, 15), rng.randint(2, 15)
            c = rng.randint(3, 10)
            d = rng.randint(2, 5)
            code1 = f"print({a} + {b})"
            code2_t = "print({x} + " + str(c) + ")"
            code3_t = "print({y} * " + str(d) + ")"
            desc2 = f"run: print(x + {c})"
            desc3 = f"run: print(y * {d})"
        elif p == 4:
            a = rng.randint(5, 25)
            b = rng.randint(2, 10)
            c = rng.randint(2, 8)
            d = rng.randint(1, 20)
            code1 = f"print({a} * {b})"
            code2_t = "print({x} - " + str(c) + ")"
            code3_t = "print({y} + " + str(d) + ")"
            desc2 = f"run: print(x - {c})"
            desc3 = f"run: print(y + {d})"
        elif p == 5:
            a, b = rng.randint(2, 10), rng.randint(2, 10)
            c = rng.randint(2, 5)
            d = rng.randint(2, 6)
            code1 = f"print({a} ** {b})"
            code2_t = "print({x} % " + str(c * 10 + rng.randint(1, 9)) + ")"
            code3_t = "print({y} * " + str(d) + ")"
            # Recompute desc after fixing the modulo value
            x_val = _run(code1)
            mod_val = int(code2_t.split("% ")[1].rstrip(")"))
            desc2 = f"run: print(x % {mod_val})"
            desc3 = f"run: print(y * {d})"
        else:
            a = rng.randint(10, 30)
            b = rng.randint(3, 10)
            c = rng.randint(2, 8)
            d = rng.randint(1, 10)
            code1 = f"print({a} + {b})"
            code2_t = "print({x} * " + str(c) + ")"
            code3_t = "print({y} - " + str(d) + ")"
            desc2 = f"run: print(x * {c})"
            desc3 = f"run: print(y - {d})"

        key = (code1, code2_t, code3_t)
        if key in seen:
            continue
        seen.add(key)
        out.append(_make_task(start_id + len(out), "easy", code1, code2_t, code3_t,
                              desc2, desc3))

    return out[:N_TOTAL]


# =========================================================================
# MEDIUM: list/string operations
# =========================================================================
def _gen_medium_chain(rng, pattern):
    """Generate a medium-difficulty 3-step chain based on a pattern index.

    Every step involves list/string/dict operations requiring mental tracking
    (sorting, filtering, comprehensions, digit manipulation, dict access, etc.)
    but no DP or recursion.
    """
    if pattern == 0:
        # Step 1: Sort a list with a random element inserted, take the median
        # Step 2: List comprehension filtering multiples up to x
        # Step 3: Sum of digits of y
        base = sorted(rng.sample(range(1, 30), 5))
        insert_val = rng.randint(1, 25)
        elems = base + [insert_val]
        rng.shuffle(elems)
        mid_idx = len(elems) // 2
        code1 = f"print(sorted({elems})[{mid_idx}])"
        div = rng.choice([2, 3, 5])
        code2_t = f"print(len([i for i in range({{x}}) if i % {div} == 0]))"
        code3_t = "print(sum(int(d) for d in str({y})))"
        desc2 = f"run: count how many integers in range(x) are divisible by {div}"
        desc3 = "run: compute the sum of digits of y"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 1:
        # Step 1: Count vowels in a phrase
        # Step 2: Build dict comprehension {i: i**2 for i in range(x)}, access key x-1
        # Step 3: Reverse the string representation and convert back to int
        phrases = ["hello world", "python programming", "machine learning",
                   "data science", "artificial intelligence", "deep networks",
                   "natural language", "computer vision", "gradient descent",
                   "random forest", "neural network", "support vector"]
        phrase = rng.choice(phrases)
        code1 = f"print(sum(1 for c in '{phrase}' if c in 'aeiou'))"
        code2_t = "print({i: i**2 for i in range({x})}[{x} - 1])"
        code3_t = "print(int(''.join(reversed(str({y})))))"
        desc2 = "run: build dict {i: i**2 for i in range(x)} and access key x-1"
        desc3 = "run: reverse the string of y and convert back to int"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 2:
        # Step 1: Count unique characters in a word
        # Step 2: Generate odd numbers up to x and sum them
        # Step 3: Count how many digits of y are greater than 3
        words = ["mississippi", "abracadabra", "programming", "concatenation",
                 "enumeration", "permutation", "combination", "distribution",
                 "elimination", "substitution", "acceleration", "optimization"]
        w = rng.choice(words)
        code1 = f"print(len(set('{w}')))"
        code2_t = "print(sum(range(1, {x} + 1, 2)))"
        code3_t = "print(sum(1 for d in str({y}) if int(d) > 3))"
        desc2 = "run: sum all odd numbers from 1 to x (inclusive)"
        desc3 = "run: count how many digits of y are greater than 3"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 3:
        # Step 1: Sort a list and take the second-largest element
        # Step 2: Build list of squares up to x, filter those > x, count them
        # Step 3: Sum of digits of y squared
        n = rng.randint(5, 9)
        elems = [rng.randint(1, 80) for _ in range(n)]
        code1 = f"print(sorted({elems})[-2])"
        code2_t = "print(len([i**2 for i in range(1, {x}) if i**2 > {x}]))"
        code3_t = "print(sum(int(d) for d in str({y} ** 2)))"
        desc2 = "run: count how many i in range(1, x) have i**2 > x"
        desc3 = "run: compute y squared, then sum its digits"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 4:
        # Step 1: String length of a word
        # Step 2: Sorted list with x inserted, take the middle element
        # Step 3: Dict comprehension mapping digits of y to their count
        words = ["algorithm", "mathematics", "complexity", "transformer",
                 "backpropagation", "optimization", "architecture", "hypothesis",
                 "convergence", "regularization", "augmentation", "classification"]
        w = rng.choice(words)
        base_list = sorted(rng.sample(range(1, 20), 5))
        code1 = f"print(len('{w}'))"
        elems_str = str(base_list)
        code2_t = f"print(sorted({elems_str} + [{{x}}])[3])"
        code3_t = "print(max(str({y}).count(d) for d in set(str({y}))))"
        desc2 = f"run: insert x into {base_list}, sort, and take the element at index 3"
        desc3 = "run: find the most frequent digit in y (return its count)"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 5:
        # Step 1: List comprehension filtering primes-ish (multiples of divisor)
        # Step 2: Zip two ranges and sum products
        # Step 3: Count unique digits
        limit = rng.randint(20, 45)
        divisor = rng.choice([3, 5, 7])
        code1 = f"print(len([x for x in range(1, {limit + 1}) if x % {divisor} == 0]))"
        n2 = rng.randint(3, 6)
        code2_t = f"print(sum(a * b for a, b in zip(range({{x}}), range({n2}, {{x}} + {n2}))))"
        code3_t = "print(len(set(str({y}))))"
        desc2 = f"run: zip range(x) with range({n2}, x+{n2}), sum element-wise products"
        desc3 = "run: count the number of unique digits in y"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 6:
        # Step 1: Sum of a random list
        # Step 2: Sum of digits of x
        # Step 3: Build list of cubes up to y, take the last one
        n = rng.randint(4, 7)
        elems = [rng.randint(5, 30) for _ in range(n)]
        code1 = f"print(sum({elems}))"
        code2_t = "print(sum(int(d) for d in str({x})))"
        code3_t = "print([i**3 for i in range(1, {y} + 1)][-1])"
        desc2 = "run: sum the digits of x"
        desc3 = "run: build list of cubes [1^3, 2^3, ..., y^3] and take the last element"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 7:
        # Step 1: Max of list minus min of list (range)
        # Step 2: List comprehension: squares of even numbers up to x
        # Step 3: Reverse-sort a derived list and pick element at index 1
        n = rng.randint(5, 9)
        elems = [rng.randint(1, 60) for _ in range(n)]
        code1 = f"print(max({elems}) - min({elems}))"
        code2_t = "print(sum(i**2 for i in range(2, {x} + 1, 2)))"
        code3_t = "print(sorted(str({y}), reverse=True)[0])"
        desc2 = "run: sum of squares of even numbers from 2 to x"
        desc3 = "run: sort the digits of y in descending order and take the largest digit"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 8:
        # Step 1: Reverse a string and get its length
        # Step 2: Filter list for values where index < x
        # Step 3: Dict mapping each char of str(y) to ordinal, sum values
        words = ["transformer", "backpropagation", "convolutional", "generative",
                 "discriminator", "reinforcement", "architecture", "quantization"]
        w = rng.choice(words)
        code1 = f"print(len(''.join(reversed('{w}'))))"
        vals = [rng.randint(2, 15) for _ in range(12)]
        code2_t = f"print(sum({vals}[:{{x}}]))"
        code3_t = "print(sum(ord(c) - ord('0') for c in str({y})))"
        desc2 = f"run: take first x elements of {vals} and sum them"
        desc3 = "run: for each digit char in str(y), compute ord(c)-ord('0'), then sum"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 9:
        # Step 1: Product of list elements via eval-style
        # Step 2: Integer converted to binary string, count '1's
        # Step 3: Build list of factorials up to y, sum them
        n = rng.randint(3, 5)
        elems = [rng.randint(2, 8) for _ in range(n)]
        product_expr = " * ".join(str(e) for e in elems)
        code1 = f"print({product_expr})"
        code2_t = "print(bin({x}).count('1'))"
        code3_t = (
            "import math\n"
            "print(sum(math.factorial(i) for i in range({y} + 1)))"
        )
        desc2 = "run: convert x to binary and count the number of 1-bits"
        desc3 = "run: compute sum of factorials from 0! to y!"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 10:
        # Step 1: Sum of range(1, n+1)
        # Step 2: String of x -> replace even digits with '0', count remaining non-zero
        # Step 3: Build dict {i: i*y for i in range(y)}, sum all values
        n = rng.randint(8, 25)
        code1 = f"print(sum(range(1, {n + 1})))"
        code2_t = "print(sum(1 for d in str({x}) if int(d) % 2 != 0))"
        code3_t = "print(sum(i * {y} for i in range({y})))"
        desc2 = "run: count the number of odd digits in x"
        desc3 = "run: compute sum of i*y for i in range(y)"
        return code1, code2_t, code3_t, desc2, desc3

    else:
        # Step 1: Min of list
        # Step 2: Enumerate a string of x, sum indices where digit > 2
        # Step 3: Sorted digits of y joined, convert to int
        n = rng.randint(5, 9)
        elems = [rng.randint(3, 40) for _ in range(n)]
        code1 = f"print(min({elems}))"
        code2_t = "print(sum(i for i, d in enumerate(str({x})) if int(d) > 2))"
        code3_t = "print(int(''.join(sorted(str({y})))))"
        desc2 = "run: for each digit in str(x), sum the indices where the digit > 2"
        desc3 = "run: sort the digits of y in ascending order, join, and convert to int"
        return code1, code2_t, code3_t, desc2, desc3


def gen_medium(start_id):
    rng = _rng()
    out = []
    seen = set()
    attempt = 0

    while len(out) < N_TOTAL and attempt < 500:
        pattern = attempt % 12
        attempt += 1
        code1, code2_t, code3_t, desc2, desc3 = _gen_medium_chain(rng, pattern)

        key = (code1, code2_t, code3_t)
        if key in seen:
            continue
        seen.add(key)

        try:
            task = _make_task(start_id + len(out), "medium", code1, code2_t, code3_t,
                              desc2, desc3)
            out.append(task)
        except Exception:
            continue

    return out[:N_TOTAL]


# =========================================================================
# HARD: loops, recursion, DP feeding into next step
# =========================================================================
def _gen_hard_chain(rng, pattern):
    """Generate a hard-difficulty 3-step chain based on a pattern index.

    Every step involves non-trivial computation (loops, recursion, DP,
    list operations) that a model cannot easily do mentally.
    """
    if pattern == 0:
        # Step 1: Factorial(n) via recursion -> x
        # Step 2: Count 1-bits in binary(x) via loop -> y
        # Step 3: Collatz steps from y -> z
        n = rng.choice([8, 9, 10, 11, 12, 13, 14])
        code1 = f"def f(n):\n    return 1 if n <= 1 else n * f(n-1)\nprint(f({n}))"
        code2_t = (
            "x = {x}\n"
            "count = 0\n"
            "while x > 0:\n"
            "    count += x & 1\n"
            "    x >>= 1\n"
            "print(count)"
        )
        code3_t = (
            "n = max({y}, 2)\nsteps = 0\n"
            "while n != 1:\n"
            "    n = n // 2 if n % 2 == 0 else 3 * n + 1\n"
            "    steps += 1\n"
            "print(steps)"
        )
        desc2 = "run: count the number of 1-bits in binary representation of x using a loop"
        desc3 = "run: count Collatz steps from max(y,2) to 1"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 1:
        # Step 1: Sum of squares 1..n via loop -> x
        # Step 2: Count divisors of x via loop -> y
        # Step 3: Fibonacci(y) via DP -> z
        n = rng.randint(10, 25)
        code1 = (
            f"s = 0\nfor i in range(1, {n + 1}):\n"
            f"    s += i * i\nprint(s)"
        )
        code2_t = (
            "n = {x}\n"
            "count = 0\n"
            "for i in range(1, n + 1):\n"
            "    if n % i == 0:\n"
            "        count += 1\n"
            "print(count)"
        )
        code3_t = (
            "n = {y}\n"
            "if n <= 1:\n"
            "    print(n)\n"
            "else:\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    print(b)"
        )
        desc2 = "run: count the number of divisors of x"
        desc3 = "run: compute the y-th Fibonacci number"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 2:
        # Step 1: Collatz sequence length for start -> x
        # Step 2: Digital root of x (repeated digit sum until single digit) -> y
        # Step 3: Compute pow(2, y, mod) using modular exponentiation loop -> z
        start = rng.choice([27, 31, 47, 54, 63, 73, 97, 113])
        mod = rng.choice([97, 127, 131, 149, 157, 163, 173])
        code1 = (
            f"n = {start}\nsteps = 0\n"
            f"while n != 1:\n"
            f"    n = n // 2 if n % 2 == 0 else 3 * n + 1\n"
            f"    steps += 1\nprint(steps)"
        )
        code2_t = (
            "n = {x}\n"
            "while n >= 10:\n"
            "    s = 0\n"
            "    while n > 0:\n"
            "        s += n % 10\n"
            "        n //= 10\n"
            "    n = s\n"
            "print(n)"
        )
        code3_t = (
            f"base, exp, mod = 2, {{y}}, {mod}\n"
            f"result = 1\n"
            f"base = base % mod\n"
            f"e = exp\n"
            f"while e > 0:\n"
            f"    if e % 2 == 1:\n"
            f"        result = (result * base) % mod\n"
            f"    e //= 2\n"
            f"    base = (base * base) % mod\n"
            f"print(result)"
        )
        desc2 = "run: compute digital root of x (repeatedly sum digits until single digit)"
        desc3 = f"run: compute pow(2, y, {mod}) using modular exponentiation loop"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 3:
        # Step 1: Triple nested loop count -> x
        # Step 2: Count distinct prime factors of x via trial division loop -> y
        # Step 3: Compute y-th triangular number's divisor count via loop -> z
        n = rng.choice([6, 7, 8, 9, 10])
        code1 = (
            f"s = 0\nfor i in range({n}):\n"
            f"    for j in range(i, {n}):\n"
            f"        for k in range(j, {n}):\n"
            f"            s += 1\nprint(s)"
        )
        code2_t = (
            "n = {x}\n"
            "count = 0\n"
            "d = 2\n"
            "while d * d <= n:\n"
            "    if n % d == 0:\n"
            "        count += 1\n"
            "        while n % d == 0:\n"
            "            n //= d\n"
            "    d += 1\n"
            "if n > 1:\n"
            "    count += 1\n"
            "print(count)"
        )
        code3_t = (
            "y = {y}\n"
            "tri = y * (y + 1) // 2\n"
            "count = 0\n"
            "for i in range(1, tri + 1):\n"
            "    if tri % i == 0:\n"
            "        count += 1\n"
            "print(count)"
        )
        desc2 = "run: count distinct prime factors of x via trial division"
        desc3 = "run: compute the y-th triangular number T(y)=y*(y+1)/2 then count its divisors"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 4:
        # Step 1: Fibonacci(n) via memoized recursion -> x
        # Step 2: Sum of all proper divisors of x (not including x itself) -> y
        # Step 3: Coin change DP for amount=y -> z
        n = rng.choice([14, 16, 18, 20, 22, 24])
        coins = rng.choice([[1, 3, 5], [1, 5, 7], [1, 4, 6], [1, 3, 7]])
        code1 = (
            f"memo = {{}}\ndef fib(n):\n"
            f"    if n in memo: return memo[n]\n"
            f"    if n < 2: return n\n"
            f"    memo[n] = fib(n-1) + fib(n-2)\n"
            f"    return memo[n]\nprint(fib({n}))"
        )
        code2_t = (
            "n = {x}\n"
            "s = 0\n"
            "for i in range(1, n):\n"
            "    if n % i == 0:\n"
            "        s += i\n"
            "print(s)"
        )
        code3_t = (
            f"coins = {coins}\namount = {{y}}\n"
            f"dp = [float('inf')] * (amount+1)\ndp[0] = 0\n"
            f"for i in range(1, amount+1):\n"
            f"    for c in coins:\n"
            f"        if c <= i and dp[i-c]+1 < dp[i]:\n"
            f"            dp[i] = dp[i-c]+1\n"
            f"print(dp[amount])"
        )
        desc2 = "run: compute sum of all proper divisors of x (excluding x itself)"
        desc3 = f"run: compute min coins for amount=y with coins={coins}"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 5:
        # Step 1: DP climbing stairs (ways to climb n stairs) -> x
        # Step 2: Collatz steps from x -> y
        # Step 3: Count primes up to y via sieve of Eratosthenes -> z
        n = rng.choice([12, 14, 16, 18, 20, 22])
        code1 = (
            f"dp = [0] * ({n}+1)\ndp[0] = dp[1] = 1\n"
            f"for i in range(2, {n}+1):\n    dp[i] = dp[i-1] + dp[i-2]\n"
            f"print(dp[{n}])"
        )
        code2_t = (
            "n = max({x}, 2)\nsteps = 0\n"
            "while n != 1:\n"
            "    n = n // 2 if n % 2 == 0 else 3 * n + 1\n"
            "    steps += 1\n"
            "print(steps)"
        )
        code3_t = (
            "limit = max({y}, 2)\n"
            "sieve = [True] * (limit + 1)\n"
            "sieve[0] = sieve[1] = False\n"
            "for i in range(2, int(limit**0.5) + 1):\n"
            "    if sieve[i]:\n"
            "        for j in range(i*i, limit + 1, i):\n"
            "            sieve[j] = False\n"
            "print(sum(sieve))"
        )
        desc2 = "run: count Collatz steps from x to 1"
        desc3 = "run: count primes up to y using sieve of Eratosthenes"
        return code1, code2_t, code3_t, desc2, desc3

    elif pattern == 6:
        # Step 1: Coin change DP -> x (min coins)
        # Step 2: Generate first x*2 Fibonacci numbers, sum the even ones -> y
        # Step 3: LIS of a seeded array with y injected -> z
        amount = rng.choice([11, 15, 19, 23, 27, 31])
        coins = rng.choice([[1, 5, 6], [1, 3, 7], [1, 5, 10], [1, 4, 6]])
        arr = [rng.randint(1, 50) for _ in range(rng.randint(8, 12))]
        code1 = (
            f"coins = {coins}\namount = {amount}\n"
            f"dp = [float('inf')] * (amount+1)\ndp[0] = 0\n"
            f"for i in range(1, amount+1):\n"
            f"    for c in coins:\n"
            f"        if c <= i and dp[i-c]+1 < dp[i]:\n"
            f"            dp[i] = dp[i-c]+1\n"
            f"print(dp[amount])"
        )
        code2_t = (
            "n = {x} * 2\n"
            "fibs = []\n"
            "a, b = 0, 1\n"
            "for _ in range(n):\n"
            "    fibs.append(a)\n"
            "    a, b = b, a + b\n"
            "print(sum(f for f in fibs if f % 2 == 0))"
        )
        code3_t = (
            f"a = {arr}\n"
            f"a[0] = {{y}} % 50\n"
            f"n = len(a)\n"
            f"dp = [1]*n\n"
            f"for i in range(1, n):\n"
            f"    for j in range(i):\n"
            f"        if a[j] < a[i] and dp[j]+1 > dp[i]:\n"
            f"            dp[i] = dp[j]+1\n"
            f"print(max(dp))"
        )
        desc2 = "run: generate first x*2 Fibonacci numbers and sum the even ones"
        desc3 = f"run: compute LIS of array {arr} with first element replaced by y%50"
        return code1, code2_t, code3_t, desc2, desc3

    else:
        raise ValueError(f"Unknown hard pattern: {pattern}")


def gen_hard(start_id):
    rng = _rng()
    out = []
    seen = set()
    attempt = 0

    while len(out) < N_TOTAL and attempt < 500:
        pattern = attempt % 7
        attempt += 1
        code1, code2_t, code3_t, desc2, desc3 = _gen_hard_chain(rng, pattern)

        key = (code1, code2_t, code3_t)
        if key in seen:
            continue
        seen.add(key)

        try:
            task = _make_task(start_id + len(out), "hard", code1, code2_t, code3_t,
                              desc2, desc3)
            out.append(task)
        except Exception:
            continue

    return out[:N_TOTAL]


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
        "code_executor_multihop_easy": gen_easy(18000),
        "code_executor_multihop_medium": gen_medium(18100),
        "code_executor_multihop_hard": gen_hard(18200),
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
