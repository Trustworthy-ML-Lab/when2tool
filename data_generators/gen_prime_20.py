"""Generate single-step Prime tasks (easy / medium / hard)."""
import json, re, math, random
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "data" / "task_sets"
OUT.mkdir(parents=True, exist_ok=True)
N_TOTAL = 70; N_SPLIT = 20; SEED = 20260304

def _rng(offset=0): return random.Random(SEED + offset)

def prime_env():
    return [{"name": "PrimeEnv", "tools": ["is_prime", "nth_prime", "factorize"], "parameters": {}}]

def _is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i+2) == 0: return False
        i += 6
    return True

def _nth_prime(n):
    # Sieve-based: fast enough for n up to ~1000
    if n < 1: return 2
    limit = max(100, n * 12)  # upper bound estimate
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i, v in enumerate(sieve) if v]
    if n <= len(primes):
        return primes[n - 1]
    # Fallback for edge case
    count, c = len(primes), primes[-1]
    while count < n:
        c += 1
        if _is_prime(c): count += 1
    return c

def _factorize(n):
    factors = []; d = 2; t = n
    while d * d <= t:
        e = 0
        while t % d == 0: e += 1; t //= d
        if e > 0: factors.append((d, e))
        d += 1
    if t > 1: factors.append((t, 1))
    return factors

def _factors_str(factors):
    parts = []
    for p, e in factors:
        if e == 1: parts.append(str(p))
        else: parts.append(f"{p}^{e}")
    return " * ".join(parts)

def task(tid, diff, instr, ans):
    return {"id": tid, "difficulty": diff, "multi_step": False, "instruction": instr,
            "environments": prime_env(), "expected": {"answer": str(ans)}, "tags": ["prime", "single_round"]}

def gen_single_easy(start_id):
    rng = _rng(150)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = rng.randint(0, 3)
        if p == 0:
            n = rng.randint(2, 99)
            ans = "True" if _is_prime(n) else "False"
            instr = f"Is {n} a prime number? Answer True or False."
        elif p == 1:
            idx = rng.randint(1, 18)
            ans = str(_nth_prime(idx))
            instr = f"What is the {idx}th prime number? Answer with only the number."
        elif p == 2:
            n = rng.randint(4, 99)
            while _is_prime(n): n = rng.randint(4, 100)
            ans = _factors_str(_factorize(n))
            instr = f"What is the prime factorization of {n}? Write as 'p1^e1 * p2^e2' (use p^1 as just p). Answer with only the factorization."
        else:
            n = rng.randint(2, 80)
            ans = "True" if _is_prime(n) else "False"
            instr = f"Is {n} prime? Answer True or False."
        key = instr
        if key in seen: continue
        seen.add(key)
        out.append(task(start_id + len(out), "easy", instr, ans))
    return out

def gen_single_medium(start_id):
    rng = _rng(260)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = rng.randint(0, 3)
        if p == 0:
            n = rng.randint(100, 999)
            ans = "True" if _is_prime(n) else "False"
            instr = f"Is {n} a prime number? Answer True or False."
        elif p == 1:
            idx = rng.randint(20, 80)
            ans = str(_nth_prime(idx))
            instr = f"What is the {idx}th prime number? Answer with only the number."
        elif p == 2:
            n = rng.randint(100, 500)
            while _is_prime(n): n = rng.randint(100, 500)
            ans = _factors_str(_factorize(n))
            instr = f"What is the prime factorization of {n}? Write as 'p1^e1 * p2^e2' (use p^1 as just p). Answer with only the factorization."
        else:
            n = rng.randint(200, 997)
            ans = "True" if _is_prime(n) else "False"
            instr = f"Is {n} prime? Answer True or False."
        key = instr
        if key in seen: continue
        seen.add(key)
        out.append(task(start_id + len(out), "medium", instr, ans))
    return out

def gen_single_hard(start_id):
    rng = _rng(370)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = rng.randint(0, 3)
        if p == 0:
            n = rng.randint(10000, 99999)
            ans = "True" if _is_prime(n) else "False"
            instr = f"Is {n} a prime number? Answer True or False."
        elif p == 1:
            idx = rng.randint(200, 500)
            ans = str(_nth_prime(idx))
            instr = f"What is the {idx}th prime number? Answer with only the number."
        elif p == 2:
            n = rng.randint(1000, 9999)
            while _is_prime(n): n = rng.randint(1000, 9999)
            ans = _factors_str(_factorize(n))
            instr = f"What is the prime factorization of {n}? Write as 'p1^e1 * p2^e2' (use p^1 as just p). Answer with only the factorization."
        else:
            n = rng.randint(100000, 999999)
            ans = "True" if _is_prime(n) else "False"
            instr = f"Is {n} prime? Answer True or False."
        key = instr
        if key in seen: continue
        seen.add(key)
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
    gens = {"prime_single_easy": gen_single_easy(2900), "prime_single_medium": gen_single_medium(3000),
            "prime_single_hard": gen_single_hard(3100)}
    for name, tasks in gens.items():
        validate(tasks, name)
        train, test = split_train_test(tasks)
        for sn, st in [("train", train), ("test", test)]:
            fname = f"{name}_{sn}.json"
            (OUT / fname).write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n")
            print(f"wrote {fname}: {len(st)}")

if __name__ == "__main__": main()
