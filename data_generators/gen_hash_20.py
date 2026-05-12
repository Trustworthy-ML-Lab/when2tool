"""Generate single-step Hash tasks (easy / medium / hard)."""
import hashlib
import json, re
import random
import string
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "data" / "task_sets"
OUT.mkdir(parents=True, exist_ok=True)
N_TOTAL = 70
N_SPLIT = 20
SEED = 20260304

# Custom hash algorithms -- must match hash_env.py exactly
_CUSTOM_HASHES = {
    "fnv1a_custom": {
        "offset": 0xCBF29CE484222325 ^ 0xDEADBEEF42,
        "prime": 0x100000001B3 + 38,
    },
    "djb2_custom": {
        "offset": 5381 ^ 0xABCD1234,
        "prime": 33 + 14,
    },
    "sdbm_custom": {
        "offset": 0x1F2E3D4C5B6A7980,
        "prime": 65599 + 42,
    },
    "murmur_custom": {
        "offset": 0x5F3759DF ^ 0xCAFEBABE,
        "prime": 0x100000001B3 + 97,
    },
    "jenkins_custom": {
        "offset": 0x7A6B5C4D3E2F1001,
        "prime": 6364136223846793005 + 13,
    },
}
_CUSTOM_MOD = 2**64


def _custom_hash(s: str, algo_name: str) -> str:
    cfg = _CUSTOM_HASHES[algo_name]
    h = cfg["offset"]
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * cfg["prime"]) % _CUSTOM_MOD
    return format(h, "016x")


def _rng(offset=0):
    return random.Random(SEED + offset)


def hash_env():
    return [{"name": "HashEnv", "tools": ["compute_hash"], "parameters": {}}]


def task_single(tid, difficulty, instruction, answer):
    return {
        "id": tid, "difficulty": difficulty, "multi_step": False,
        "instruction": instruction, "environments": hash_env(),
        "expected": {"answer": str(answer)}, "tags": ["hash", "single_round"],
    }


# Easy: MD5 / SHA1 of very short well-known strings (model might have memorized)
EASY_INPUTS = [
    # MD5
    ("", "md5"), ("hello", "md5"), ("test", "md5"), ("abc", "md5"),
    ("password", "md5"), ("123456", "md5"), ("admin", "md5"), ("root", "md5"),
    ("foo", "md5"), ("bar", "md5"), ("world", "md5"), ("1234", "md5"),
    ("a", "md5"), ("b", "md5"), ("0", "md5"), ("name", "md5"),
    ("login", "md5"), ("user", "md5"), ("key", "md5"), ("data", "md5"),
    ("sun", "md5"), ("dog", "md5"), ("cat", "md5"), ("red", "md5"),
    ("one", "md5"), ("two", "md5"), ("yes", "md5"), ("no", "md5"),
    ("ok", "md5"), ("hi", "md5"), ("go", "md5"), ("up", "md5"),
    ("run", "md5"), ("fly", "md5"), ("top", "md5"), ("big", "md5"),
    ("new", "md5"), ("old", "md5"), ("hot", "md5"), ("ice", "md5"),
    # SHA1
    ("", "sha1"), ("hello", "sha1"), ("test", "sha1"), ("abc", "sha1"),
    ("foo", "sha1"), ("bar", "sha1"), ("world", "sha1"), ("1234", "sha1"),
    ("password", "sha1"), ("admin", "sha1"), ("root", "sha1"), ("login", "sha1"),
    ("sun", "sha1"), ("dog", "sha1"), ("cat", "sha1"), ("red", "sha1"),
    ("a", "sha1"), ("b", "sha1"), ("0", "sha1"), ("name", "sha1"),
    ("key", "sha1"), ("data", "sha1"), ("user", "sha1"), ("code", "sha1"),
    ("run", "sha1"), ("fly", "sha1"), ("top", "sha1"), ("big", "sha1"),
    ("new", "sha1"), ("old", "sha1"), ("hot", "sha1"), ("ice", "sha1"),
]

# Medium: SHA256 / SHA1 / MD5 of short phrases (less likely memorized)
MEDIUM_INPUTS = [
    # SHA256
    ("hello world", "sha256"), ("openai", "sha256"), ("machine learning", "sha256"),
    ("deep learning", "sha256"), ("python3", "sha256"), ("tensorflow", "sha256"),
    ("transformer", "sha256"), ("attention", "sha256"), ("neural network", "sha256"),
    ("gradient", "sha256"), ("backprop", "sha256"), ("optimizer", "sha256"),
    ("dataset", "sha256"), ("benchmark", "sha256"), ("evaluation", "sha256"),
    ("inference", "sha256"), ("tokenizer", "sha256"), ("embedding", "sha256"),
    ("fine-tuning", "sha256"), ("pre-training", "sha256"),
    ("reinforcement", "sha256"), ("supervised", "sha256"),
    ("unsupervised", "sha256"), ("generative", "sha256"),
    # SHA1
    ("hello world", "sha1"), ("machine learning", "sha1"), ("deep learning", "sha1"),
    ("transformer", "sha1"), ("neural network", "sha1"), ("gradient descent", "sha1"),
    ("batch size", "sha1"), ("epoch count", "sha1"), ("validation set", "sha1"),
    ("test split", "sha1"), ("model checkpoint", "sha1"), ("loss function", "sha1"),
    ("activation relu", "sha1"), ("pooling layer", "sha1"), ("skip connection", "sha1"),
    ("self attention", "sha1"), ("layer norm", "sha1"), ("positional encoding", "sha1"),
    ("beam search", "sha1"), ("greedy decode", "sha1"), ("temperature scaling", "sha1"),
    ("top-k sampling", "sha1"), ("nucleus sampling", "sha1"), ("prompt tuning", "sha1"),
    # MD5
    ("hello world", "md5"), ("machine learning", "md5"), ("deep learning", "md5"),
    ("transformer model", "md5"), ("neural network", "md5"), ("gradient descent", "md5"),
    ("batch size", "md5"), ("epoch count", "md5"), ("validation set", "md5"),
    ("test split", "md5"), ("model checkpoint", "md5"), ("loss function", "md5"),
    ("activation relu", "md5"), ("pooling layer", "md5"), ("skip connection", "md5"),
    ("self attention", "md5"), ("layer norm", "md5"), ("positional encoding", "md5"),
    ("beam search", "md5"), ("greedy decode", "md5"), ("temperature scaling", "md5"),
    ("top-k sampling", "md5"), ("nucleus sampling", "md5"), ("prompt tuning", "md5"),
]


def gen_single_easy(start_id):
    rng = _rng(180)
    pool = list(EASY_INPUTS)
    rng.shuffle(pool)
    out = []
    for i, (data, algo) in enumerate(pool[:N_TOTAL]):
        h = hashlib.new(algo)
        h.update(data.encode("utf-8"))
        ans = h.hexdigest()
        display_data = f"'{data}'" if data else "the empty string ''"
        instr = f"What is the {algo.upper()} hash of {display_data}? Answer with only the hex string."
        out.append(task_single(start_id + i, "easy", instr, ans))
    return out


def gen_single_medium(start_id):
    rng = _rng(290)
    pool = list(MEDIUM_INPUTS)
    rng.shuffle(pool)
    out = []
    for i, (data, algo) in enumerate(pool[:N_TOTAL]):
        h = hashlib.new(algo)
        h.update(data.encode("utf-8"))
        ans = h.hexdigest()
        instr = f"What is the {algo.upper()} hash of '{data}'? Answer with only the hex string."
        out.append(task_single(start_id + i, "medium", instr, ans))
    return out


# Hard: 5 custom algorithms -- impossible for any model to know
CUSTOM_ALGO_NAMES = list(_CUSTOM_HASHES.keys())


def gen_single_hard(start_id):
    rng = _rng(400)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        algo = CUSTOM_ALGO_NAMES[len(out) % len(CUSTOM_ALGO_NAMES)]
        length = rng.randint(3, 15)
        data = "".join(rng.choices(string.ascii_letters + string.digits, k=length))
        key = (algo, data)
        if key in seen:
            continue
        seen.add(key)
        ans = _custom_hash(data, algo)
        algo_upper = algo.upper()
        instr = (f"What is the {algo_upper} hash of '{data}'? "
                 f"This is a custom hash algorithm available via the compute_hash tool. "
                 f"Answer with only the hex string.")
        out.append(task_single(start_id + len(out), "hard", instr, ans))
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
        "hash_single_easy": gen_single_easy(3800),
        "hash_single_medium": gen_single_medium(3900),
        "hash_single_hard": gen_single_hard(4000),
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
