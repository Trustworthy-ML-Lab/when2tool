"""Generate single-step Decoding tasks (easy / medium / hard)."""
import json, re
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "data" / "task_sets"
OUT.mkdir(parents=True, exist_ok=True)
N_TOTAL = 70
N_SPLIT = 20
SEED = 20260304

# --------------- Replicate cipher helpers from decoding_env.py ---------------

MORSE_TO_CHAR = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
}
CHAR_TO_MORSE = {v: k for k, v in MORSE_TO_CHAR.items()}

CUSTOM_ENCODINGS = {
    "alpha7": {chr(i): chr(((i - 65 + 7) % 26) + 65) for i in range(65, 91)},
    "reverse": {chr(i): chr(90 - (i - 65)) for i in range(65, 91)},
    "scramble1": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "QWERTYUIOPASDFGHJKLZXCVBNM")),
    "scramble2": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "MXBFVZNLHTKGDWRJYPAQOIUCES")),
}
CUSTOM_DECODINGS = {name: {v: k for k, v in mapping.items()} for name, mapping in CUSTOM_ENCODINGS.items()}


def _morse_encode(msg):
    words = msg.upper().split(" ")
    encoded = []
    for word in words:
        encoded.append(" ".join(CHAR_TO_MORSE.get(c, "?") for c in word))
    return " / ".join(encoded)


def _morse_decode(msg):
    words = msg.strip().split(" / ")
    decoded = []
    for word in words:
        letters = word.strip().split(" ")
        decoded.append("".join(MORSE_TO_CHAR.get(l, "?") for l in letters))
    return " ".join(decoded)


def _caesar_encode(msg, shift):
    out = []
    for c in msg:
        if c.isalpha():
            base = 65 if c.isupper() else 97
            out.append(chr((ord(c) - base + shift) % 26 + base))
        else:
            out.append(c)
    return "".join(out)


def _caesar_decode(msg, shift):
    out = []
    for c in msg:
        if c.isalpha():
            base = 65 if c.isupper() else 97
            out.append(chr((ord(c) - base - shift) % 26 + base))
        else:
            out.append(c)
    return "".join(out)


def _custom_encode(msg, name):
    mapping = CUSTOM_ENCODINGS[name]
    return "".join(mapping.get(c.upper(), c) for c in msg.upper())


def _custom_decode(msg, name):
    mapping = CUSTOM_DECODINGS[name]
    return "".join(mapping.get(c.upper(), c) for c in msg.upper())


# --------------- Generator ---------------

def _rng(offset=0):
    return random.Random(SEED + offset)


def decoding_env():
    return [{"name": "DecodingEnv", "tools": ["encode", "decode"], "parameters": {}}]


def task_single(tid, difficulty, instruction, answer):
    return {
        "id": tid, "difficulty": difficulty, "multi_step": False,
        "instruction": instruction, "environments": decoding_env(),
        "expected": {"answer": str(answer)}, "tags": ["decoding", "single_round"],
    }


# Easy: Morse code for short words
EASY_WORDS = [
    "SOS", "HELLO", "WORLD", "HI", "OK", "YES", "NO", "HELP",
    "CAT", "DOG", "SUN", "MOON", "STAR", "FIRE", "ICE", "RUN",
    "GO", "STOP", "OPEN", "LOVE", "HOME", "TREE", "FISH", "BIRD",
    "RED", "BLUE", "RAIN", "WIND", "EAST", "WEST", "LAND", "DEEP",
    "FAST", "SLOW", "COOL", "WARM", "GOLD", "DARK", "LION", "BEAR",
    "ROCK", "LAKE", "HILL", "ROSE", "SALT", "SAND", "WAVE", "SNOW",
    "JUMP", "WALK", "SING", "BOOK", "DOOR", "RING", "BELL", "DRUM",
    "COIN", "DICE", "CARD", "FLAG", "MAPS", "CODE", "BONE", "FORK",
    "FROG", "DUCK", "HAWK", "WOLF", "SEAL", "CRAB", "DEER", "GOAT",
    "LAMP", "MASK", "NAIL", "ROPE", "SHIP", "TANK", "VINE", "YARN",
]

MEDIUM_WORDS = [
    "PYTHON", "CIPHER", "DECODE", "SECRET", "HIDDEN", "PUZZLE",
    "MATRIX", "KERNEL", "VECTOR", "LAMBDA", "SIGNAL", "BINARY",
    "QUANTUM", "NEURAL", "TENSOR", "BUFFER", "STREAM", "MODULE",
    "APACHE", "SERVER", "CLIENT", "ROUTER", "SWITCH", "BRIDGE",
    "ROCKET", "PLANET", "GALAXY", "NEBULA", "PRISM", "LASER",
    "ANCHOR", "BEACON", "FALCON", "HYBRID", "JAVELIN", "KNIGHT",
    "MAGNET", "OCTANE", "PRONTO", "QUARTZ", "RIDDLE", "SPHINX",
    "TUNNEL", "UPLINK", "VERTEX", "ZEPHYR", "COBALT", "FLARES",
    "GOTHIC", "HARMON", "IMPACT", "JOKERS", "KALEID", "LOCKET",
    "MORTAR", "NIMBLE", "OPAQUE", "PARCEL", "QUIVER", "RASCAL",
    "TUNDRA", "VELVET", "WALRUS", "XYLOSE", "YONDER", "ZODIAC",
    "BLAZER", "CANOPY", "DYNAMO", "EMPIRE", "FATHOM", "GRAVEL",
    "HELIUM", "ISLAND", "JIGSAW", "KIMONO", "LUSTER", "MEADOW",
]

# Hard: custom ciphers (model has never seen these mappings)
HARD_WORDS = [
    "HELLO", "WORLD", "PYTHON", "CIPHER", "MACHINE", "LEARNING",
    "COMPUTE", "NETWORK", "QUANTUM", "SIGNAL", "DECODE", "SECRET",
    "ATTACK", "DEFEND", "PUZZLE", "ANSWER", "HIDDEN", "REVEAL",
    "BINARY", "STREAM", "KERNEL", "VECTOR", "TENSOR", "MODULE",
    "APACHE", "SERVER", "CLIENT", "ROUTER", "SWITCH", "BRIDGE",
    "CRYPTO", "SYSTEM", "SECURE", "ACCESS", "PORTAL", "LAUNCH",
    "ROCKET", "ENGINE", "TURBO", "ALPHA", "BRAVO", "DELTA",
    "FALCON", "HYBRID", "IMPACT", "KNIGHT", "MAGNET", "OCTANE",
    "PRONTO", "QUARTZ", "RIDDLE", "SPHINX", "TUNNEL", "UPLINK",
    "VERTEX", "ZEPHYR", "COBALT", "ANCHOR", "BEACON", "BLAZER",
    "CANOPY", "DYNAMO", "EMPIRE", "FATHOM", "GRAVEL", "HELIUM",
    "ISLAND", "JIGSAW", "KIMONO", "LUSTER", "MEADOW", "NIMBLE",
    "OPAQUE", "PARCEL", "QUIVER", "RASCAL", "TUNDRA", "VELVET",
    "WALRUS", "XYLOSE", "YONDER", "ZODIAC",
]


def gen_single_easy(start_id):
    rng = _rng(190)
    out, seen = [], set()
    pool = list(EASY_WORDS)
    rng.shuffle(pool)
    idx = 0
    attempts = 0
    while len(out) < N_TOTAL and attempts < 500:
        attempts += 1
        p = len(out) % 3
        word = pool[idx % len(pool)]
        idx += 1
        if p == 0:
            # Encode to Morse
            ans = _morse_encode(word)
            key = f"morse_enc_{word}"
            if key in seen:
                continue
            seen.add(key)
            instr = (f"Encode '{word}' in Morse code. "
                     f"Answer with only the Morse code (dots and dashes separated by spaces, words by /).")
        elif p == 1:
            # Decode from Morse
            morse = _morse_encode(word)
            ans = word
            key = f"morse_dec_{morse}"
            if key in seen:
                continue
            seen.add(key)
            instr = f"Decode this Morse code: '{morse}'. Answer with only the decoded text in uppercase."
        else:
            # Caesar shift=13 (ROT13)
            ans = _caesar_encode(word, 13)
            key = f"rot13_{word}"
            if key in seen:
                continue
            seen.add(key)
            instr = f"Apply Caesar cipher with shift 13 to '{word}'. Answer with only the result."
        out.append(task_single(start_id + len(out), "easy", instr, ans))
    return out


def gen_single_medium(start_id):
    rng = _rng(300)
    out, seen = [], set()
    pool = list(MEDIUM_WORDS)
    rng.shuffle(pool)
    idx = 0
    attempts = 0
    while len(out) < N_TOTAL and attempts < 500:
        attempts += 1
        p = len(out) % 4
        word = pool[idx % len(pool)]
        idx += 1
        if p == 0:
            shift = rng.randint(3, 15)
            encoded = _caesar_encode(word, shift)
            ans = word
            key = f"caesar_dec_{encoded}_{shift}"
            if key in seen:
                continue
            seen.add(key)
            instr = (f"Decode '{encoded}' using Caesar cipher with shift {shift}. "
                     f"Answer with only the decoded text.")
        elif p == 1:
            # Encode with Morse
            ans = _morse_encode(word)
            key = f"morse_enc_{word}"
            if key in seen:
                continue
            seen.add(key)
            instr = (f"Encode '{word}' in Morse code. "
                     f"Answer with only the Morse code (dots and dashes separated by spaces, words by /).")
        elif p == 2:
            shift = rng.randint(5, 20)
            ans = _caesar_encode(word, shift)
            key = f"caesar_enc_{word}_{shift}"
            if key in seen:
                continue
            seen.add(key)
            instr = (f"Encode '{word}' using Caesar cipher with shift {shift}. "
                     f"Answer with only the result.")
        else:
            # Decode Morse for medium words
            morse = _morse_encode(word)
            ans = word
            key = f"morse_dec_{morse}"
            if key in seen:
                continue
            seen.add(key)
            instr = f"Decode this Morse code: '{morse}'. Answer with only the decoded text in uppercase."
        out.append(task_single(start_id + len(out), "medium", instr, ans))
    return out


HARD_CIPHERS = ["scramble1", "scramble2", "alpha7", "reverse"]


def gen_single_hard(start_id):
    rng = _rng(410)
    out, seen = [], set()
    pool = list(HARD_WORDS)
    rng.shuffle(pool)
    idx = 0
    attempts = 0
    while len(out) < N_TOTAL and attempts < 1000:
        attempts += 1
        p = rng.randint(0, 1)
        word = pool[idx % len(pool)]
        cipher = HARD_CIPHERS[idx % len(HARD_CIPHERS)]
        idx += 1
        if p == 0:
            # Encode with custom cipher, ask to decode
            encoded = _custom_encode(word, cipher)
            ans = word
            key = f"{cipher}_dec_{encoded}"
            if key in seen:
                continue
            seen.add(key)
            instr = (f"Decode '{encoded}' using the custom cipher called '{cipher}'. "
                     f"Use the decode tool with encoding_type='custom' and key='{cipher}'. "
                     f"Answer with only the decoded text.")
        else:
            # Encode with custom cipher
            ans = _custom_encode(word, cipher)
            key = f"{cipher}_enc_{word}"
            if key in seen:
                continue
            seen.add(key)
            instr = (f"Encode '{word}' using the custom cipher called '{cipher}'. "
                     f"Use the encode tool with encoding_type='custom' and key='{cipher}'. "
                     f"Answer with only the encoded text.")
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
        "decoding_single_easy": gen_single_easy(4100),
        "decoding_single_medium": gen_single_medium(4200),
        "decoding_single_hard": gen_single_hard(4300),
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
