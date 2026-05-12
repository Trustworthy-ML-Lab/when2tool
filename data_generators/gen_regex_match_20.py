"""Generate single-step RegexMatch tasks (easy / medium / hard)."""
import json, re as _re
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "data" / "task_sets"
OUT.mkdir(parents=True, exist_ok=True)
N_TOTAL = 70; N_SPLIT = 20; SEED = 20260304

def _rng(offset=0): return random.Random(SEED + offset)

def regex_env():
    return [{"name": "RegexMatchEnv", "tools": ["regex_match"], "parameters": {}}]

def task(tid, diff, instr, ans):
    return {"id": tid, "difficulty": diff, "multi_step": False, "instruction": instr,
            "environments": regex_env(), "expected": {"answer": str(ans)}, "tags": ["regex_match", "single_round"]}

# ---- Easy: simple patterns, short text ----
def gen_single_easy(start_id):
    rng = _rng(460)
    templates = [
        (r"\d+", "abc123def456", "findall", "['123', '456']"),
        (r"\d+", "hello world 42", "findall", "['42']"),
        (r"[a-z]+", "Hello World 123", "findall", "['ello', 'orld']"),
        (r"\b\w{3}\b", "The cat sat on a mat", "findall", "['The', 'cat', 'sat', 'mat']"),
        (r"\d{2,}", "a1b23c456d7", "findall", "['23', '456']"),
        (r"[A-Z]", "Hello World", "findall", "['H', 'W']"),
        (r"\s+", "hello  world\tfoo", "findall", "['  ', '\\t']"),
        (r"[aeiou]", "beautiful", "findall", "['e', 'a', 'u', 'i', 'u']"),
        (r"\w+@\w+", "user@host other@place", "findall", "['user@host', 'other@place']"),
        (r"^[A-Z]", "Hello world", "search", "H"),
        (r"\d+$", "abc 123", "search", "123"),
        (r"[^a-zA-Z0-9\s]", "Hello, World! #2024", "findall", "[',', '!', '#']"),
        (r"the", "the cat and the dog and the fish", "findall", "['the', 'the', 'the']"),
        (r"\bcat\b", "the cat concatenated", "findall", "['cat']"),
        (r"\d", "a1b2c3d4e5", "findall", "['1', '2', '3', '4', '5']"),
        (r"[A-Z][a-z]+", "Hello World Foo Bar", "findall", "['Hello', 'World', 'Foo', 'Bar']"),
        (r"\.", "3.14.159", "findall", "['.', '.']"),
        (r"o+", "foooobar boo", "findall", "['oooo', 'oo']"),
        (r"\b\d+\b", "item 42 costs 100 dollars", "findall", "['42', '100']"),
        (r"[xyz]", "extra yellow zone", "findall", "['x', 'y', 'z']"),
        (r"\w+", "hello-world foo_bar", "findall", "['hello', 'world', 'foo_bar']"),
        (r"[0-9]{3}", "abc 123 def 4567 ghi 89", "findall", "['123', '456']"),
        (r"a.c", "abc adc aec a c", "findall", "['abc', 'adc', 'aec', 'a c']"),
        (r"\S+", "  hello   world  ", "findall", "['hello', 'world']"),
        (r"[bcdfghjklmnpqrstvwxyz]", "hello", "findall", "['h', 'l', 'l']"),
        (r"^hello", "hello world", "search", "hello"),
        (r"world$", "hello world", "search", "world"),
        (r"\b\w+ing\b", "running jumping sitting still", "findall", "['running', 'jumping', 'sitting']"),
        (r"[A-Za-z]+", "123abc456DEF", "findall", "['abc', 'DEF']"),
        (r"\d+\.\d+", "pi is 3.14 and e is 2.72", "findall", "['3.14', '2.72']"),
        (r"(ab)+", "ababab cd abab", "findall", "['ab', 'ab']"),
        (r"colou?r", "color and colour", "findall", "['color', 'colour']"),
        (r"\b[A-Z]\w*", "Alice met Bob at Central Park", "findall", "['Alice', 'Bob', 'Central', 'Park']"),
        (r"[!?.]", "Hello! How are you? Fine.", "findall", "['!', '?', '.']"),
        (r"(.)\1", "aabbcc deed", "findall", "['a', 'b', 'c', 'e']"),
        (r"\d{2}-\d{2}", "12-34 and 56-78", "findall", "['12-34', '56-78']"),
        (r"an?", "a an and banana", "findall", "['a', 'an', 'an', 'an', 'a', 'a']"),
        (r"[^0-9]+", "abc123def456", "findall", "['abc', 'def']"),
        (r"cat|dog", "my cat and your dog", "findall", "['cat', 'dog']"),
        (r"\w{5}", "the quick brown fox jumps", "findall", "['quick', 'brown', 'jumps']"),
        (r"[aeiou]{2,}", "beautiful queue", "findall", "['eau', 'i', 'ue', 'ue']"),
        (r"\b\w{4}\b", "the quick brown fox", "findall", "['quick', 'brown']"),
        # Additional easy templates
        (r"[A-Z]+", "USA FBI CIA NSA", "findall", ""),
        (r"\b\w+ly\b", "quickly slowly happily sad", "findall", ""),
        (r"[0-9]+", "abc 12 def 345 ghi 6", "findall", ""),
        (r"\b[aeiou]\w*", "apple orange under ice every", "findall", ""),
        (r"[^aeiou\s]+", "hello world", "findall", ""),
        (r"\w+(?=\.)", "file.txt image.png doc.pdf", "findall", ""),
        (r"(?<=@)\w+", "user@gmail admin@yahoo test@msn", "findall", ""),
        (r"\b\w{2}\b", "I am at my in on so do", "findall", ""),
        (r"[A-Za-z]\d", "a1 b2 cc d3 4e f5", "findall", ""),
        (r"\d[A-Za-z]", "1a 2b c3 4d 5e", "findall", ""),
        (r"\b\w+ed\b", "walked jumped playing stopped", "findall", ""),
        (r"[!@#$%]", "hello! email@test #tag $100 25%", "findall", ""),
        (r"\b[A-Z]\b", "A B c d E f G", "findall", ""),
        (r"[a-z]{4,}", "the cat sat on a mat quickly", "findall", ""),
        (r"\d+\s\d+", "12 34 and 56 78", "findall", ""),
        (r"[aeiou]+", "queue beautiful", "findall", ""),
        (r"\b\w+tion\b", "nation action motion still", "findall", ""),
        (r"[^a-z]+", "Hello World 123", "findall", ""),
        (r"\b[b-df-hj-np-tv-z]+\b", "dry fly try cry why", "findall", ""),
        (r"[0-9a-f]+", "abc 123 def 456 ghi", "findall", ""),
        (r"\b\w+ness\b", "darkness kindness happy sadness", "findall", ""),
        (r"(?:Mr|Ms|Dr)\.", "Mr. Smith and Ms. Jones and Dr. Brown", "findall", ""),
        (r"\b\d{1,2}\b", "5 42 100 7 999 13", "findall", ""),
        (r"[A-Z][a-z]*", "Hello World IBM Test", "findall", ""),
        (r"\b\w+er\b", "taller bigger smaller loud", "findall", ""),
        (r"\s{2,}", "hello  world   foo    bar", "findall", ""),
        (r"[a-z](?=[A-Z])", "helloWorld fooBar testCase", "findall", ""),
        (r"\b\w+ful\b", "beautiful helpful joyful sad", "findall", ""),
        (r"[+-]?\d+", "+5 -3 42 -17 +0", "findall", ""),
        (r"\b\w+able\b", "readable writable stable quick", "findall", ""),
    ]
    # Recompute answers to ensure correctness
    verified = []
    for pat, text, op, _ in templates:
        if op == "findall":
            ans = str(_re.findall(pat, text))
        elif op == "search":
            m = _re.search(pat, text)
            ans = m.group() if m else "None"
        verified.append((pat, text, op, ans))

    rng.shuffle(verified)
    out = []
    seen = set()
    for pat, text, op, ans in verified[:N_TOTAL]:
        key = f"{pat}|{text}|{op}"
        if key in seen: continue
        seen.add(key)
        if op == "findall":
            instr = f"What does the regex pattern r'{pat}' find in the text \"{text}\" using findall? Answer with only the Python list."
        else:
            instr = f"What does the regex pattern r'{pat}' match (search) in \"{text}\"? Answer with only the matched string."
        out.append(task(start_id + len(out), "easy", instr, ans))
    return out

# ---- Medium: groups, quantifiers, alternation on moderate text ----
MEDIUM_TEMPLATES = [
    (r"(\w+)@(\w+)\.(\w+)", "user@example.com admin@test.org", "findall"),
    (r"(\d{1,3}\.){3}\d{1,3}", "IP: 192.168.1.1 and 10.0.0.255", "findall"),
    (r"\b(\w+)\s+\1\b", "the the cat sat sat on on on a mat", "findall"),
    (r"(?:Mr|Mrs|Dr)\.\s\w+", "Dr. Smith and Mrs. Jones met Mr. Brown", "findall"),
    (r"\b\w*([aeiou])\1\w*\b", "book feel good pool deer bat", "findall"),
    (r"(\d+)-(\d+)-(\d+)", "Date: 2024-01-15 and 2023-12-25", "findall"),
    (r"https?://\S+", "Visit http://example.com or https://test.org/path", "findall"),
    (r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+", "John Smith met Mary Jane at New York City", "findall"),
    (r"(?<=@)\w+\.\w+", "user@gmail.com and admin@yahoo.org", "findall"),
    (r"\b\w+(?:ing|tion|ment)\b", "The running motivation and development are improving", "findall"),
    (r"\(([^)]+)\)", "call (555) 123-4567 or (800) 999-0000", "findall"),
    (r"[A-Z]{2,}", "NASA and FBI met at the UN headquarters", "findall"),
    (r"(\w)\1{2,}", "aaa bbb cc dddd e fffff", "findall"),
    (r"(?:^|\s)#\w+", "Hello #world this is #python and #regex", "findall"),
    (r"\b\d+(?:\.\d+)?%", "Scores: 95.5% and 87% and 100.0%", "findall"),
    (r"<(\w+)>.*?</\1>", "<b>bold</b> and <i>italic</i> text", "findall"),
    (r"\b[a-z]+(?:[A-Z][a-z]+)+\b", "use camelCase and myVariableName here", "findall"),
    (r"(?:\+\d{1,3}\s?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}", "Call +1 (555) 123-4567 or 800-999-0000", "findall"),
    (r"\b(\w{3})\w*\1\b", "abcabc murmur abracadabra", "findall"),
    (r"(?i)the", "The THE the tHe ThE", "findall"),
    (r"[^,]+", "apple,banana,cherry,date", "findall"),
    (r"\b\w+[^aeiou\W]\b", "cat dog fish bird ant", "findall"),
    (r"(\w+)=(\w+)", "name=John age=30 city=NYC", "findall"),
    (r"(?:19|20)\d{2}", "Years: 1999 2000 2024 1850 2099", "findall"),
    (r"\b[ATCG]{5,}\b", "The sequence ATCGATCG and GCCTA and AT", "findall"),
    (r"(?<=\d)\s(?=\w)", "Room 42 A and Floor 3 B", "findall"),
    (r"\b\w+(?:ed|ly|er|est)\b", "She walked quickly to the tallest darker building", "findall"),
    (r"[^\w\s]", "Hello, World! How's it going? Fine... #great", "findall"),
    (r"\b(\w+)\b(?=.*\b\1\b)", "the cat and the dog and the cat", "findall"),
    (r"(?:mon|tues|wednes|thurs|fri|satur|sun)day", "On monday and friday we rest, not saturday", "findall"),
    (r"\d+(?:st|nd|rd|th)", "1st 2nd 3rd 4th 11th 22nd 33rd", "findall"),
    (r"(?<!\w)\d+(?!\w)", "abc 42 def56 78 ghi", "findall"),
    (r"\b\w*oo\w*\b", "the moon shone on the cool pool", "findall"),
    (r"[A-Z]\w+(?=\s[A-Z])", "New York San Francisco Los Angeles", "findall"),
    (r"\b\w+(?:ness|ment|tion)\b", "happiness improvement creation sadness", "findall"),
    (r"(?:\d{1,3}\.){3}\d{1,3}", "Servers: 10.0.0.1 and 172.16.0.100 down", "findall"),
    (r"[a-f0-9]{6}", "Colors: ff0000 and 00ff00 and zzzzzz", "findall"),
    (r"\b\w+s\b", "cats dogs fish boxes the class", "findall"),
    (r'"[^"]*"', 'He said "hello" and "goodbye" today', "findall"),
    (r"(\d+)\s*\+\s*(\d+)", "Compute 12 + 34 and 56+78", "findall"),
    (r"(?:^|\s)[A-Z]{2,}(?:\s|$)", "NASA went to ISS and met the FBI", "findall"),
    (r"\b\d{4}-\d{2}-\d{2}\b", "Dates: 2024-01-15 and 2023-12-25 and 99-1-1", "findall"),
    # Additional medium templates
    (r"\b[A-Z][a-z]+\b", "The Quick Brown Fox Jumps Over Lazy Dog", "findall"),
    (r"\b\w+(?:ed|ing)\b", "He was walking and talked while running fast", "findall"),
    (r"(?<=\s)\d+(?=\s)", "item 42 costs 100 dollars today", "findall"),
    (r"\b\w+@\w+\.\w{2,4}\b", "Contact user@test.com or admin@site.org for help", "findall"),
    (r"(?:https?://)\S+", "Visit http://a.com and https://b.org/path?q=1", "findall"),
    (r"\b(\w)\w*\1\b", "level noon racecar hello test", "findall"),
    (r"[A-Z][a-z]+(?=\s[a-z])", "Alice met Bob at the Central park today", "findall"),
    (r"\d+(?:\.\d+)?(?:e[+-]?\d+)?", "Values: 3.14 1e10 2.5e-3 42", "findall"),
    (r"\b\w{6,8}\b", "the quickly jumping fox leaped over", "findall"),
    (r"(?:^|(?<=\s))@\w+", "Hey @alice and @bob check @this out", "findall"),
    (r"\b\w+ous\b", "famous dangerous curious happy anxious", "findall"),
    (r"\([^)]+\)", "Call (John) or (Jane) at (555) 1234", "findall"),
    (r"\b\w+(?:ize|ise)\b", "organize supervise maximize and optimize", "findall"),
    (r"[A-Z]{3,}", "Use NASA GPS and FBI not Hi", "findall"),
    (r"\d{2}:\d{2}", "Meet at 09:30 or 14:45 not 9am", "findall"),
    (r"\b\w+ly\b", "She quickly and happily ran slowly away", "findall"),
    (r"(?<=\d)\s*(?:kg|lb|oz|g)\b", "Items: 5 kg 3 lb 100 g 8 oz", "findall"),
    (r"\b[a-z]+(?=[A-Z])", "camelCase myVariable firstName lastName", "findall"),
    (r"\b\d{5}(?:-\d{4})?\b", "ZIP: 90210 and 10001-1234 and 123", "findall"),
    (r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", "Meet Jan 5 or Mar 10 or May", "findall"),
    (r"(?:^|\s)#[a-zA-Z]\w*", "Post #hello #World #123 #a", "findall"),
    (r"\b\w+(?:ment|ness|tion|sion)\b", "development happiness solution decision quick", "findall"),
    (r"\b\d+(?:st|nd|rd|th)\b", "Rank: 1st 2nd 3rd 4th 11th 22nd", "findall"),
    (r"(?:^|\n)\s*-\s*.+", "List:\n- item one\n- item two\nnot this", "findall"),
    (r"\w+://\w+\.\w+", "Use http://test.com or ftp://files.net", "findall"),
    (r"\b[A-Z]\w+(?:\s[A-Z]\w+){1,2}", "Visit New York City or San Francisco Bay", "findall"),
    (r"\b\w+[^aeiouAEIOU\W]\b", "cat dog fish jump walk run", "findall"),
    (r"\b(?:is|am|are|was|were)\b", "He is smart and they are kind and I am here", "findall"),
    (r"(?:todo|fixme|hack):", "Check todo: fix this and fixme: later hack: temp", "findall"),
    (r"\b\d+x\d+\b", "Size 1920x1080 or 800x600 or 4K", "findall"),
]

def gen_single_medium(start_id):
    rng = _rng(570)
    verified = []
    for pat, text, op in MEDIUM_TEMPLATES:
        if op == "findall":
            ans = str(_re.findall(pat, text))
        elif op == "search":
            m = _re.search(pat, text)
            ans = m.group() if m else "None"
        verified.append((pat, text, op, ans))
    rng.shuffle(verified)
    out = []
    seen = set()
    for pat, text, op, ans in verified[:N_TOTAL]:
        key = f"{pat}|{text}"
        if key in seen: continue
        seen.add(key)
        instr = f"What does re.findall(r'{pat}', \"{text}\") return? Answer with only the Python list."
        out.append(task(start_id + len(out), "medium", instr, ans))
    return out

# ---- Hard: complex patterns on long text requiring many backtracks ----
def _gen_hard_tasks(rng):
    """Generate hard regex tasks with long text and complex patterns."""
    tasks_data = []

    # Overlapping matches with lookahead on generated text
    for seed_offset in range(18):
        r = random.Random(SEED + 700 + seed_offset)
        words = [r.choice(["ab","bc","cd","abc","bcd","abcd","de","ef"]) for _ in range(20)]
        text = "".join(words)
        pat = r"(?=([a-z]{3}))"
        ans = str(_re.findall(pat, text))
        tasks_data.append((pat, text, "findall", ans))

    # Nested groups with quantifiers
    for n_pairs in range(5, 23):
        r = random.Random(SEED + 800 + n_pairs)
        pairs = [(f"k{r.randint(1,99)}", f"v{r.randint(100,999)}") for _ in range(n_pairs)]
        text = "&".join(f"{k}={v}" for k, v in pairs)
        pat = r"(\w+)=(\w+)"
        ans = str(_re.findall(pat, text))
        tasks_data.append((pat, text, "findall", ans))

    # Greedy vs non-greedy on HTML-like text
    for seed_offset in range(18):
        r = random.Random(SEED + 900 + seed_offset)
        tags = r.sample(["div","span","b","i","p","a","em","strong","code","pre"], k=r.randint(3,5))
        parts = []
        for t in tags:
            content = "".join(r.choices("abcdefg ", k=r.randint(3,10)))
            parts.append(f"<{t}>{content}</{t}>")
        text = " ".join(parts)
        pat = r"<(\w+)>[^<]+</\1>"
        ans = str(_re.findall(pat, text))
        tasks_data.append((pat, text, "findall", ans))

    # Complex character class on long string
    for seed_offset in range(18):
        r = random.Random(SEED + 1000 + seed_offset)
        text = "".join(r.choices("abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*() ", k=60))
        pat = r"[a-z]+(?=\d)"
        ans = str(_re.findall(pat, text))
        tasks_data.append((pat, text, "findall", ans))

    return tasks_data

def gen_single_hard(start_id):
    rng = _rng(680)
    all_tasks = _gen_hard_tasks(rng)
    rng.shuffle(all_tasks)
    out = []
    seen = set()
    for pat, text, op, ans in all_tasks[:N_TOTAL]:
        key = f"{pat}|{text}"
        if key in seen: continue
        seen.add(key)
        display_text = text if len(text) <= 80 else text
        instr = f"What does re.findall(r'{pat}', \"{display_text}\") return? Answer with only the Python list."
        out.append(task(start_id + len(out), "hard", instr, ans))
    return out

def validate(tasks, label, expected_n=N_TOTAL):
    import re
    if len(tasks) != expected_n:
        raise ValueError(f"{label}: expected {expected_n}, got {len(tasks)}")
    inst = [re.sub(r"\s+", " ", t["instruction"].strip().lower()) for t in tasks]
    if len(set(inst)) != len(inst):
        raise ValueError(f"{label}: duplicate instruction found")

def split_train_test(tasks): return tasks[:N_SPLIT], tasks[N_SPLIT:]

def main():
    gens = {
        "regex_match_single_easy": gen_single_easy(5300),
        "regex_match_single_medium": gen_single_medium(5400),
        "regex_match_single_hard": gen_single_hard(5500),
    }
    for name, tasks in gens.items():
        validate(tasks, name)
        train, test = split_train_test(tasks)
        for sn, st in [("train", train), ("test", test)]:
            fname = f"{name}_{sn}.json"
            (OUT / fname).write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n")
            print(f"wrote {fname}: {len(st)}")

if __name__ == "__main__": main()
