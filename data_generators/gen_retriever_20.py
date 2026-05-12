import json
import random
import re
from pathlib import Path
from paragraphs_retriever import EASY_PARAGRAPHS, MEDIUM_PARAGRAPHS, HARD_PARAGRAPHS, HARD_FACTS

# Merged paragraph lookup: (template, subject) -> long paragraph
_ALL_PARAGRAPHS = {}
_ALL_PARAGRAPHS.update(EASY_PARAGRAPHS)
_ALL_PARAGRAPHS.update(MEDIUM_PARAGRAPHS)
_ALL_PARAGRAPHS.update(HARD_PARAGRAPHS)


BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "data" / "task_sets"
OUT.mkdir(parents=True, exist_ok=True)
N_TOTAL = 70
N_SPLIT = 20
SEED = 20260305

RETRIEVER_TOOLS = ["search_corpus", "read_doc"]


def retriever_env(corpus):
    return [
        {
            "name": "RetrieverEnv",
            "tools": RETRIEVER_TOOLS,
            "parameters": {"corpus": corpus},
        }
    ]


# 10 relation templates for question -> answer mappings.
SINGLE_TEMPLATES = {
    "capital_city": {
        "question": "What is the capital of {subject}? Return city only.",
        "title": "{subject} capital city reference",
        "text": "The capital of {subject} is {answer}.",
        "step1_desc": "capital of {subject}",
    },
    "country_currency": {
        "question": "What currency is used in {subject}? Return currency name only.",
        "title": "{subject} currency reference",
        "text": "The currency used in {subject} is {answer}.",
        "step1_desc": "currency used in {subject}",
    },
    "element_symbol": {
        "question": "What is the chemical symbol for {subject}? Return symbol only.",
        "title": "{subject} chemical symbol reference",
        "text": "The chemical symbol for {subject} is {answer}.",
        "step1_desc": "chemical symbol for {subject}",
    },
    "work_author": {
        "question": "Who wrote {subject}? Return full name.",
        "title": "{subject} author reference",
        "text": "{subject} was written by {answer}.",
        "step1_desc": "author of {subject}",
    },
    "planet_nickname": {
        "question": "Which planet is known as {subject}? Return planet name only.",
        "title": "{subject} planet nickname reference",
        "text": "The planet known as {subject} is {answer}.",
        "step1_desc": "planet known as {subject}",
    },
    "highest_mountain": {
        "question": "What is the highest mountain in {subject}? Return mountain name only.",
        "title": "{subject} highest mountain reference",
        "text": "The highest mountain in {subject} is {answer}.",
        "step1_desc": "highest mountain in {subject}",
    },
    "largest_moon": {
        "question": "What is the largest moon of {subject}? Return moon name only.",
        "title": "{subject} largest moon reference",
        "text": "The largest moon of {subject} is {answer}.",
        "step1_desc": "largest moon of {subject}",
    },
    "inventor": {
        "question": "Who is credited with inventing the {subject}? Return full name.",
        "title": "{subject} inventor reference",
        "text": "The {subject} is commonly credited to {answer}.",
        "step1_desc": "inventor of the {subject}",
    },
    "official_language": {
        "question": "What is an official language of {subject}? Return one language.",
        "title": "{subject} official language reference",
        "text": "An official language of {subject} is {answer}.",
        "step1_desc": "an official language of {subject}",
    },
    "country_continent": {
        "question": "Which continent is {subject} in? Return continent only.",
        "title": "{subject} continent reference",
        "text": "{subject} is in {answer}.",
        "step1_desc": "continent of {subject}",
    },
}


# Easy facts: common knowledge, often answerable without retrieval.
EASY_FACTS = [
    ("capital_city", "France", "Paris"),
    ("capital_city", "Japan", "Tokyo"),
    ("capital_city", "Italy", "Rome"),
    ("capital_city", "Spain", "Madrid"),
    ("capital_city", "Germany", "Berlin"),
    ("capital_city", "China", "Beijing"),
    ("capital_city", "India", "New Delhi"),
    ("capital_city", "Egypt", "Cairo"),
    ("country_currency", "Japan", "yen"),
    ("country_currency", "China", "yuan"),
    ("country_currency", "India", "Indian rupee"),
    ("country_currency", "United States", "US dollar"),
    ("country_currency", "United Kingdom", "pound sterling"),
    ("element_symbol", "Hydrogen", "H"),
    ("element_symbol", "Oxygen", "O"),
    ("element_symbol", "Carbon", "C"),
    ("element_symbol", "Nitrogen", "N"),
    ("element_symbol", "Gold", "Au"),
    ("element_symbol", "Silver", "Ag"),
    ("element_symbol", "Iron", "Fe"),
    ("work_author", "Hamlet", "William Shakespeare"),
    ("work_author", "1984", "George Orwell"),
    ("work_author", "Pride and Prejudice", "Jane Austen"),
    ("work_author", "The Hobbit", "J. R. R. Tolkien"),
    ("work_author", "Moby-Dick", "Herman Melville"),
    ("work_author", "The Iliad", "Homer"),
    ("planet_nickname", "the Red Planet", "Mars"),
    ("planet_nickname", "the Blue Planet", "Earth"),
    ("planet_nickname", "the Ringed Planet", "Saturn"),
    ("planet_nickname", "the Morning Star", "Venus"),
    ("highest_mountain", "Japan", "Mount Fuji"),
    ("highest_mountain", "Nepal", "Mount Everest"),
    ("highest_mountain", "Tanzania", "Mount Kilimanjaro"),
    ("highest_mountain", "United States", "Denali"),
    ("largest_moon", "Mars", "Phobos"),
    ("largest_moon", "Jupiter", "Ganymede"),
    ("largest_moon", "Saturn", "Titan"),
    ("inventor", "telephone", "Alexander Graham Bell"),
    ("inventor", "light bulb", "Thomas Edison"),
    ("inventor", "printing press", "Johannes Gutenberg"),
    ("inventor", "radio", "Guglielmo Marconi"),
    ("official_language", "Brazil", "Portuguese"),
    ("official_language", "Germany", "German"),
    ("official_language", "Mexico", "Spanish"),
    ("official_language", "Egypt", "Arabic"),
    ("official_language", "Canada", "English"),
    ("country_continent", "Japan", "Asia"),
    ("country_continent", "Brazil", "South America"),
    ("country_continent", "France", "Europe"),
    ("country_continent", "Kenya", "Africa"),
    ("country_continent", "Australia", "Oceania"),
    # Additional easy facts for N_TOTAL=70
    ("capital_city", "Russia", "Moscow"),
    ("capital_city", "South Korea", "Seoul"),
    ("capital_city", "Canada", "Ottawa"),
    ("capital_city", "Brazil", "Brasilia"),
    ("country_currency", "Australia", "Australian dollar"),
    ("country_currency", "Mexico", "Mexican peso"),
    ("country_currency", "Russia", "Russian ruble"),
    ("element_symbol", "Helium", "He"),
    ("element_symbol", "Calcium", "Ca"),
    ("element_symbol", "Sodium", "Na"),
    ("element_symbol", "Copper", "Cu"),
    ("work_author", "Romeo and Juliet", "William Shakespeare"),
    ("work_author", "The Great Gatsby", "F. Scott Fitzgerald"),
    ("work_author", "War and Peace", "Leo Tolstoy"),
    ("inventor", "dynamite", "Alfred Nobel"),
    ("inventor", "penicillin", "Alexander Fleming"),
    ("official_language", "Japan", "Japanese"),
    ("official_language", "South Korea", "Korean"),
    ("country_continent", "Germany", "Europe"),
    ("country_continent", "Mexico", "North America"),
    ("country_continent", "India", "Asia"),
    ("planet_nickname", "the Stormy Planet", "Neptune"),
    ("highest_mountain", "France", "Mont Blanc"),
]


# Medium facts: less common, model might know some.
MEDIUM_FACTS = [
    ("capital_city", "Australia", "Canberra"),
    ("capital_city", "New Zealand", "Wellington"),
    ("capital_city", "Switzerland", "Bern"),
    ("capital_city", "Kazakhstan", "Astana"),
    ("capital_city", "Slovenia", "Ljubljana"),
    ("capital_city", "Croatia", "Zagreb"),
    ("capital_city", "Morocco", "Rabat"),
    ("capital_city", "Iceland", "Reykjavik"),
    ("capital_city", "Myanmar", "Naypyidaw"),
    ("capital_city", "Finland", "Helsinki"),
    ("country_currency", "Switzerland", "Swiss franc"),
    ("country_currency", "Norway", "Norwegian krone"),
    ("country_currency", "Sweden", "Swedish krona"),
    ("country_currency", "Denmark", "Danish krone"),
    ("country_currency", "Poland", "zloty"),
    ("country_currency", "Czech Republic", "Czech koruna"),
    ("country_currency", "Thailand", "baht"),
    ("country_currency", "South Korea", "won"),
    ("country_currency", "Hungary", "forint"),
    ("country_currency", "Brazil", "Brazilian real"),
    ("element_symbol", "Potassium", "K"),
    ("element_symbol", "Mercury", "Hg"),
    ("element_symbol", "Lead", "Pb"),
    ("element_symbol", "Tin", "Sn"),
    ("element_symbol", "Antimony", "Sb"),
    ("element_symbol", "Tungsten", "W"),
    ("work_author", "Don Quixote", "Miguel de Cervantes"),
    ("work_author", "The Divine Comedy", "Dante Alighieri"),
    ("work_author", "The Trial", "Franz Kafka"),
    ("work_author", "Ulysses", "James Joyce"),
    ("work_author", "The Stranger", "Albert Camus"),
    ("work_author", "Crime and Punishment", "Fyodor Dostoevsky"),
    ("planet_nickname", "the Gas Giant with Great Red Spot", "Jupiter"),
    ("planet_nickname", "the Ice Giant with tilted axis", "Uranus"),
    ("planet_nickname", "the Dwarf Planet in the Kuiper Belt", "Pluto"),
    ("largest_moon", "Uranus", "Titania"),
    ("largest_moon", "Neptune", "Triton"),
    ("largest_moon", "Pluto", "Charon"),
    ("highest_mountain", "Australia", "Mount Kosciuszko"),
    ("highest_mountain", "Canada", "Mount Logan"),
    ("highest_mountain", "Spain", "Mount Teide"),
    ("highest_mountain", "Germany", "Zugspitze"),
    ("highest_mountain", "Argentina", "Aconcagua"),
    ("highest_mountain", "Mexico", "Pico de Orizaba"),
    ("inventor", "World Wide Web", "Tim Berners-Lee"),
    ("inventor", "AC induction motor", "Nikola Tesla"),
    ("inventor", "diesel engine", "Rudolf Diesel"),
    ("inventor", "vaccination", "Edward Jenner"),
    ("official_language", "Switzerland", "German"),
    ("official_language", "Austria", "German"),
    ("official_language", "Belgium", "Dutch"),
    ("official_language", "Finland", "Finnish"),
    ("official_language", "Netherlands", "Dutch"),
    ("official_language", "Turkey", "Turkish"),
    ("country_continent", "Kazakhstan", "Asia"),
    ("country_continent", "Greenland", "North America"),
    ("country_continent", "Chile", "South America"),
    ("country_continent", "Norway", "Europe"),
    ("country_continent", "Morocco", "Africa"),
    ("country_continent", "Thailand", "Asia"),
    # Additional medium facts for N_TOTAL=70
    ("capital_city", "Sri Lanka", "Sri Jayawardenepura Kotte"),
    ("capital_city", "Ivory Coast", "Yamoussoukro"),
    ("country_currency", "Israel", "new shekel"),
    ("country_currency", "Philippines", "Philippine peso"),
    ("element_symbol", "Strontium", "Sr"),
    ("element_symbol", "Zirconium", "Zr"),
    ("work_author", "One Hundred Years of Solitude", "Gabriel Garcia Marquez"),
    ("work_author", "The Brothers Karamazov", "Fyodor Dostoevsky"),
    ("inventor", "airplane", "Wright brothers"),
    ("inventor", "phonograph", "Thomas Edison"),
    ("country_continent", "Argentina", "South America"),
    ("country_continent", "Indonesia", "Asia"),
]


# 10 synthetic templates for hard single-step retrieval.
HARD_SINGLE_TEMPLATES = [
    "handoff hub",
    "clearance lane",
    "checksum token",
    "docking bay",
    "coolant class",
    "archive shelf",
    "relay station",
    "failover node",
    "comm window",
    "launch corridor",
]


def _normalize_inst(x):
    return re.sub(r"\s+", " ", x.strip().lower())


def _fact_to_doc(fact):
    t = SINGLE_TEMPLATES[fact["template"]]
    title = t["title"].format(subject=fact["subject"])
    # Use long paragraph if available, fallback to short template
    para_key = (fact["template"], fact["subject"])
    if para_key in _ALL_PARAGRAPHS:
        text = _ALL_PARAGRAPHS[para_key]
    else:
        text = t["text"].format(subject=fact["subject"], answer=fact["answer"])
    return {"title": title, "text": text}


def _fact_to_question(fact, local_only=False):
    t = SINGLE_TEMPLATES[fact["template"]]
    q = t["question"].format(subject=fact["subject"])
    if local_only:
        return f"From local corpus only: {q}"
    return q


def _fact_step1_desc(fact):
    t = SINGLE_TEMPLATES[fact["template"]]
    return t["step1_desc"].format(subject=fact["subject"])


def _build_fact_pool(raw_items):
    out = []
    for tpl, subject, answer in raw_items:
        out.append({"template": tpl, "subject": subject, "answer": answer})
    return out


def _new_subject(rng: random.Random):
    left = [
        "Project",
        "Operation",
        "Mission",
        "Protocol",
        "Program",
        "Campaign",
        "Directive",
        "Taskforce",
    ]
    mid = [
        "Aster",
        "Nimbus",
        "Helios",
        "Meridian",
        "Vortex",
        "Cinder",
        "Lumen",
        "Orchid",
        "Quartz",
        "Pioneer",
    ]
    return f"{rng.choice(left)} {rng.choice(mid)}-{rng.randint(10, 99)}"


def _code(rng: random.Random, prefix: str):
    return f"{prefix}-{rng.randint(10, 99)}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"


def _hard_answer_for_relation(rng: random.Random, relation: str):
    if relation == "handoff hub":
        return f"{rng.choice(['North', 'South', 'East', 'West'])} {rng.choice(['Pier', 'Quay', 'Dock', 'Terminal'])}"
    if relation == "clearance lane":
        return f"Lane-{rng.randint(1, 12)}"
    if relation == "checksum token":
        return f"{rng.choice(['Q', 'R', 'S', 'T'])}{rng.randint(10, 99)}-{rng.choice(['ALPHA', 'DELTA', 'OMEGA', 'KITE'])}-{rng.randint(100, 999)}"
    if relation == "docking bay":
        return f"Bay-{rng.randint(1, 40)}"
    if relation == "coolant class":
        return f"Class-{rng.choice(['A', 'B', 'C', 'D'])}{rng.randint(1, 9)}"
    if relation == "archive shelf":
        return f"Shelf-{rng.choice(['A', 'B', 'C', 'D'])}{rng.randint(10, 99)}"
    if relation == "relay station":
        return f"RS-{rng.randint(100, 999)}"
    if relation == "failover node":
        return f"Node-{rng.randint(200, 999)}"
    if relation == "comm window":
        return f"{rng.randint(0, 23):02d}:{rng.choice([0, 15, 30, 45]):02d} UTC"
    if relation == "launch corridor":
        return f"Corridor-{rng.choice(['A', 'B', 'C', 'D'])}{rng.randint(1, 8)}"
    raise ValueError(relation)


def _build_hard_fact(rng: random.Random):
    relation = rng.choice(HARD_SINGLE_TEMPLATES)
    subject = _new_subject(rng)
    answer = _hard_answer_for_relation(rng, relation)
    return {"relation": relation, "subject": subject, "answer": answer}


def _build_hard_fact_with_relation(rng: random.Random, relation: str):
    subject = _new_subject(rng)
    answer = _hard_answer_for_relation(rng, relation)
    return {"relation": relation, "subject": subject, "answer": answer}


def _hard_fact_doc(h):
    return {
        "title": f"{h['subject']} {h['relation']} record",
        "text": f"{h['subject']} {h['relation']} is {h['answer']}.",
    }


def _hard_fact_question(h):
    return f"From local corpus only: what is the {h['relation']} for {h['subject']}? Return answer only."


def _doc_with_id(idx, title, text):
    return {"id": f"doc{idx}", "title": title, "text": text}


def _build_noise_docs_from_facts(rng: random.Random, pool, k):
    picked = rng.sample(pool, k)
    docs = []
    for f in picked:
        d = _fact_to_doc(f)
        docs.append({"title": d["title"], "text": d["text"]})
    return docs


def _sample_diverse_facts(rng: random.Random, pool, n: int):
    by_tpl = {}
    for f in pool:
        by_tpl.setdefault(f["template"], []).append(f)
    selected = [rng.choice(v) for _, v in sorted(by_tpl.items())]
    if len(selected) > n:
        rng.shuffle(selected)
        return selected[:n]
    remaining = n - len(selected)
    left = [f for f in pool if f not in selected]
    if remaining > 0:
        selected.extend(rng.sample(left, remaining))
    rng.shuffle(selected)
    return selected


def _single_task(task_id, difficulty, instruction, corpus, answer):
    return {
        "id": task_id,
        "difficulty": difficulty,
        "multi_step": False,
        "instruction": instruction,
        "environments": retriever_env(corpus),
        "expected": {"answer": str(answer)},
        "tags": ["retrieval", "single_round"],
    }


def _materialize_corpus(raw_docs, rng=None):
    docs = list(raw_docs)
    if rng is not None:
        rng.shuffle(docs)
    out = []
    for i, d in enumerate(docs, start=1):
        out.append(_doc_with_id(i, d["title"], d["text"]))
    return out


def gen_single_easy(start_id):
    rng = random.Random(SEED + 11)
    pool = _build_fact_pool(EASY_FACTS)
    selected = _sample_diverse_facts(rng, pool, N_TOTAL)
    out = []
    seen = set()
    for i, target in enumerate(selected):
        target_doc = _fact_to_doc(target)
        same_tpl = [x for x in pool if x["template"] == target["template"] and x["subject"] != target["subject"]]
        other = [x for x in pool if x["template"] != target["template"]]
        distract = rng.sample(same_tpl, min(3, len(same_tpl))) + rng.sample(other, 6)
        rng.shuffle(distract)
        raw_docs = [target_doc] + [_fact_to_doc(x) for x in distract]
        corpus = _materialize_corpus(raw_docs[:10], rng=rng)
        instruction = _fact_to_question(target, local_only=False)
        if _normalize_inst(instruction) in seen:
            continue
        seen.add(_normalize_inst(instruction))
        out.append(_single_task(start_id + len(out), "easy", instruction, corpus, target["answer"]))
    return out


def gen_single_medium(start_id):
    rng = random.Random(SEED + 22)
    medium_pool = _build_fact_pool(MEDIUM_FACTS)
    easy_pool = _build_fact_pool(EASY_FACTS)
    selected = _sample_diverse_facts(rng, medium_pool, N_TOTAL)
    out = []
    seen = set()
    for target in selected:
        target_doc = _fact_to_doc(target)
        same_tpl = [x for x in medium_pool if x["template"] == target["template"] and x["subject"] != target["subject"]]
        mixed = [x for x in medium_pool + easy_pool if x["template"] != target["template"]]
        distract = rng.sample(same_tpl, min(3, len(same_tpl))) + rng.sample(mixed, 6)
        rng.shuffle(distract)
        raw_docs = [target_doc] + [_fact_to_doc(x) for x in distract]
        corpus = _materialize_corpus(raw_docs[:10], rng=rng)
        instruction = f"From local corpus: {_fact_to_question(target, local_only=False)}"
        if _normalize_inst(instruction) in seen:
            continue
        seen.add(_normalize_inst(instruction))
        out.append(_single_task(start_id + len(out), "medium", instruction, corpus, target["answer"]))
    return out


def _hard_fact_doc_from_entry(category, entity):
    """Build a doc dict from a HARD_FACTS entry using HARD_PARAGRAPHS."""
    title = f"{entity} {category.replace('_', ' ')} reference"
    text = HARD_PARAGRAPHS[(category, entity)]
    return {"title": title, "text": text}


def gen_single_hard(start_id):
    rng = random.Random(SEED + 33)
    easy_pool = _build_fact_pool(EASY_FACTS)
    medium_pool = _build_fact_pool(MEDIUM_FACTS)
    all_noise = easy_pool + medium_pool

    # Shuffle HARD_FACTS deterministically and take first N_TOTAL
    facts = list(HARD_FACTS)
    rng.shuffle(facts)
    selected = facts[:N_TOTAL]

    out = []
    seen = set()
    for category, entity, answer, question_template in selected:
        instruction = f"From local corpus only: {question_template.format(entity=entity)}"
        if _normalize_inst(instruction) in seen:
            continue
        seen.add(_normalize_inst(instruction))

        target_doc = _hard_fact_doc_from_entry(category, entity)

        # Pick one hard distractor from a different fact
        others = [(c, e, a, q) for c, e, a, q in HARD_FACTS if (c, e) != (category, entity)]
        alt_cat, alt_ent, _, _ = rng.choice(others)
        alt_doc = _hard_fact_doc_from_entry(alt_cat, alt_ent)

        raw_docs = [target_doc, alt_doc]
        raw_docs.extend(_build_noise_docs_from_facts(rng, all_noise, 8))
        corpus = _materialize_corpus(raw_docs[:10], rng=rng)
        out.append(_single_task(start_id + len(out), "hard", instruction, corpus, answer))
    return out


def validate(tasks, label, expected_n=N_TOTAL):
    if len(tasks) != expected_n:
        raise ValueError(f"{label}: expected {expected_n}, got {len(tasks)}")
    inst = [_normalize_inst(t["instruction"]) for t in tasks]
    if len(inst) != len(set(inst)):
        raise ValueError(f"{label}: duplicate instructions")
    for t in tasks:
        env = t["environments"][0]
        corpus = env["parameters"]["corpus"]
        if not corpus:
            raise ValueError(f"{label}: empty corpus")
        # Titles should carry cues; avoid generic useless titles.
        for d in corpus:
            title = d.get("title", "").strip()
            if len(title.split()) < 2:
                raise ValueError(f"{label}: weak title '{title}'")


def split_train_test(tasks):
    return tasks[:N_SPLIT], tasks[N_SPLIT:]


def main():
    generators = {
        "retriever_single_easy": gen_single_easy(3000),
        "retriever_single_medium": gen_single_medium(3100),
        "retriever_single_hard": gen_single_hard(3200),
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
