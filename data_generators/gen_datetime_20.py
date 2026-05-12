"""Generate single-step DateTime tasks (easy / medium / hard)."""
import json, re
import random
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "data" / "task_sets"
OUT.mkdir(parents=True, exist_ok=True)
N_TOTAL = 70
N_SPLIT = 20
SEED = 20260304

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


def _rng(offset=0):
    return random.Random(SEED + offset)


def dt_env():
    return [{"name": "DateTimeEnv", "tools": ["date_diff", "date_add", "day_of_week"], "parameters": {}}]


def task_single(tid, difficulty, instruction, answer):
    return {
        "id": tid, "difficulty": difficulty, "multi_step": False,
        "instruction": instruction, "environments": dt_env(),
        "expected": {"answer": str(answer)}, "tags": ["datetime", "single_round"],
    }


# ---- Easy: within one month, no edge cases ----
def gen_single_easy(start_id):
    rng = _rng(200)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = len(out) % 3
        year = rng.randint(2020, 2026)
        month = rng.randint(1, 12)
        if p == 0:
            # Days between two dates in same month
            d1 = rng.randint(1, 15)
            d2 = rng.randint(d1 + 2, 28)
            date1 = f"{year}-{month:02d}-{d1:02d}"
            date2 = f"{year}-{month:02d}-{d2:02d}"
            key = f"diff_{date1}_{date2}"
            if key in seen:
                continue
            seen.add(key)
            ans = d2 - d1
            m_name = MONTH_NAMES[month - 1]
            instr = (f"How many days are between {m_name} {d1} and {m_name} {d2}, {year}? "
                     f"Answer with only the number.")
        elif p == 1:
            # Add small number of days within month
            day = rng.randint(1, 20)
            add = rng.randint(1, 7)
            date_str = f"{year}-{month:02d}-{day:02d}"
            result = datetime(year, month, day) + timedelta(days=add)
            key = f"add_{date_str}_{add}"
            if key in seen:
                continue
            seen.add(key)
            ans = result.strftime("%Y-%m-%d")
            m_name = MONTH_NAMES[month - 1]
            instr = (f"What date is {add} days after {m_name} {day}, {year}? "
                     f"Answer in YYYY-MM-DD format only.")
        else:
            # Day of week for a recent date
            day = rng.randint(1, 28)
            dt = datetime(year, month, day)
            key = f"dow_{year}_{month}_{day}"
            if key in seen:
                continue
            seen.add(key)
            ans = DAY_NAMES[dt.weekday()]
            m_name = MONTH_NAMES[month - 1]
            instr = (f"What day of the week is {m_name} {day}, {year}? "
                     f"Answer with only the day name (e.g. Monday).")
        out.append(task_single(start_id + len(out), "easy", instr, ans))
    return out


# ---- Medium: crossing month/year boundaries, leap year ----
def gen_single_medium(start_id):
    rng = _rng(310)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = len(out) % 3
        if p == 0:
            # Days between dates crossing month boundary
            year = rng.randint(2020, 2028)
            month1 = rng.randint(1, 11)
            day1 = rng.randint(20, 28)
            month2 = month1 + rng.randint(1, 2)
            day2 = rng.randint(1, 15)
            try:
                d1 = datetime(year, month1, day1)
                d2 = datetime(year, month2, day2)
            except ValueError:
                continue
            key = f"diff_{d1}_{d2}"
            if key in seen:
                continue
            seen.add(key)
            ans = abs((d2 - d1).days)
            m1_name = MONTH_NAMES[month1 - 1]
            m2_name = MONTH_NAMES[month2 - 1]
            instr = (f"How many days are between {m1_name} {day1} and {m2_name} {day2}, {year}? "
                     f"Answer with only the number.")
        elif p == 1:
            # Add days crossing year boundary
            year = rng.randint(2020, 2026)
            month = 12
            day = rng.randint(25, 31)
            add = rng.randint(5, 30)
            try:
                d = datetime(year, month, day)
            except ValueError:
                continue
            result = d + timedelta(days=add)
            key = f"add_{year}_{month}_{day}_{add}"
            if key in seen:
                continue
            seen.add(key)
            ans = result.strftime("%Y-%m-%d")
            instr = (f"What date is {add} days after December {day}, {year}? "
                     f"Answer in YYYY-MM-DD format only.")
        else:
            # Leap year Feb 29 related
            leap_year = rng.choice([2020, 2024, 2028, 2032])
            d1 = datetime(leap_year, 2, 25)
            add = rng.randint(5, 15)
            result = d1 + timedelta(days=add)
            key = f"leap_{leap_year}_{add}"
            if key in seen:
                continue
            seen.add(key)
            ans = result.strftime("%Y-%m-%d")
            instr = (f"What date is {add} days after February 25, {leap_year}? "
                     f"(Note: {leap_year} is a leap year.) Answer in YYYY-MM-DD format only.")
        out.append(task_single(start_id + len(out), "medium", instr, ans))
    return out


# ---- Hard: day of week for arbitrary dates, multi-year calculations ----
def gen_single_hard(start_id):
    rng = _rng(420)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = len(out) % 3
        if p == 0:
            # Day of week for arbitrary future/past date
            year = rng.randint(1900, 2100)
            month = rng.randint(1, 12)
            day = rng.randint(1, 28)
            dt = datetime(year, month, day)
            key = f"dow_{year}_{month}_{day}"
            if key in seen:
                continue
            seen.add(key)
            ans = DAY_NAMES[dt.weekday()]
            m_name = MONTH_NAMES[month - 1]
            instr = (f"What day of the week is {m_name} {day}, {year}? "
                     f"Answer with only the day name (e.g. Monday).")
        elif p == 1:
            # Days between dates years apart
            y1 = rng.randint(1950, 2000)
            y2 = rng.randint(2020, 2050)
            m1, d1 = rng.randint(1, 12), rng.randint(1, 28)
            m2, d2 = rng.randint(1, 12), rng.randint(1, 28)
            try:
                dt1 = datetime(y1, m1, d1)
                dt2 = datetime(y2, m2, d2)
            except ValueError:
                continue
            key = f"diff_{dt1}_{dt2}"
            if key in seen:
                continue
            seen.add(key)
            ans = abs((dt2 - dt1).days)
            m1_name = MONTH_NAMES[m1 - 1]
            m2_name = MONTH_NAMES[m2 - 1]
            instr = (f"How many days are between {m1_name} {d1}, {y1} and {m2_name} {d2}, {y2}? "
                     f"Answer with only the number.")
        else:
            # Add large number of days
            year = rng.randint(2020, 2030)
            month = rng.randint(1, 12)
            day = rng.randint(1, 28)
            add = rng.randint(365, 3650)
            try:
                d = datetime(year, month, day)
            except ValueError:
                continue
            result = d + timedelta(days=add)
            key = f"add_{year}_{month}_{day}_{add}"
            if key in seen:
                continue
            seen.add(key)
            ans = result.strftime("%Y-%m-%d")
            instr = (f"What date is {add} days after {MONTH_NAMES[month-1]} {day}, {year}? "
                     f"Answer in YYYY-MM-DD format only.")
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
        "datetime_single_easy": gen_single_easy(4400),
        "datetime_single_medium": gen_single_medium(4500),
        "datetime_single_hard": gen_single_hard(4600),
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
