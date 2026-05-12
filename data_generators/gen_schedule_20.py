"""Generate single-step Schedule tasks (easy / medium / hard)."""
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


def schedule_env():
    return [{"name": "ScheduleEnv", "tools": ["find_free_slots", "check_conflict"], "parameters": {}}]


def _to_time(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _gen_meetings(rng, n, start_hour=9, end_hour=17):
    """Generate n non-overlapping meetings."""
    meetings = []
    cursor = start_hour * 60
    end = end_hour * 60
    for _ in range(n):
        gap = rng.randint(0, 60)
        cursor += gap
        if cursor >= end - 30:
            break
        duration = rng.choice([30, 45, 60, 90])
        meeting_end = min(cursor + duration, end)
        meetings.append([_to_time(cursor), _to_time(meeting_end)])
        cursor = meeting_end
    return meetings


def _find_free(busy, day_start, day_end, min_dur):
    start = int(day_start.split(":")[0]) * 60 + int(day_start.split(":")[1])
    end = int(day_end.split(":")[0]) * 60 + int(day_end.split(":")[1])
    busy_mins = sorted([(int(b[0].split(":")[0])*60+int(b[0].split(":")[1]),
                         int(b[1].split(":")[0])*60+int(b[1].split(":")[1])) for b in busy])
    merged = []
    for s, e in busy_mins:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    free = []
    cursor = start
    for s, e in merged:
        if s > cursor and (s - cursor) >= min_dur:
            free.append([_to_time(cursor), _to_time(s)])
        cursor = max(cursor, e)
    if end > cursor and (end - cursor) >= min_dur:
        free.append([_to_time(cursor), _to_time(end)])
    return free


def task_single(tid, difficulty, instruction, answer):
    return {
        "id": tid, "difficulty": difficulty, "multi_step": False,
        "instruction": instruction, "environments": schedule_env(),
        "expected": {"answer": str(answer)}, "tags": ["schedule", "single_round"],
    }


def gen_single_easy(start_id):
    rng = _rng(230)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = len(out) % 3
        n_meetings = rng.randint(2, 3)
        meetings = _gen_meetings(rng, n_meetings)
        key = str(meetings) + str(p)
        if key in seen:
            continue
        seen.add(key)
        if p == 0:
            free = _find_free(meetings, "09:00", "17:00", 60)
            ans = len(free)
            instr = (f"Given these meetings: {meetings}. "
                     f"How many free 1-hour slots are available between 09:00 and 17:00? "
                     f"Answer with only the number.")
        elif p == 1:
            free = _find_free(meetings, "09:00", "17:00", 30)
            ans = len(free)
            instr = (f"Given these meetings: {meetings}. "
                     f"How many free 30-minute slots are available between 09:00 and 17:00? "
                     f"Answer with only the number.")
        else:
            # Check conflict (easy: no conflicts since we generate non-overlapping)
            ans = False
            instr = (f"Do any of these meetings overlap? {meetings}. "
                     f"Answer with only true or false.")
        out.append(task_single(start_id + len(out), "easy", instr, ans))
    return out


def gen_single_medium(start_id):
    rng = _rng(340)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = len(out) % 3
        n_meetings = rng.randint(5, 8)
        meetings = _gen_meetings(rng, n_meetings)
        key = str(meetings) + str(p)
        if key in seen:
            continue
        seen.add(key)
        if p == 0:
            free = _find_free(meetings, "09:00", "17:00", 30)
            ans = str(free)
            instr = (f"Given these meetings: {meetings}. "
                     f"List all free slots of at least 30 minutes between 09:00 and 17:00. "
                     f"Answer with a list of [start, end] pairs in HH:MM format.")
        elif p == 1:
            free = _find_free(meetings, "09:00", "17:00", 60)
            ans = len(free)
            instr = (f"Given these meetings: {meetings}. "
                     f"How many free 1-hour slots are available between 09:00 and 17:00? "
                     f"Answer with only the number.")
        else:
            free = _find_free(meetings, "08:00", "18:00", 45)
            ans = len(free)
            instr = (f"Given these meetings: {meetings}. "
                     f"How many free 45-minute slots are available between 08:00 and 18:00? "
                     f"Answer with only the number.")
        out.append(task_single(start_id + len(out), "medium", instr, ans))
    return out


def gen_single_hard(start_id):
    rng = _rng(440)
    out, seen = [], set()
    while len(out) < N_TOTAL:
        p = len(out) % 3
        n_meetings = rng.randint(10, 15)
        meetings = _gen_meetings(rng, n_meetings, start_hour=8, end_hour=20)
        key = str(meetings) + str(p)
        if key in seen:
            continue
        seen.add(key)
        if p == 0:
            free = _find_free(meetings, "08:00", "20:00", 30)
            ans = str(free)
            instr = (f"Given these meetings: {meetings}. "
                     f"List all free slots of at least 30 minutes between 08:00 and 20:00. "
                     f"Answer with a list of [start, end] pairs in HH:MM format.")
        elif p == 1:
            free = _find_free(meetings, "08:00", "20:00", 60)
            ans = len(free)
            instr = (f"Given these meetings: {meetings}. "
                     f"How many free 1-hour slots are available between 08:00 and 20:00? "
                     f"Answer with only the number.")
        else:
            free = _find_free(meetings, "08:00", "20:00", 30)
            ans = len(free)
            instr = (f"Given these meetings: {meetings}. "
                     f"How many free 30-minute slots are available between 08:00 and 20:00? "
                     f"Answer with only the number.")
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
        "schedule_single_easy": gen_single_easy(5000),
        "schedule_single_medium": gen_single_medium(5100),
        "schedule_single_hard": gen_single_hard(5200),
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
