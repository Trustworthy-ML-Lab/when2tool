from .base_env import BaseEnv


def _to_minutes(time_str):
    h, m = time_str.strip().split(":")
    return int(h) * 60 + int(m)


def _to_time(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


class ScheduleEnv(BaseEnv):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})

    def find_free_slots(self, busy, day_start, day_end, min_duration):
        start = _to_minutes(day_start)
        end = _to_minutes(day_end)
        busy_mins = sorted([(_to_minutes(b[0]), _to_minutes(b[1])) for b in busy])

        # Merge overlapping busy intervals
        merged = []
        for s, e in busy_mins:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        free = []
        cursor = start
        for s, e in merged:
            if s > cursor and (s - cursor) >= min_duration:
                free.append([_to_time(cursor), _to_time(s)])
            cursor = max(cursor, e)
        if end > cursor and (end - cursor) >= min_duration:
            free.append([_to_time(cursor), _to_time(end)])

        return {
            "success": True,
            "result": free,
            "total_free_slots": len(free),
            "search_window": [day_start, day_end],
            "min_duration_minutes": min_duration,
            "busy_intervals": len(busy),
            "merged_busy_intervals": len(merged),
        }

    def check_conflict(self, intervals):
        mins = [(_to_minutes(i[0]), _to_minutes(i[1])) for i in intervals]
        conflicts = []
        for i in range(len(mins)):
            for j in range(i + 1, len(mins)):
                s1, e1 = mins[i]
                s2, e2 = mins[j]
                if s1 < e2 and s2 < e1:
                    conflicts.append([intervals[i], intervals[j]])
        has_conflict = len(conflicts) > 0
        return {
            "success": True,
            "result": {"has_conflict": has_conflict, "conflicts": conflicts},
            "total_intervals": len(intervals),
            "total_conflicts": len(conflicts),
        }
