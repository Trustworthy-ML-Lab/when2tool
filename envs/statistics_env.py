import math
from .base_env import BaseEnv


class StatisticsEnv(BaseEnv):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})

    def compute_stat(self, data, stat_type, data2=None, percentile_value=None):
        if not data:
            return {"success": False, "message": "Data list is empty."}
        data = [float(x) for x in data]
        n = len(data)

        if stat_type == "mean":
            result = sum(data) / n
        elif stat_type == "median":
            s = sorted(data)
            if n % 2 == 1:
                result = s[n // 2]
            else:
                result = (s[n // 2 - 1] + s[n // 2]) / 2
        elif stat_type == "std":
            mean = sum(data) / n
            result = math.sqrt(sum((x - mean) ** 2 for x in data) / n)
        elif stat_type == "variance":
            mean = sum(data) / n
            result = sum((x - mean) ** 2 for x in data) / n
        elif stat_type == "min":
            result = min(data)
        elif stat_type == "max":
            result = max(data)
        elif stat_type == "sum":
            result = sum(data)
        elif stat_type == "percentile":
            if percentile_value is None:
                return {"success": False, "message": "percentile_value is required for stat_type=percentile."}
            s = sorted(data)
            k = (percentile_value / 100) * (n - 1)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                result = s[int(k)]
            else:
                result = s[f] + (k - f) * (s[c] - s[f])
        elif stat_type == "correlation":
            if data2 is None:
                return {"success": False, "message": "data2 is required for stat_type=correlation."}
            data2 = [float(x) for x in data2]
            if len(data2) != n:
                return {"success": False, "message": "data and data2 must have the same length."}
            mean_x = sum(data) / n
            mean_y = sum(data2) / n
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(data, data2)) / n
            std_x = math.sqrt(sum((x - mean_x) ** 2 for x in data) / n)
            std_y = math.sqrt(sum((y - mean_y) ** 2 for y in data2) / n)
            if std_x == 0 or std_y == 0:
                return {"success": False, "message": "Standard deviation is zero, correlation undefined."}
            result = cov / (std_x * std_y)
        else:
            return {"success": False, "message": f"Unknown stat_type: {stat_type}"}

        s = sorted(data)
        return {
            "success": True,
            "result": round(result, 10),
            "stat_type": stat_type,
            "n": n,
            "input_summary": {"min": s[0], "max": s[-1], "sum": round(sum(data), 10), "mean": round(sum(data) / n, 10)},
        }

    def describe(self, data):
        if not data:
            return {"success": False, "message": "Data list is empty."}
        data = [float(x) for x in data]
        n = len(data)
        s = sorted(data)
        mean = sum(data) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in data) / n)

        def percentile(sorted_data, p):
            k = (p / 100) * (len(sorted_data) - 1)
            f, c = math.floor(k), math.ceil(k)
            if f == c:
                return sorted_data[int(k)]
            return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

        return {
            "success": True,
            "result": {
                "count": n,
                "mean": round(mean, 10),
                "std": round(std, 10),
                "min": s[0],
                "25%": round(percentile(s, 25), 10),
                "50%": round(percentile(s, 50), 10),
                "75%": round(percentile(s, 75), 10),
                "max": s[-1],
            },
        }
