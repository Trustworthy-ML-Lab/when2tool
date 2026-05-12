from datetime import datetime, timedelta
from .base_env import BaseEnv

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class DateTimeEnv(BaseEnv):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})

    def _parse_date(self, date_str):
        try:
            return datetime.strptime(date_str.strip(), "%Y-%m-%d")
        except ValueError:
            return None

    def date_diff(self, date1, date2):
        d1 = self._parse_date(date1)
        d2 = self._parse_date(date2)
        if d1 is None or d2 is None:
            return {"success": False, "message": "Invalid date format. Use YYYY-MM-DD."}
        diff = abs((d2 - d1).days)
        return {"success": True, "result": diff}

    def date_add(self, date, days):
        d = self._parse_date(date)
        if d is None:
            return {"success": False, "message": "Invalid date format. Use YYYY-MM-DD."}
        result = d + timedelta(days=days)
        return {"success": True, "result": result.strftime("%Y-%m-%d")}

    def day_of_week(self, date):
        d = self._parse_date(date)
        if d is None:
            return {"success": False, "message": "Invalid date format. Use YYYY-MM-DD."}
        return {"success": True, "result": _DAY_NAMES[d.weekday()]}
