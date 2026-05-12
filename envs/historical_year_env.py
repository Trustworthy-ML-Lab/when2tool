from .base_env import BaseEnv


class HistoricalYearEnv(BaseEnv):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})
        self._corpus = self.parameters.get("corpus", {})

    def lookup_year(self, event):
        event_lower = event.strip().lower()
        for key, passage in self._corpus.items():
            if key.lower() == event_lower or event_lower in key.lower() or key.lower() in event_lower:
                return {
                    "success": True,
                    "event": key,
                    "passage": passage,
                    "query": event,
                    "source": "historical_records",
                    "confidence": "exact_match",
                    "related_entries": list(self._corpus.keys())[:5],
                }
        return {"success": False, "message": f"Event not found: {event}", "query": event, "available_entries": len(self._corpus)}
