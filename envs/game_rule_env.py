from .base_env import BaseEnv


class GameRuleEnv(BaseEnv):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})
        self._corpus = self.parameters.get("corpus", {})

    def lookup_rule(self, query):
        query_lower = query.strip().lower()
        for key, passage in self._corpus.items():
            if key.lower() == query_lower or query_lower in key.lower() or key.lower() in query_lower:
                return {
                    "success": True,
                    "query": key,
                    "passage": passage,
                    "source": "game_rules_database",
                    "confidence": "exact_match",
                    "related_rules": list(self._corpus.keys())[:5],
                }
        return {"success": False, "message": f"Rule not found: {query}", "query": query, "available_entries": len(self._corpus)}
