import re
from .base_env import BaseEnv


class RegexMatchEnv(BaseEnv):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})

    def regex_match(self, pattern, text, operation=None):
        op = (operation or "findall").strip().lower()
        try:
            if op == "findall":
                result = re.findall(pattern, text)
                return {"success": True, "result": result, "match_count": len(result), "pattern": pattern, "text_length": len(text)}
            elif op == "search":
                m = re.search(pattern, text)
                if m is None:
                    return {"success": True, "result": None}
                return {"success": True, "result": m.group(), "groups": list(m.groups()), "span": list(m.span())}
            elif op == "match":
                m = re.match(pattern, text)
                if m is None:
                    return {"success": True, "result": None}
                return {"success": True, "result": m.group(), "groups": list(m.groups()), "span": list(m.span())}
            elif op == "sub":
                # For sub, pattern should contain replacement after a separator
                # Use format: "pattern|||replacement"
                if "|||" in pattern:
                    pat, repl = pattern.split("|||", 1)
                    result = re.sub(pat, repl, text)
                    return {"success": True, "result": result}
                return {"success": False, "message": "For sub operation, use pattern format: 'regex|||replacement'"}
            else:
                return {"success": False, "message": f"Unknown operation: {operation}. Use findall, search, match, or sub."}
        except re.error as e:
            return {"success": False, "message": f"Regex error: {str(e)}"}
