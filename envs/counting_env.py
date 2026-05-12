import math
from .base_env import BaseEnv


class CountingEnv(BaseEnv):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})

    def _check_result_size(self, result):
        if len(str(result)) > 5000:
            return {"success": False, "message": f"Result too large ({len(str(result))} digits). Use smaller inputs."}
        return {"success": True, "result": result}

    def combination(self, n, k):
        if k < 0 or k > n:
            return {"success": False, "message": f"Invalid: k={k} must be between 0 and n={n}."}
        return self._check_result_size(math.comb(n, k))

    def permutation(self, n, k):
        if k < 0 or k > n:
            return {"success": False, "message": f"Invalid: k={k} must be between 0 and n={n}."}
        return self._check_result_size(math.perm(n, k))

    def factorial(self, n):
        if n < 0:
            return {"success": False, "message": "n must be non-negative."}
        return self._check_result_size(math.factorial(n))
