import math
from .base_env import BaseEnv


class PrimeEnv(BaseEnv):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})

    def _is_prime(self, n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def is_prime(self, n):
        return {"success": True, "result": self._is_prime(n)}

    def nth_prime(self, n):
        if n < 1:
            return {"success": False, "message": "n must be >= 1."}
        count = 0
        candidate = 1
        while count < n:
            candidate += 1
            if self._is_prime(candidate):
                count += 1
        return {"success": True, "result": candidate}

    def factorize(self, n):
        if n < 2:
            return {"success": False, "message": "n must be >= 2."}
        factors = []
        d = 2
        temp = n
        while d * d <= temp:
            exp = 0
            while temp % d == 0:
                exp += 1
                temp //= d
            if exp > 0:
                factors.append([d, exp])
            d += 1
        if temp > 1:
            factors.append([temp, 1])
        return {"success": True, "result": factors}
