import hashlib
from .base_env import BaseEnv


# 5 custom hash algorithms with different constants — impossible for any model to know.
_CUSTOM_HASHES = {
    "fnv1a_custom": {
        "offset": 0xCBF29CE484222325 ^ 0xDEADBEEF42,
        "prime": 0x100000001B3 + 38,
    },
    "djb2_custom": {
        "offset": 5381 ^ 0xABCD1234,
        "prime": 33 + 14,
    },
    "sdbm_custom": {
        "offset": 0x1F2E3D4C5B6A7980,
        "prime": 65599 + 42,
    },
    "murmur_custom": {
        "offset": 0x5F3759DF ^ 0xCAFEBABE,
        "prime": 0x100000001B3 + 97,
    },
    "jenkins_custom": {
        "offset": 0x7A6B5C4D3E2F1001,
        "prime": 6364136223846793005 + 13,
    },
}
_CUSTOM_MOD = 2**64


class HashEnv(BaseEnv):

    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})

    @staticmethod
    def _custom_hash(s, algo_name):
        cfg = _CUSTOM_HASHES[algo_name]
        h = cfg["offset"]
        for byte in s.encode("utf-8"):
            h ^= byte
            h = (h * cfg["prime"]) % _CUSTOM_MOD
        return format(h, "016x")

    def compute_hash(self, input_string, algorithm):
        algo = algorithm.strip().lower()
        if algo == "md5":
            result = hashlib.md5(input_string.encode("utf-8")).hexdigest()
        elif algo == "sha256":
            result = hashlib.sha256(input_string.encode("utf-8")).hexdigest()
        elif algo == "sha1":
            result = hashlib.sha1(input_string.encode("utf-8")).hexdigest()
        elif algo in _CUSTOM_HASHES:
            result = self._custom_hash(input_string, algo)
        else:
            return {"success": False, "message": f"Unknown algorithm: {algorithm}"}
        display_input = input_string if len(input_string) <= 200 else input_string[:200] + "..."
        return {"success": True, "algorithm": algo, "input": display_input, "hash": result}
