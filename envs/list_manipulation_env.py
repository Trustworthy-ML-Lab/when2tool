from copy import deepcopy

from .base_env import BaseEnv

try:
    import numpy as np
except Exception:
    np = None


class ListManipulationEnv(BaseEnv):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.current_list = deepcopy(self.parameters.get("initial_list", []))
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})

    def _is_2d(self, arr):
        return isinstance(arr, list) and len(arr) > 0 and all(isinstance(x, list) for x in arr)

    def _normalize_list_input(self, arr):
        # Accept malformed shape produced by some models:
        # [{"values": [...]}, ...] -> [[...], ...]
        if not isinstance(arr, list):
            return arr
        if len(arr) == 0:
            return arr
        if all(isinstance(x, dict) and isinstance(x.get("values"), list) for x in arr):
            return [deepcopy(x["values"]) for x in arr]
        return arr

    def _validate_axis(self, axis):
        if axis not in (0, 1):
            return False, "axis must be 0 or 1 for 2D operations"
        return True, "ok"

    def _apply_1d_op(self, arr, op_name, value=None, index=None):
        if op_name == "append":
            arr.append(value)
            return True, "ok"

        if op_name == "insert":
            if index is None:
                return False, "index is required for insert"
            if not isinstance(index, int):
                return False, "index must be integer for insert"
            if index < 0 or index > len(arr):
                return False, f"index {index} out of range"
            arr.insert(index, value)
            return True, "ok"

        if op_name == "remove":
            try:
                arr.remove(value)
                return True, "ok"
            except ValueError:
                return False, f"value {value} not found"

        if op_name == "sort":
            try:
                arr.sort()
            except TypeError:
                return False, "sort failed: list contains non-comparable element types"
            return True, "ok"

        if op_name == "reverse":
            arr.reverse()
            return True, "ok"

        return False, f"unsupported op: {op_name}"

    def _apply_2d_op(self, arr, op_name, axis=None, value=None, index=None):
        ok, msg = self._validate_axis(axis)
        if not ok:
            return False, msg

        # axis=0 means operate along columns; axis=1 means operate within each row.
        if op_name == "sort":
            if len(arr) == 0 or len(arr[0]) == 0:
                return True, "ok"

            # Fast path for larger matrices.
            if np is not None:
                try:
                    arr_np = np.asarray(arr)
                    sorted_np = np.sort(arr_np, axis=axis)
                    arr[:] = sorted_np.tolist()
                    return True, "ok"
                except Exception:
                    # Fall back to pure-Python implementation below.
                    pass

            if axis == 0:
                # NumPy-like semantics: sort each column independently.
                n_rows = len(arr)
                n_cols = len(arr[0])
                for c in range(n_cols):
                    col_sorted = sorted(arr[r][c] for r in range(n_rows))
                    for r in range(n_rows):
                        arr[r][c] = col_sorted[r]
            else:
                for row in arr:
                    row.sort()
            return True, "ok"

        if op_name == "reverse":
            if axis == 0:
                # NumPy-like semantics: flip along axis 0 (reverse row order).
                arr.reverse()
            else:
                # NumPy-like semantics: flip along axis 1 (reverse each row).
                for row in arr:
                    row.reverse()
            return True, "ok"

        if op_name == "remove":
            if index is None or not isinstance(index, int):
                return False, "index(integer) is required for 2D remove"

            if axis == 0:
                if index < 0 or index >= len(arr):
                    return False, f"row index {index} out of range"
                arr.pop(index)
                return True, "ok"

            # axis==1
            for row in arr:
                if index < 0 or index >= len(row):
                    return False, f"column index {index} out of range"
            for row in arr:
                row.pop(index)
            return True, "ok"

        if op_name == "insert":
            if index is None or not isinstance(index, int):
                return False, "index(integer) is required for 2D insert"

            if axis == 0:
                if not isinstance(value, list):
                    return False, "for axis=0 insert, value must be a row list"
                if len(arr) > 0 and len(value) != len(arr[0]):
                    return False, "inserted row length mismatch"
                if index < 0 or index > len(arr):
                    return False, f"row index {index} out of range"
                arr.insert(index, value)
                return True, "ok"

            # axis==1
            if index < 0 or (len(arr) > 0 and index > len(arr[0])):
                return False, f"column index {index} out of range"
            for row in arr:
                row.insert(index, value)
            return True, "ok"

        if op_name == "append":
            if axis == 0:
                if not isinstance(value, list):
                    return False, "for axis=0 append, value must be a row list"
                if len(arr) > 0 and len(value) != len(arr[0]):
                    return False, "appended row length mismatch"
                arr.append(value)
                return True, "ok"

            # axis==1 append value to each row
            for row in arr:
                row.append(value)
            return True, "ok"

        return False, f"unsupported op: {op_name}"

    def _apply_op(self, arr, op_obj):
        op_name = op_obj.get("op")
        value = op_obj.get("value")
        index = op_obj.get("index")
        axis = op_obj.get("axis")

        # Best-effort normalization for malformed tool arguments.
        arr[:] = self._normalize_list_input(arr)

        if self._is_2d(arr):
            return self._apply_2d_op(arr, op_name, axis=axis, value=value, index=index)
        if axis is not None:
            return False, "axis is only valid for 2D lists"
        return self._apply_1d_op(arr, op_name, value=value, index=index)

    def _run_single(self, values, op, value=None, index=None, axis=None):
        arr = self._normalize_list_input(deepcopy(values))
        op_obj = {"op": op}
        if value is not None:
            op_obj["value"] = value
        if index is not None:
            op_obj["index"] = index
        if axis is not None:
            op_obj["axis"] = axis

        success, msg = self._apply_op(arr, op_obj)
        return {
            "success": success,
            "message": msg,
            "data": {"current_list": deepcopy(arr)},
        }

    # Stateless per-op tools
    def append(self, values, value, axis=None):
        return self._run_single(values, "append", value=value, axis=axis)

    def remove(self, values, value=None, index=None, axis=None):
        return self._run_single(values, "remove", value=value, index=index, axis=axis)

    def insert(self, values, index, value, axis=None):
        return self._run_single(values, "insert", value=value, index=index, axis=axis)

    def sort(self, values, axis=None):
        return self._run_single(values, "sort", axis=axis)

    def reverse(self, values, axis=None):
        return self._run_single(values, "reverse", axis=axis)
