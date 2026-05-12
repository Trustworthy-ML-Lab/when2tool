import ast
import operator as op

from .base_env import BaseEnv


_ALLOWED = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


class CalculatorEnv(BaseEnv):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})
        self.last_result = self.parameters.get("last_result", None)

    def _eval_node(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
            return _ALLOWED[type(node.op)](self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED:
            return _ALLOWED[type(node.op)](self._eval_node(node.operand))
        raise ValueError("Unsupported expression")

    def evaluate_expression(self, expression):
        expr = ast.parse(expression, mode="eval")
        value = self._eval_node(expr.body)
        result_str = str(value)
        if len(result_str) > 5000:
            return {"success": False, "message": f"Result too large ({len(result_str)} digits). Simplify the expression."}
        self.last_result = value
        return {
            "success": True,
            "result": value,
            "expression": expression,
        }

    def get_last_result(self):
        if self.last_result is None:
            return {"success": False, "message": "No previous result available."}
        return {"success": True, "data": {"last_result": self.last_result}}

    def clear_last_result(self):
        self.last_result = None
        return {"success": True}
