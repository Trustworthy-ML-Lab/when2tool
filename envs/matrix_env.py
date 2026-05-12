from .base_env import BaseEnv


class MatrixEnv(BaseEnv):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})

    def _det(self, m):
        n = len(m)
        if n == 1:
            return m[0][0]
        if n == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        det = 0
        for c in range(n):
            sub = [[m[r][j] for j in range(n) if j != c] for r in range(1, n)]
            det += ((-1) ** c) * m[0][c] * self._det(sub)
        return det

    def matrix_determinant(self, matrix):
        n = len(matrix)
        if n == 0:
            return {"success": False, "message": "Empty matrix."}
        for row in matrix:
            if len(row) != n:
                return {"success": False, "message": "Matrix must be square."}
        result = self._det(matrix)
        return {"success": True, "result": result, "matrix_size": f"{n}x{n}", "method": "cofactor_expansion"}

    def matrix_multiply(self, matrix_a, matrix_b):
        ra, ca = len(matrix_a), len(matrix_a[0]) if matrix_a else 0
        rb, cb = len(matrix_b), len(matrix_b[0]) if matrix_b else 0
        if ca != rb:
            return {"success": False, "message": f"Incompatible dimensions: {ra}x{ca} and {rb}x{cb}."}
        result = []
        for i in range(ra):
            row = []
            for j in range(cb):
                val = sum(matrix_a[i][k] * matrix_b[k][j] for k in range(ca))
                row.append(val)
            result.append(row)
        return {"success": True, "result": result, "dimensions": f"{ra}x{ca} * {rb}x{cb} = {ra}x{cb}"}

    def matrix_trace(self, matrix):
        n = len(matrix)
        for row in matrix:
            if len(row) != n:
                return {"success": False, "message": "Matrix must be square."}
        result = sum(matrix[i][i] for i in range(n))
        diag = [matrix[i][i] for i in range(n)]
        return {"success": True, "result": result, "diagonal_elements": diag, "matrix_size": f"{n}x{n}"}
