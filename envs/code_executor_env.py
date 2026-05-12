import io
import contextlib
from .base_env import BaseEnv


class CodeExecutorEnv(BaseEnv):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})

    def run_code(self, code):
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out (5s limit)")

        stdout = io.StringIO()
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        try:
            signal.alarm(5)  # 5 second timeout
            globs = {"__builtins__": __builtins__}
            with contextlib.redirect_stdout(stdout):
                exec(code, globs, globs)
            signal.alarm(0)
            output = stdout.getvalue().rstrip("\n")
            if len(output) > 5000:
                output = output[:5000] + f"... [truncated, total {len(output)} chars]"
            return {
                "success": True,
                "stdout": output,
                "stderr": "",
                "exit_code": 0,
                "result": output,
            }
        except TimeoutError as e:
            partial = stdout.getvalue().rstrip("\n")
            return {
                "success": False,
                "stdout": partial[:500] if partial else "",
                "stderr": str(e),
                "exit_code": -1,
                "message": str(e),
            }
        except Exception as e:
            partial = stdout.getvalue().rstrip("\n")
            return {
                "success": False,
                "stdout": partial[:500] if partial else "",
                "stderr": f"{type(e).__name__}: {str(e)}",
                "exit_code": 1,
                "message": f"{type(e).__name__}: {str(e)}",
            }
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
