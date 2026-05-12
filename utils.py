import ast
import json
import os
import re
import sys
from copy import deepcopy
from typing import Dict, Tuple

# Python 3.12+ limits integer-to-string conversion; large combinatorics results need higher limit
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(10000)

from envs import ENV_REGISTRY


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

MODEL_NAME_MAP: Dict[str, str] = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
}


def normalize_model_key(name: str) -> str:
    return str(name or "").strip().lower().replace("_", "-")


def resolve_model_path(name_or_path: str) -> Tuple[str, bool]:
    key = normalize_model_key(name_or_path)
    if key in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[key], True
    return str(name_or_path), False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_XML = (
    "You can use tools when helpful. "
    "If you call a tool, emit exactly one tool call wrapped in <tool_call>...</tool_call>, "
    "with exactly one JSON object inside: {\"name\": \"tool_name\", \"arguments\": {...}}. "
    "Do not call multiple tools at once. "
    "Tool arguments must strictly match each tool schema; do not invent wrapper fields. "
    "When you are done, provide the final answer in LaTeX boxed format: \\boxed{...}."
)

SYSTEM_PROMPT_NATIVE = (
    "You can use tools when helpful. "
    "Do not call multiple tools at once. "
    "Tool arguments must strictly match each tool schema; do not invent wrapper fields. "
    "When you are done, provide the final answer in LaTeX boxed format: \\boxed{...}."
)

# Default (xml) for backward compatibility
SYSTEM_PROMPT = SYSTEM_PROMPT_XML


def detect_tool_format(model_name_or_path: str) -> str:
    """Detect whether the model uses xml (<tool_call> tags) or native tool calling.

    - xml: tool calls as <tool_call>JSON</tool_call> in text, responses as user role.
           Used by Qwen, Mistral, and models without native function calling.
    - native: bare JSON tool calls, responses as tool/ipython role.
           Used by Llama 3.1+ which has built-in function calling via chat template.
    """
    key = normalize_model_key(model_name_or_path)
    # Llama 3.1+ uses native function calling (bare JSON + ipython/tool role)
    if "llama" in key:
        return "native"
    # Everything else (Qwen, Mistral, Gemma, etc.) uses xml-style text tool calls
    return "xml"


def get_system_prompt(tool_format: str = "xml") -> str:
    if tool_format == "native":
        return SYSTEM_PROMPT_NATIVE
    return SYSTEM_PROMPT_XML

PROMPT_MODES = ("current", "necessary_tool", "sparse_tool", "force_tool", "no_tool")
# hard_no_tool is not a user-facing eval mode; it's used internally by extract_features
# to collect ground-truth labels. It uses the "current" prompt but rejects all tool calls.
LABEL_MODES = ("hard_no_tool",)
REASONING_MODES = ("reasoning", "no_reasoning")

# Shared reasoning instruction: assess tool necessity first, then act accordingly.
_REASONING_POLICY = (
    "1) First, use few sentences to evaluate whether you need to use a tool or can solve this directly. "
    "If you can solve directly, do so without using a tool. Otherwise, use the tool. Then proceed.\n"
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_user_message(task_instruction: str, prompt_mode: str, require_reasoning: bool = True):
    # hard_no_tool: same prompt as "no_tool", plus tool calls rejected in step_state.
    if prompt_mode == "hard_no_tool":
        prompt_mode = "no_tool"
    if prompt_mode == "force_tool":
        if require_reasoning:
            return (
                task_instruction
                + "\n\nResponse policy (required every turn):\n"
                + _REASONING_POLICY
                + "2) Tool use is mandatory in this task.\n"
                + "3) Provide final answer in \\boxed{...} if you think the task is complete."
            )
        return (
            task_instruction
            + "\n\nResponse policy (required every turn):\n"
            + "1) Tool use is mandatory in this task.\n"
            + "2) Provide final answer in \\boxed{...} if you think the task is complete."
        )
    if prompt_mode == "no_tool":
        if require_reasoning:
            return (
                task_instruction
                + "\n\nResponse policy (required every turn):\n"
                + _REASONING_POLICY
                + "2) Do not use any tools in this task.\n"
                + "3) Provide final answer in \\boxed{...} if you think the task is complete."
            )
        return (
            task_instruction
            + "\n\nResponse policy (required every turn):\n"
            + "1) Do not use any tools in this task.\n"
            + "2) Provide final answer in \\boxed{...} if you think the task is complete."
        )
    if prompt_mode == "necessary_tool":
        if require_reasoning:
            return (
                task_instruction
                + "\n\nResponse policy (required every turn):\n"
                + _REASONING_POLICY
                + "2) Use tools only if necessary in this task.\n"
                + "3) Provide final answer in \\boxed{...} if you think the task is complete."
            )
        return (
            task_instruction
            + "\n\nResponse policy (required every turn):\n"
            + "1) Use tools only if necessary in this task.\n"
            + "2) Provide final answer in \\boxed{...} if you think the task is complete."
        )
    if prompt_mode == "sparse_tool":
        if require_reasoning:
            return (
                task_instruction
                + "\n\nResponse policy (required every turn):\n"
                + _REASONING_POLICY
                + "2) Tool calls are expensive in this mode; default to zero tool calls.\n"
                + "3) Try your best to solve directly first and finalize without tools whenever possible.\n"
                + "4) Provide final answer in \\boxed{...} if you think the task is complete."
            )
        return (
            task_instruction
            + "\n\nResponse policy (required every turn):\n"
            + "1) Tool calls are expensive in this mode; default to zero tool calls.\n"
            + "2) Try your best to solve directly first and finalize without tools whenever possible.\n"
            + "3) Provide final answer in \\boxed{...} if you think the task is complete."
        )
    # default: "current"
    if require_reasoning:
        return (
            task_instruction
            + "\n\nResponse policy (required every turn):\n"
            + _REASONING_POLICY
            + "2) You can choose to use a tool or not in this task.\n"
            + "3) Provide final answer in \\boxed{...} if you think the task is complete."
        )
    return (
        task_instruction
        + "\n\nResponse policy (required every turn):\n"
        + "1) You can choose to use a tool or not in this task.\n"
        + "2) Provide final answer in \\boxed{...} if you think the task is complete."
    )


# ---------------------------------------------------------------------------
# Tool schema construction
# ---------------------------------------------------------------------------

def build_tools_schema(task):
    """Build the tool schema list for a task (for chat template injection)."""
    tools = []
    for env_cfg in task.get("environments", []):
        name = env_cfg.get("name", "")
        if not name or name not in ENV_REGISTRY:
            continue
        params = deepcopy(env_cfg.get("parameters", {}))
        env = ENV_REGISTRY[name](parameters=params)
        allowed_tools = set(env_cfg.get("tools", []))
        for schema in env.get_tool_descs(sorted(allowed_tools)):
            tools.append({"type": "function", "function": schema})
    return tools


# ---------------------------------------------------------------------------
# Answer scoring
# ---------------------------------------------------------------------------

def clean(s):
    if s is None:
        return ""
    return str(s).strip()


def extract_boxed(text):
    t = clean(text)
    if not t:
        return ""
    marker = "\\boxed{"
    idx = t.find(marker)
    if idx < 0:
        return ""
    i = idx + len(marker)
    depth = 1
    out = []
    while i < len(t):
        ch = t[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip() if out else t


def normalize_scalar(s):
    t = re.sub(r"\s+", " ", clean(s)).strip()
    if len(t) >= 2 and ((t[0] == "'" and t[-1] == "'") or (t[0] == '"' and t[-1] == '"')):
        t = t[1:-1].strip()
    t = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", t)
    t = t.replace("{", "").replace("}", "")
    t = t.replace("\\", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()


def normalize_structured(x):
    if isinstance(x, tuple):
        x = list(x)
    if isinstance(x, list):
        return [normalize_structured(v) for v in x]
    if isinstance(x, dict):
        return {k: normalize_structured(v) for k, v in x.items()}
    if isinstance(x, str):
        s = x.strip()
        if re.fullmatch(r"[-+]?\d+", s):
            try:
                return int(s)
            except Exception:
                return s
        if re.fullmatch(r"[-+]?\d*\.\d+", s):
            try:
                return float(s)
            except Exception:
                return s
        return s
    return x


def parse_structured(text):
    t = clean(text)
    if not t:
        return None
    try:
        return normalize_structured(ast.literal_eval(t))
    except Exception:
        return None


def compare_values(pred, gold):
    p_struct = parse_structured(pred)
    g_struct = parse_structured(gold)
    if p_struct is not None and g_struct is not None:
        return p_struct == g_struct
    p = normalize_scalar(pred)
    g = normalize_scalar(gold)
    if not g:
        return False
    return p == g


# ---------------------------------------------------------------------------
# Item-level eval helpers (operate on finalized output dicts)
# ---------------------------------------------------------------------------

def last_assistant_text(item):
    if isinstance(item.get("final_response"), str) and item.get("final_response"):
        return item["final_response"]
    out = item.get("output", [])
    if not isinstance(out, list):
        return ""
    for m in reversed(out):
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def item_final_eval(item):
    """Evaluate a finalized output item. Returns (raw, boxed, cleaned, correct)."""
    expected = item.get("expected", {}) or {}
    gold_final = clean(expected.get("answer", ""))
    model_final_raw = last_assistant_text(item)
    model_final_box = extract_boxed(model_final_raw)
    model_final = clean(model_final_box)
    final_correct = compare_values(model_final, gold_final)
    return model_final_raw, model_final_box, model_final, final_correct


def item_tool_calls(item):
    if "tool_calls" in item:
        try:
            return max(0, int(item.get("tool_calls", 0)))
        except Exception:
            return 0
    trace = item.get("trace", [])
    if isinstance(trace, list):
        n = 0
        for t in trace:
            p = (t or {}).get("parsed_output", {})
            if (p or {}).get("type") == "tool":
                n += 1
        return n
    return 0


def item_expected_steps(item):
    expected = item.get("expected", {}) or {}
    steps = expected.get("steps")
    if isinstance(steps, list) and len(steps) > 0:
        return len(steps)
    return 3 if bool(item.get("multi_step")) else 1


# ---------------------------------------------------------------------------
# Eval engine: reasoning checks
# ---------------------------------------------------------------------------

def has_boxed_answer(text):
    t = str(text or "")
    return "\\boxed{" in t


def _clean_reasoning_text(s):
    txt = str(s or "")
    txt = re.sub(r"\\boxed\s*\{[\s\S]*?\}", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _has_nontrivial_reasoning(s):
    txt = _clean_reasoning_text(s)
    if len(txt) < 12:
        return False
    return re.search(r"[A-Za-z]", txt) is not None


def has_reasoning_before_tool(raw_text):
    text = str(raw_text or "")
    if not text.strip():
        return False
    idx = text.find("<tool_call>")
    if idx >= 0:
        return _has_nontrivial_reasoning(text[:idx])
    first_json = text.find("{")
    if first_json <= 0:
        return False
    return _has_nontrivial_reasoning(text[:first_json])


def has_reasoning_for_direct_answer(raw_text):
    text = str(raw_text or "")
    idx = text.find("\\boxed{")
    if idx >= 0:
        return _has_nontrivial_reasoning(text[:idx])
    return _has_nontrivial_reasoning(text)


# ---------------------------------------------------------------------------
# Eval engine: env interaction
# ---------------------------------------------------------------------------

def build_envs(task, include_metadata=False):
    """Build environment instances and tool schemas for a task."""
    envs = []
    tools = []
    for env_cfg in task.get("environments", []):
        name = env_cfg.get("name", "")
        if not name:
            continue
        if name not in ENV_REGISTRY:
            raise ValueError(f"Unsupported environment: {name}")
        params = deepcopy(env_cfg.get("parameters", {}))
        env = ENV_REGISTRY[name](parameters=params)
        env.include_metadata = include_metadata
        allowed_tools = set(env_cfg.get("tools", []))
        if name == "ListManipulationEnv":
            allowed_tools = allowed_tools.intersection({"append", "remove", "insert", "sort", "reverse"})
        envs.append((env, allowed_tools))
        for schema in env.get_tool_descs(sorted(allowed_tools)):
            tools.append({"type": "function", "function": schema})
    return envs, tools


def route_tool_call(envs, tool_name, arguments):
    for env, allowed in envs:
        if tool_name not in allowed:
            continue
        if not env.has_tool(tool_name):
            continue
        return env.call_tool(tool_name, arguments)
    return {"success": False, "message": f"Tool {tool_name} not available in this task."}


# ---------------------------------------------------------------------------
# Eval engine: state machine
# ---------------------------------------------------------------------------

def init_state(task, system_prompt, record_mode="lite", prompt_mode="current", require_reasoning=True, tool_format="xml", tokenizer=None):
    envs, tools = build_envs(task)
    messages = [{"role": "system", "content": system_prompt}]
    env_names = {e.get("name", "") for e in task.get("environments", []) if e.get("name", "")}

    if "ListManipulationEnv" in env_names:
        messages.append(
            {
                "role": "system",
                "content": (
                    "ListManipulation format contract:\n"
                    "1) There is no set_list tool. For every list-op call, you must provide values=<current list> explicitly.\n"
                    "2) For tool arguments, use plain lists [a,b,c] (1D) or [[...],[...]] (2D). Never use objects like {\"values\": [...]}.\n"
                    "3) At each step, parse current_list from the latest tool output, then call exactly one operation tool (append/remove/insert/sort/reverse).\n"
                    "4) Use one operation per tool call; do not batch multiple operations in one call.\n"
                    "5) For final answer, output exactly one plain list literal wrapped in box, e.g. \\boxed{[1, 2, 3]} or \\boxed{[[1, 2], [3, 4]]}; do NOT format it as LaTeX array/matrix.\n"
                ),
            }
        )

    if "dialog" in task:
        messages.extend(task["dialog"])
    else:
        messages.append(
            {
                "role": "user",
                "content": build_user_message(task["instruction"], prompt_mode, require_reasoning=require_reasoning),
            }
        )

    return {
        "task": deepcopy(task),
        "envs": envs,
        "tools": tools,
        "messages": messages,
        "rounds": 0,
        "done": False,
        "trace": [],
        "record_mode": record_mode,
        "final_response": "",
        "prompt_mode": prompt_mode,
        "require_reasoning": bool(require_reasoning),
        "tool_calls_used": 0,
        "tool_format": tool_format,
        "tokenizer": tokenizer,
        "generation_tokens": 0,
        "prefill_tokens": 0,
    }


def _count_tokens(state, text):
    """Count tokens using the tokenizer stored in state, or 0 if unavailable."""
    tok = state.get("tokenizer")
    if tok is None or not text:
        return 0
    return len(tok.encode(text, add_special_tokens=False))


def _append_and_track(state, role, content):
    """Append a user/tool message and count its tokens as prefill cost."""
    state["messages"].append({"role": role, "content": content})
    state["prefill_tokens"] += _count_tokens(state, content)


def step_state(state, out, trace_item=None):
    state["rounds"] += 1
    raw_text = out.get("raw_text", out.get("content", ""))
    state["messages"].append({"role": "assistant", "content": raw_text})
    # Count generation tokens (model output)
    state["generation_tokens"] += _count_tokens(state, raw_text)
    prompt_mode = state.get("prompt_mode", "current")
    tool_format = state.get("tool_format", "xml")
    require_reasoning = bool(state.get("require_reasoning", True))
    # Skip ALL reasoning enforcement for prefilled tasks — the prefill text
    # looks like reasoning but was injected by the probe, not the model.
    # Setting both flags ensures neither "missing reasoning" nor "reasoning
    # not allowed" rejections fire.
    skip_reasoning_check = bool(state.get("has_prefill"))

    if out["type"] == "tool":
        # hard_no_tool: reject any tool call and force direct solving
        if prompt_mode == "hard_no_tool":
            _append_and_track(state, "user",
                "Tool use is not available. "
                "Solve the problem directly without tools and provide your final answer in \\boxed{...}."
            )
            if trace_item is not None:
                trace_item["tool_result"] = {
                    "success": False,
                    "message": "Tool call rejected: hard_no_tool mode.",
                }
                trace_item["state_done_after_step"] = False
                state["trace"].append(trace_item)
            return

        if not skip_reasoning_check and not state.get("_last_tool_success"):
            has_reasoning = has_reasoning_before_tool(out.get("raw_text", ""))
            if require_reasoning and (not has_reasoning):
                _append_and_track(state, "user",
                    "Tool call rejected: missing reasoning before tool call. "
                    "Retry by first giving a short reasoning, then provide the tool call."
                )
                if trace_item is not None:
                    trace_item["tool_result"] = {
                        "success": False,
                        "message": "Tool call rejected: missing reasoning before tool call.",
                    }
                    trace_item["state_done_after_step"] = False
                    state["trace"].append(trace_item)
                return
            if (not require_reasoning) and has_reasoning:
                _append_and_track(state, "user",
                    "Tool call rejected: reasoning is not allowed in no_reasoning mode. "
                    "Retry with tool call only."
                )
                if trace_item is not None:
                    trace_item["tool_result"] = {
                        "success": False,
                        "message": "Tool call rejected: reasoning is not allowed in no_reasoning mode.",
                    }
                    trace_item["state_done_after_step"] = False
                    state["trace"].append(trace_item)
                return

        tool_name = out["tool_name"]
        args = deepcopy(out.get("arguments", {}))
        tool_res = route_tool_call(state["envs"], tool_name, args)
        state["tool_calls_used"] = int(state.get("tool_calls_used", 0)) + 1

        if tool_format == "native":
            tool_content = json.dumps(tool_res, ensure_ascii=False)
            _append_and_track(state, "tool", tool_content)
        else:
            tool_content = "<tool_response>\n" + json.dumps(tool_res, ensure_ascii=False) + "\n</tool_response>"
            _append_and_track(state, "user", tool_content)
        # After a successful tool call, accept the next \boxed{} without reasoning —
        # the model already reasoned before the tool call.
        state["_last_tool_success"] = True
        if trace_item is not None:
            trace_item["tool_result"] = deepcopy(tool_res)
            trace_item["state_done_after_step"] = False
            state["trace"].append(trace_item)
        return

    if has_boxed_answer(raw_text):
        if skip_reasoning_check or state.get("_last_tool_success"):
            state["final_response"] = raw_text
            state["done"] = True
        else:
            has_reasoning = has_reasoning_for_direct_answer(out.get("raw_text", raw_text))
            if require_reasoning and (not has_reasoning):
                _append_and_track(state, "user",
                    "Final answer rejected: missing reasoning before final answer. "
                    "Retry by first giving a short reasoning, then provide the final answer wrapped in \\boxed{...}."
                )
            elif (not require_reasoning) and has_reasoning:
                _append_and_track(state, "user",
                    "Final answer rejected: reasoning is not allowed in no_reasoning mode. "
                    "Retry with final answer in \\boxed{...} only."
                )
            else:
                state["final_response"] = raw_text
                state["done"] = True
    else:
        # Clear the post-tool-success flag since model didn't give \boxed{}
        state.pop("_last_tool_success", None)
        sparse_hint = (
            "Sparse-tool policy: prefer final answer now if possible; if absolutely blocked, use at most one high-value tool call and avoid redundant calls. "
            if prompt_mode == "sparse_tool"
            else ""
        )
        continue_content = (
            "Continue. "
            + sparse_hint
            + (
                "You must do exactly one of these next:\n"
                "1) Provide short reasoning, then one valid tool call.\n"
                "2) Provide short reasoning, then final answer in \\boxed{...}."
                if require_reasoning
                else (
                    "You must do exactly one of these next (no reasoning text):\n"
                    "1) Provide one valid tool call.\n"
                    "2) Provide final answer in \\boxed{...}."
                )
            )
        )
        _append_and_track(state, "user", continue_content)
    if trace_item is not None:
        trace_item["tool_result"] = None
        trace_item["state_done_after_step"] = state["done"]
        state["trace"].append(trace_item)


def finalize_state(state):
    x = deepcopy(state["task"])
    x["rounds"] = state["rounds"]
    x["final_response"] = state.get("final_response", "")
    x["tool_calls"] = int(state.get("tool_calls_used", 0))
    x["generation_tokens"] = state.get("generation_tokens", 0)
    x["prefill_tokens"] = state.get("prefill_tokens", 0)
    x["token_cost"] = x["generation_tokens"] + 0.2 * x["prefill_tokens"]
    x["prompt_mode"] = state.get("prompt_mode", "current")
    x["reasoning_mode"] = "reasoning" if bool(state.get("require_reasoning", True)) else "no_reasoning"

    mode = state.get("record_mode", "lite")
    if mode == "full":
        x["output"] = state["messages"]
        x["trace"] = state.get("trace", [])
    elif mode == "lite":
        x["trace"] = state.get("trace", [])
    elif mode == "off":
        pass
    else:
        x["trace"] = state.get("trace", [])
    return x


def evaluate_batched(tasks, model, max_rounds=8, record_mode="lite", prompt_mode="current",
                     require_reasoning=True, prefills=None, tool_format="xml"):
    """Run multi-round eval loop for a batch of tasks.

    Args:
        tasks: list of task dicts
        model: AgentModel instance (must have .system_prompt and .generate_batch)
        max_rounds: max interaction rounds per task
        record_mode: "full", "lite", or "off"
        prompt_mode: one of PROMPT_MODES
        require_reasoning: whether to enforce reasoning in outputs
        prefills: optional dict mapping task_id -> prefill string.
                  Applied only in the first round to steer the model's initial response.
        tool_format: "xml" (Qwen: <tool_call> tags) or "native" (Llama: bare JSON + tool role)
    """
    states = [
        init_state(
            t,
            model.system_prompt,
            record_mode=record_mode,
            prompt_mode=prompt_mode,
            require_reasoning=require_reasoning,
            tool_format=tool_format,
            tokenizer=model.tokenizer,
        )
        for t in tasks
    ]

    for r in range(1, max_rounds + 1):
        active = [i for i, st in enumerate(states) if (not st["done"] and st["rounds"] < max_rounds)]
        if not active:
            break

        messages_batch = [states[i]["messages"] for i in active]
        tools_batch = [states[i]["tools"] for i in active]

        # Apply prefills only in the first round
        round_prefills = None
        if r == 1 and prefills:
            round_prefills = [
                prefills.get(states[i]["task"]["id"]) for i in active
            ]
            # Mark prefilled tasks so reasoning enforcement is skipped
            for i, pf in zip(active, round_prefills):
                if pf:
                    states[i]["has_prefill"] = True

        # Pre-filter tasks whose prompt exceeds model's max context length
        max_ctx = getattr(model, "max_model_len", 32768) or 32768
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            too_long = set()
            for idx_in_batch, i in enumerate(active):
                try:
                    prompt_ids = tokenizer.apply_chat_template(
                        states[i]["messages"], tools=states[i]["tools"],
                        add_generation_prompt=True, tokenize=True,
                    )
                    if len(prompt_ids) > max_ctx - 128:  # leave room for generation
                        too_long.add(idx_in_batch)
                        states[i]["done"] = True
                        states[i]["final_response"] = "[CONTEXT_LENGTH_EXCEEDED]"
                except Exception:
                    pass
            if too_long:
                active = [i for idx, i in enumerate(active) if idx not in too_long]
                messages_batch = [states[i]["messages"] for i in active]
                tools_batch = [states[i]["tools"] for i in active]
                if round_prefills is not None:
                    round_prefills = [p for idx, p in enumerate(round_prefills) if idx not in too_long]
            if not active:
                continue

        outs = model.generate_batch(messages_batch, tools_batch, prefills=round_prefills)

        for i, out in zip(active, outs):
            mode = states[i].get("record_mode", "lite")
            if mode == "full":
                trace_item = {
                    "round": states[i]["rounds"] + 1,
                    "prompt_text": out.get("prompt_text", ""),
                    "tools": deepcopy(states[i]["tools"]),
                    "model_raw_output": out.get("raw_text", ""),
                    "model_finish_reason": out.get("finish_reason", "unknown"),
                    "parsed_output": {
                        "type": out.get("type", "unknown"),
                        "tool_name": out.get("tool_name"),
                        "arguments": deepcopy(out.get("arguments", {})) if out.get("type") == "tool" else None,
                        "content": out.get("content", "") if out.get("type") == "content" else None,
                    },
                }
            elif mode == "lite":
                trace_item = {
                    "round": states[i]["rounds"] + 1,
                    "prompt_text": out.get("prompt_text", ""),
                    "model_raw_output": out.get("raw_text", ""),
                    "parsed_output": {
                        "type": out.get("type", "unknown"),
                        "tool_name": out.get("tool_name"),
                        "arguments": deepcopy(out.get("arguments", {})) if out.get("type") == "tool" else None,
                        "content": out.get("content", "") if out.get("type") == "content" else None,
                    },
                }
            else:
                trace_item = None
            step_state(states[i], out, trace_item=trace_item)

        done = sum(1 for st in states if st["done"])
        print(f"round {r}: active={len(active)} done={done}/{len(states)}")

    return [finalize_state(st) for st in states]


# ---------------------------------------------------------------------------
# Data loading: local JSON or HuggingFace dataset
# ---------------------------------------------------------------------------
HF_DATASET_ID = "cesun/When2Tool"

_PATH_TO_HF = {
    "tasks_v1_train.json": ("single_hop", "train"),
    "tasks_v1_test.json": ("single_hop", "test"),
    "tasks_v1_multihop_train.json": ("multi_hop", "train"),
    "tasks_v1_multihop_test.json": ("multi_hop", "test"),
}


def _load_from_hf(config: str, split: str) -> list:
    """Load from HuggingFace and reconstruct the original task format."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Local data not found. Install datasets to auto-download from HF: "
            "pip install datasets"
        )
    print(f"Loading from HuggingFace: {HF_DATASET_ID} ({config}/{split})")
    ds = load_dataset(HF_DATASET_ID, config, split=split)
    tasks = []
    for row in ds:
        task = {
            "id": row["id"],
            "difficulty": row["difficulty"],
            "multi_step": row["multi_step"],
            "instruction": row["instruction"],
            "environments": [{
                "name": row["env_name"],
                "tools": json.loads(row["tools"]),
                "parameters": json.loads(row["parameters"]),
            }],
            "expected": {"answer": row["answer"]},
            "tags": json.loads(row["tags"]),
        }
        steps = json.loads(row["steps"])
        if steps:
            task["expected"]["steps"] = steps
        tasks.append(task)
    return tasks


def load_tasks_from_path(data_path: str) -> list:
    """Load tasks from a local JSON file, falling back to HuggingFace if not found.

    Automatically detects which HF config/split to use based on the filename.
    """
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    basename = os.path.basename(data_path)
    if basename in _PATH_TO_HF:
        config, split = _PATH_TO_HF[basename]
        return _load_from_hf(config, split)

    raise FileNotFoundError(
        f"{data_path} not found locally. Run 'bash generate_data.sh' to generate data, "
        f"or install 'pip install datasets' to auto-download from HuggingFace."
    )
