import argparse
import json
import os

from model import AgentModel
from utils import (
    resolve_model_path,
    detect_tool_format,
    get_system_prompt,
    PROMPT_MODES,
    REASONING_MODES,
    build_user_message,
    evaluate_batched,
    load_tasks_from_path,
    clean as _clean,
    item_final_eval as _item_final_eval,
    item_tool_calls as _item_tool_calls,
    item_expected_steps as _item_expected_steps,
)
SETTING_ORDER = [
    "easy",
    "medium",
    "hard",
]


def parse_csv_set(s):
    if not s:
        return None
    return {x.strip() for x in s.split(",") if x.strip()}


def task_env_signature(task):
    names = sorted({e.get("name", "") for e in task.get("environments", []) if e.get("name", "")})
    return "+".join(names) if names else "no_env"


def keep_by_filters(task, env_filter=None, difficulty_filter=None):
    if env_filter is not None:
        task_envs = {e.get("name", "") for e in task.get("environments", []) if e.get("name", "")}
        if len(task_envs.intersection(env_filter)) == 0:
            return False

    if difficulty_filter is not None:
        if task.get("difficulty") not in difficulty_filter:
            return False

    return True


def dump_grouped_outputs(outputs, output_path):
    out_dir = os.path.dirname(output_path)
    base = os.path.splitext(os.path.basename(output_path))[0]
    by_env_dir = os.path.join(out_dir, f"{base}_by_env")
    by_difficulty_dir = os.path.join(out_dir, f"{base}_by_difficulty")
    os.makedirs(by_env_dir, exist_ok=True)
    os.makedirs(by_difficulty_dir, exist_ok=True)

    env_groups = {}
    difficulty_groups = {}
    for item in outputs:
        ekey = task_env_signature(item)
        dkey = item.get("difficulty", "unknown")
        env_groups.setdefault(ekey, []).append(item)
        difficulty_groups.setdefault(dkey, []).append(item)

    for key, items in env_groups.items():
        with open(os.path.join(by_env_dir, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    for key, items in difficulty_groups.items():
        with open(os.path.join(by_difficulty_dir, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    return by_env_dir, by_difficulty_dir, env_groups, difficulty_groups


def build_metrics(outputs):
    per_task = []
    by_difficulty = {}
    total_tool_calls_all = 0
    total_expected_steps_all = 0

    for item in outputs:
        task_id = item.get("id")
        difficulty = item.get("difficulty", "unknown")
        expected = item.get("expected", {}) or {}

        gold_final = _clean(expected.get("answer", ""))
        model_final_raw, model_final_box, model_final, final_correct = _item_final_eval(item)
        tool_calls = _item_tool_calls(item)
        expected_steps = _item_expected_steps(item)
        tool_call_rate = (tool_calls / expected_steps) if expected_steps > 0 else None

        rec = {
            "id": task_id,
            "difficulty": difficulty,
            "final_correct": final_correct,
            "model_answer_raw": model_final_raw,
            "model_boxed_content": model_final_box,
            "model_answer": model_final,
            "ground_truth": gold_final,
            "tool_calls": tool_calls,
            "expected_steps": expected_steps,
            "tool_call_rate": tool_call_rate,
        }
        per_task.append(rec)

        agg = by_difficulty.setdefault(
            difficulty, {"n": 0, "final_correct": 0, "tool_calls": 0, "expected_steps": 0,
                         "generation_tokens": 0, "prefill_tokens": 0, "token_cost": 0.0},
        )
        agg["n"] += 1
        agg["final_correct"] += int(final_correct)
        agg["tool_calls"] += tool_calls
        agg["expected_steps"] += expected_steps
        agg["generation_tokens"] += item.get("generation_tokens", 0)
        agg["prefill_tokens"] += item.get("prefill_tokens", 0)
        agg["token_cost"] += item.get("token_cost", 0.0)

        total_tool_calls_all += tool_calls
        total_expected_steps_all += expected_steps

    total_gen_tokens_all = sum(agg.get("generation_tokens", 0) for agg in by_difficulty.values())
    total_prefill_tokens_all = sum(agg.get("prefill_tokens", 0) for agg in by_difficulty.values())
    total_token_cost_all = sum(agg.get("token_cost", 0.0) for agg in by_difficulty.values())

    for d, agg in by_difficulty.items():
        agg["final_accuracy"] = (agg["final_correct"] / agg["n"]) if agg["n"] > 0 else 0.0
        agg["avg_tool_calls"] = (agg["tool_calls"] / agg["n"]) if agg["n"] > 0 else 0.0
        agg["tool_call_rate"] = (agg["tool_calls"] / agg["expected_steps"]) if agg["expected_steps"] > 0 else 0.0
        agg["avg_token_cost"] = (agg["token_cost"] / agg["n"]) if agg["n"] > 0 else 0.0

    overall = {
        "n": len(per_task),
        "final_accuracy": (sum(int(x["final_correct"]) for x in per_task) / len(per_task)) if per_task else 0.0,
        "total_tool_calls": total_tool_calls_all,
        "avg_tool_calls": (total_tool_calls_all / len(per_task)) if per_task else 0.0,
        "tool_call_rate": (total_tool_calls_all / total_expected_steps_all) if total_expected_steps_all > 0 else 0.0,
        "total_token_cost": total_token_cost_all,
        "avg_token_cost": (total_token_cost_all / len(per_task)) if per_task else 0.0,
        "total_generation_tokens": total_gen_tokens_all,
        "total_prefill_tokens": total_prefill_tokens_all,
    }

    return {"overall": overall, "by_difficulty": by_difficulty, "per_task": per_task}


def build_setting_table(outputs):
    envs = sorted({task_env_signature(x) for x in outputs})
    rows = []
    for env_name in envs:
        for diff in SETTING_ORDER:
            subset = [
                x
                for x in outputs
                if task_env_signature(x) == env_name
                and x.get("difficulty") == diff
            ]
            n = len(subset)
            correct = 0
            total_tool_calls = 0
            total_expected_steps = 0
            total_gen_tokens = 0
            total_prefill_tokens = 0
            total_token_cost = 0.0
            for item in subset:
                _raw, _box, _final, ok = _item_final_eval(item)
                correct += int(ok)
                tc = _item_tool_calls(item)
                es = _item_expected_steps(item)
                total_tool_calls += tc
                total_expected_steps += es
                total_gen_tokens += item.get("generation_tokens", 0)
                total_prefill_tokens += item.get("prefill_tokens", 0)
                total_token_cost += item.get("token_cost", 0.0)
            acc = (correct / n) if n > 0 else None
            avg_tool_calls = (total_tool_calls / n) if n > 0 else None
            tool_call_rate = (total_tool_calls / total_expected_steps) if total_expected_steps > 0 else None
            avg_token_cost = (total_token_cost / n) if n > 0 else None
            rows.append(
                {
                    "env": env_name,
                    "setting": diff,
                    "n": n,
                    "correct": correct,
                    "accuracy": acc,
                    "total_tool_calls": total_tool_calls,
                    "avg_tool_calls": avg_tool_calls,
                    "tool_call_rate": tool_call_rate,
                    "total_token_cost": total_token_cost,
                    "avg_token_cost": avg_token_cost,
                    "total_generation_tokens": total_gen_tokens,
                    "total_prefill_tokens": total_prefill_tokens,
                }
            )
    return rows


def setting_table_markdown(rows, title):
    lines = [
        f"## {title}",
        "",
        "| Env | Setting | Correct | Total | Accuracy | ToolCalls | AvgToolCalls | ToolCallRate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        acc = "NA" if r["accuracy"] is None else f"{r['accuracy']:.4f}"
        avg_tc = "NA" if r["avg_tool_calls"] is None else f"{r['avg_tool_calls']:.4f}"
        tcr = "NA" if r["tool_call_rate"] is None else f"{r['tool_call_rate']:.4f}"
        lines.append(
            f"| {r['env']} | {r['setting']} | {r['correct']} | {r['n']} | {acc} | {r['total_tool_calls']} | {avg_tc} | {tcr} |"
        )
    return "\n".join(lines) + "\n"


def _mode_output_path(base_path, mode, append_suffix):
    # Directory mode: write mode-labeled files directly under the directory.
    # Example: base_path="./outputs/qwen2.5-7b/" -> "./outputs/qwen2.5-7b/current__reasoning.json"
    base_name = os.path.basename(base_path.rstrip("/\\"))
    if base_path.endswith(("/", "\\")) or base_name in {"", ".", ".."}:
        return os.path.join(base_path, f"{mode}.json")
    if not append_suffix:
        return base_path
    root, ext = os.path.splitext(base_path)
    ext = ext or ".json"
    return f"{root}_{mode}{ext}"


def _run_label(prompt_mode, reasoning_mode, include_reasoning_in_label):
    if include_reasoning_in_label:
        return f"{prompt_mode}__{reasoning_mode}"
    return prompt_mode


def _aggregate_multi_run_metrics(all_run_metrics):
    """Compute mean and std across multiple run metrics."""
    import numpy as np

    def _collect_values(runs, *keys):
        vals = []
        for r in runs:
            v = r
            for k in keys:
                if isinstance(v, dict) and k in v:
                    v = v[k]
                else:
                    v = None
                    break
            if v is not None and isinstance(v, (int, float)):
                vals.append(v)
        return vals

    # Collect scalar metrics to aggregate
    result = {"n_runs": len(all_run_metrics)}
    scalar_keys = ["final_accuracy", "avg_tool_calls", "tool_call_rate", "avg_token_cost", "total_token_cost"]

    # Overall
    for key in scalar_keys:
        vals = _collect_values(all_run_metrics, "overall", key)
        if vals:
            result.setdefault("overall", {})[key] = round(float(np.mean(vals)), 6)
            result.setdefault("overall_std", {})[key] = round(float(np.std(vals)), 6)

    # Per difficulty
    for diff in ["easy", "medium", "hard"]:
        for key in scalar_keys:
            vals = _collect_values(all_run_metrics, "by_difficulty", diff, key)
            if vals:
                result.setdefault("by_difficulty", {}).setdefault(diff, {})[key] = round(float(np.mean(vals)), 6)
                result.setdefault("by_difficulty_std", {}).setdefault(diff, {})[key] = round(float(np.std(vals)), 6)

    return result


def _run_one_mode(args, data, model, mode, output_path, require_reasoning=True, run_label=None, tool_format="xml"):
    label = run_label or mode
    n_runs = getattr(args, "n_runs", 1)

    all_run_outputs = {}
    all_run_metrics = []
    all_run_settings = []

    for run_idx in range(n_runs):
        if n_runs > 1:
            print(f"\n[{label}] Run {run_idx}/{n_runs}...")

        outputs = evaluate_batched(
            data,
            model,
            max_rounds=args.max_rounds,
            record_mode=args.record_mode,
            prompt_mode=mode,
            require_reasoning=require_reasoning,
            tool_format=tool_format,
        )

        all_run_outputs[f"run_{run_idx}"] = outputs
        all_run_metrics.append(build_metrics(outputs))
        all_run_settings.append(build_setting_table(outputs))

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Save outputs (nested by run)
    if n_runs == 1:
        # Backward compatible: single run saves flat list
        save_outputs = all_run_outputs["run_0"]
        save_metrics = all_run_metrics[0]
        save_settings = all_run_settings[0]
    else:
        save_outputs = all_run_outputs
        save_metrics = {
            f"run_{i}": m for i, m in enumerate(all_run_metrics)
        }
        save_metrics["mean"] = _aggregate_multi_run_metrics(all_run_metrics)
        save_settings = {
            f"run_{i}": s for i, s in enumerate(all_run_settings)
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_outputs, f, ensure_ascii=False, indent=2)

    # For grouped outputs and display, use the first run
    first_outputs = all_run_outputs["run_0"]
    by_env_dir, by_difficulty_dir, env_groups, difficulty_groups = dump_grouped_outputs(first_outputs, output_path)

    metrics_path = os.path.splitext(output_path)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(save_metrics, f, ensure_ascii=False, indent=2)

    settings_json_path = os.path.splitext(output_path)[0] + "_settings_table.json"
    with open(settings_json_path, "w", encoding="utf-8") as f:
        json.dump(save_settings, f, ensure_ascii=False, indent=2)
    settings_md_path = os.path.splitext(output_path)[0] + "_settings_table.md"
    with open(settings_md_path, "w", encoding="utf-8") as f:
        first_settings = all_run_settings[0]
        f.write(setting_table_markdown(first_settings, f"Settings Accuracy ({label})"))

    print(f"[{label}] Saved {len(first_outputs)} episodes x {n_runs} runs to {output_path}")
    print(f"[{label}] Saved metrics: {metrics_path}")
    if n_runs > 1:
        agg = save_metrics["mean"]
        overall = agg.get("overall", {})
        overall_std = save_metrics["mean"].get("overall_std", {})
        print(f"[{label}] Mean accuracy: {overall.get('final_accuracy', '?')} ± {overall_std.get('final_accuracy', '?')}")
    print(f"[{label}] env counts:")
    for k, v in sorted(env_groups.items()):
        print(f"  {k}: {len(v)}")
    print(f"[{label}] difficulty counts:")
    for k, v in sorted(difficulty_groups.items()):
        print(f"  {k}: {len(v)}")

    return {
        "mode": mode,
        "reasoning_mode": "reasoning" if require_reasoning else "no_reasoning",
        "label": label,
        "output_path": output_path,
        "metrics_path": metrics_path,
        "settings_json_path": settings_json_path,
        "settings_md_path": settings_md_path,
        "settings_rows": all_run_settings[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HF model repo path or alias (e.g. qwen2.5-7b, llama3.1-8b).",
    )
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "hf"])
    parser.add_argument("--data_path", type=str, default="./data/tasks_v1_test.json",
                        help="Path to task JSON (auto-downloads from HuggingFace if not found locally)")
    parser.add_argument("--output_path", type=str, default="./outputs/gen_res.json")
    parser.add_argument("--max_rounds", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=3,
                        help="Number of evaluation runs per setting. Results are nested by run_id.")

    parser.add_argument("--envs", type=str, default="", help="Comma-separated env filter")
    parser.add_argument("--difficulties", type=str, default="", help="Comma-separated difficulty filter (easy,medium,hard)")

    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--enable_thinking",
        type=str,
        default="false",
        choices=["auto", "true", "false"],
        help="Control chat-template thinking mode when supported: auto (do nothing), true, false.",
    )
    parser.add_argument(
        "--record_mode",
        type=str,
        default="off",
        choices=["full", "lite", "off"],
        help="full=full trace+messages, lite=prompt/raw output + parsed action only, off=minimal episode records",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="current",
        choices=["current", "necessary_tool", "sparse_tool", "force_tool", "no_tool", "all"],
        help="Prompting/eval mode. 'all' runs all prompt modes and writes per-setting outputs.",
    )
    parser.add_argument(
        "--reasoning_mode",
        type=str,
        default="reasoning",
        choices=["reasoning", "no_reasoning", "all"],
        help="Whether to require short reasoning in outputs. 'all' runs both and writes per-setting outputs.",
    )

    args = parser.parse_args()

    enable_thinking = None
    if args.enable_thinking == "true":
        enable_thinking = True
    elif args.enable_thinking == "false":
        enable_thinking = False

    data = load_tasks_from_path(args.data_path)

    env_filter = parse_csv_set(args.envs)
    difficulty_filter = parse_csv_set(args.difficulties)
    data = [d for d in data if keep_by_filters(d, env_filter=env_filter, difficulty_filter=difficulty_filter)]

    if args.max_samples > 0:
        data = data[: args.max_samples]

    print(f"Loaded {len(data)} tasks (backend={args.backend})")

    resolved_model_path, was_alias = resolve_model_path(args.model_path)
    if was_alias:
        print(f"Resolved model alias: {args.model_path} -> {resolved_model_path}")

    tool_format = detect_tool_format(args.model_path)
    system_prompt = get_system_prompt(tool_format)
    print(f"Tool format: {tool_format}")

    model = AgentModel(
        model_path=resolved_model_path,
        backend=args.backend,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        vllm_dtype=args.vllm_dtype,
        enable_thinking=enable_thinking,
        system_prompt_override=system_prompt,
    )

    prompt_modes = list(PROMPT_MODES) if args.prompt_mode == "all" else [args.prompt_mode]
    reasoning_modes = list(REASONING_MODES) if args.reasoning_mode == "all" else [args.reasoning_mode]
    run_pairs = [(pm, rm) for rm in reasoning_modes for pm in prompt_modes]

    include_reasoning_in_label = len(reasoning_modes) > 1
    append_suffix = len(run_pairs) > 1
    for pm, rm in run_pairs:
        label = _run_label(pm, rm, include_reasoning_in_label)
        out_path_mode = _mode_output_path(args.output_path, label, append_suffix=append_suffix)
        _run_one_mode(
            args,
            data,
            model,
            pm,
            out_path_mode,
            require_reasoning=(rm == "reasoning"),
            run_label=label,
            tool_format=tool_format,
        )


if __name__ == "__main__":
    main()
