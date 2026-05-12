"""
Probe-guided evaluation: use a trained probe to decide per-task prefill.

For each task:
  1. Load pre-computed hidden state from extract_features.py
  2. Run probe: P(tool_necessary)
  3. If P < threshold: prefill "I can solve this directly without using a tool. "
     If P >= threshold: prefill "I need to use a tool for this. "
  4. Run normal eval loop with the prefill

Usage:
    python run_probe_eval.py \
        --model_path qwen2.5-7b \
        --probe_dir ./probe_data/qwen2.5-7b \
        --data_path ./data/tasks_v1_test.json \
        --threshold 0.7 \
        --prompt_mode current \
        --output_dir ./outputs/qwen2.5-7b

    # Sweep thresholds
    for t in 0.3 0.5 0.7 0.8; do
        python run_probe_eval.py --model_path qwen2.5-7b \
            --probe_dir ./probe_data/qwen2.5-7b \
            --data_path ./data/tasks_v1_test.json \
            --threshold $t --output_dir ./outputs/qwen2.5-7b
    done
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils import (
    resolve_model_path,
    detect_tool_format,
    evaluate_batched,
    load_tasks_from_path,
    item_final_eval,
    item_tool_calls,
    PROMPT_MODES,
)


def load_probe(probe_dir, mode="no_reasoning"):
    """Load trained probe and reconstruct sklearn objects."""
    probe_data = torch.load(
        os.path.join(probe_dir, f"probe_{mode}.pt"),
        map_location="cpu", weights_only=True,
    )

    # Reconstruct scaler
    scaler = StandardScaler()
    scaler.mean_ = probe_data["scaler_mean"].numpy()
    scaler.scale_ = probe_data["scaler_scale"].numpy()
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    # Reconstruct classifier
    clf = LogisticRegression()
    clf.coef_ = probe_data["coef"].numpy().reshape(1, -1)
    clf.intercept_ = np.array([probe_data["intercept"]])
    clf.classes_ = np.array([0, 1])

    layer = probe_data.get("layer", -1)
    if isinstance(layer, str) and layer == "all":
        layer = "all"
    else:
        layer = int(layer)
    return clf, scaler, layer


PREFILL_TEMPLATES = {
    # "soft": natural language steering, model can still override
    "soft": {
        "xml": {
            "no_tool": "I can solve this directly without using a tool.\n",
            "use_tool": "I need to use a tool for this question.\n",
        },
        "native": {
            "no_tool": "I can solve this directly without using a tool.\n",
            "use_tool": "I need to use a tool for this question.\n",
        },
    },
    # "hard": force the model into a specific output format
    "hard": {
        "xml": {
            "no_tool": "\\boxed{",
            "use_tool": "<tool_call>\n",
        },
        "native": {
            "no_tool": "\\boxed{",
            "use_tool": '{"name": "',
        },
    },
}


def compute_prefills(hidden_states, task_ids, clf, scaler, layer, threshold,
                     temperature=1.0, prefill_mode="soft", tool_format="xml"):
    """Compute per-task prefills based on probe predictions."""
    if layer == "all":
        X = hidden_states.reshape(hidden_states.shape[0], -1).numpy()
    elif hidden_states.dim() == 3:
        X = hidden_states[:, layer, :].numpy()
    else:
        X = hidden_states.numpy()

    X_scaled = scaler.transform(X)

    # Compute probabilities with temperature scaling
    logits = X_scaled @ clf.coef_[0] + clf.intercept_[0]
    probs = 1.0 / (1.0 + np.exp(-logits / temperature))

    templates = PREFILL_TEMPLATES[prefill_mode][tool_format]

    prefills = {}
    stats = {"skip_tool": 0, "use_tool": 0}
    for tid, prob in zip(task_ids, probs):
        if prob < threshold:
            prefills[tid] = templates["no_tool"]
            stats["skip_tool"] += 1
        else:
            prefills[tid] = templates["use_tool"]
            stats["use_tool"] += 1

    return prefills, probs, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--probe_dir", type=str, required=True,
                        help="Directory with probe_*.pt")
    parser.add_argument("--hidden_dir", type=str, default=None,
                        help="Directory with test_hidden_*.pt and test_labels_*.json. Defaults to --probe_dir.")
    parser.add_argument("--data_path", type=str, default="./data/tasks_v1_test.json")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen2.5-7b")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Probe threshold: P(tool_necessary) < threshold -> skip tool")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for probe logits. >1 softens predictions, <1 sharpens.")
    parser.add_argument("--prefill_mode", type=str, default="soft",
                        choices=["soft", "hard"],
                        help="soft: natural language steering. "
                             "hard: force tool_call token or boxed answer.")
    parser.add_argument("--prompt_mode", type=str, default="current", choices=PROMPT_MODES)
    parser.add_argument("--probe_mode", type=str, default="no_reasoning",
                        choices=["no_reasoning", "reasoning"])
    parser.add_argument("--max_rounds", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--record_mode", type=str, default="lite")
    parser.add_argument("--n_runs", type=int, default=3,
                        help="Number of evaluation runs. Results are nested by run_id.")
    args = parser.parse_args()

    resolved_path, was_alias = resolve_model_path(args.model_path)
    if was_alias:
        print(f"Resolved: {args.model_path} -> {resolved_path}")

    # Detect tool format
    tool_format = detect_tool_format(args.model_path)
    print(f"Tool format: {tool_format}")

    # Load tasks
    tasks = load_tasks_from_path(args.data_path)
    print(f"Loaded {len(tasks)} tasks from {args.data_path}")

    # Load probe
    print(f"Loading probe from {args.probe_dir} (mode={args.probe_mode})")
    clf, scaler, probe_layer = load_probe(args.probe_dir, args.probe_mode)
    print(f"  Probe layer: {probe_layer}")

    # Load pre-computed hidden states
    hidden_dir = args.hidden_dir or args.probe_dir
    hidden_path = os.path.join(hidden_dir, f"test_hidden_{args.probe_mode}.pt")
    hidden_states = torch.load(hidden_path, map_location="cpu", weights_only=True)
    print(f"  Hidden states: {hidden_states.shape} (from {hidden_dir})")

    # Load test labels for task ID mapping
    with open(os.path.join(hidden_dir, f"test_labels_{args.probe_mode}.json")) as f:
        label_data = json.load(f)
    task_ids = [m["id"] for m in label_data["task_meta"]]

    # Compute prefills
    prefills, probs, stats = compute_prefills(
        hidden_states, task_ids, clf, scaler, probe_layer, args.threshold,
        temperature=args.temperature, prefill_mode=args.prefill_mode,
        tool_format=tool_format,
    )
    print(f"\nProbe decisions (threshold={args.threshold}, temperature={args.temperature}):")
    print(f"  Skip tool: {stats['skip_tool']}")
    print(f"  Use tool:  {stats['use_tool']}")

    # Run eval with prefills
    from model import AgentModel
    model = AgentModel(
        model_path=resolved_path,
        backend="vllm",
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        vllm_dtype="bfloat16",
        enable_thinking=False,
    )

    # The prefill IS the steering signal — no reasoning policy needed in the prompt.
    n_runs = args.n_runs
    all_run_outputs = {}
    all_run_summaries = []

    for run_idx in range(n_runs):
        if n_runs > 1:
            print(f"\n--- Run {run_idx}/{n_runs} ---")
        print(f"Running probe-guided eval (prompt_mode={args.prompt_mode}, threshold={args.threshold})...")
        outputs = evaluate_batched(
            tasks, model,
            max_rounds=args.max_rounds,
            record_mode=args.record_mode,
            prompt_mode=args.prompt_mode,
            require_reasoning=False,
            prefills=prefills,
            tool_format=tool_format,
        )
        all_run_outputs[f"run_{run_idx}"] = outputs

        # Compute metrics for this run
        total, correct, total_tc = 0, 0, 0
        by_diff = {}
        for out in outputs:
            _, _, _, is_correct = item_final_eval(out)
            tc = item_tool_calls(out)
            diff = out.get("difficulty", "unknown")
            total += 1
            correct += int(is_correct)
            total_tc += tc
            if diff not in by_diff:
                by_diff[diff] = {"total": 0, "correct": 0, "tc": 0}
            by_diff[diff]["total"] += 1
            by_diff[diff]["correct"] += int(is_correct)
            by_diff[diff]["tc"] += tc

        acc = correct / total if total else 0
        avg_tc = total_tc / total if total else 0

        run_summary = {
            "n_tasks": total,
            "accuracy": round(acc, 4),
            "avg_tool_calls": round(avg_tc, 4),
            "per_difficulty": {
                diff: {
                    "accuracy": round(d["correct"] / d["total"], 4),
                    "avg_tool_calls": round(d["tc"] / d["total"], 4),
                    "n": d["total"],
                }
                for diff, d in by_diff.items()
            },
        }
        all_run_summaries.append(run_summary)

        print(f"  Overall: acc={acc:.4f}  avg_tc={avg_tc:.2f}  n={total}")
        for diff in ["easy", "medium", "hard"]:
            if diff in by_diff:
                d = by_diff[diff]
                print(f"  {diff:8s}: acc={d['correct']/d['total']:.4f}  avg_tc={d['tc']/d['total']:.2f}  n={d['total']}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    temp_tag = f"_temp{args.temperature}"
    mode_tag = f"_{args.prefill_mode}"
    tag = f"probe_guided_t{args.threshold}{temp_tag}{mode_tag}__{args.prompt_mode}"
    out_path = os.path.join(args.output_dir, f"{tag}.json")

    if n_runs == 1:
        save_outputs = all_run_outputs["run_0"]
        save_summary = all_run_summaries[0]
    else:
        save_outputs = all_run_outputs
        save_summary = {f"run_{i}": s for i, s in enumerate(all_run_summaries)}
        # Aggregate mean/std
        import numpy as np
        accs = [s["accuracy"] for s in all_run_summaries]
        tcs = [s["avg_tool_calls"] for s in all_run_summaries]
        mean_summary = {
            "n_runs": n_runs,
            "accuracy": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "avg_tool_calls": round(float(np.mean(tcs)), 4),
            "avg_tool_calls_std": round(float(np.std(tcs)), 4),
            "per_difficulty": {},
        }
        for diff in ["easy", "medium", "hard"]:
            d_accs = [s["per_difficulty"][diff]["accuracy"] for s in all_run_summaries if diff in s["per_difficulty"]]
            d_tcs = [s["per_difficulty"][diff]["avg_tool_calls"] for s in all_run_summaries if diff in s["per_difficulty"]]
            if d_accs:
                mean_summary["per_difficulty"][diff] = {
                    "accuracy": round(float(np.mean(d_accs)), 4),
                    "accuracy_std": round(float(np.std(d_accs)), 4),
                    "avg_tool_calls": round(float(np.mean(d_tcs)), 4),
                    "avg_tool_calls_std": round(float(np.std(d_tcs)), 4),
                    "n": all_run_summaries[0]["per_difficulty"].get(diff, {}).get("n", 0),
                }
        save_summary["mean"] = mean_summary

    with open(out_path, "w") as f:
        json.dump(save_outputs, f, ensure_ascii=False, indent=2)

    # Add common fields to summary
    if n_runs == 1:
        save_summary.update({
            "threshold": args.threshold,
            "prompt_mode": args.prompt_mode,
            "probe_mode": args.probe_mode,
            "probe_layer": probe_layer,
            "prefill_stats": stats,
        })
    else:
        save_summary["config"] = {
            "threshold": args.threshold,
            "prompt_mode": args.prompt_mode,
            "probe_mode": args.probe_mode,
            "probe_layer": probe_layer,
            "prefill_stats": stats,
            "n_runs": n_runs,
        }

    summary_path = os.path.join(args.output_dir, f"{tag}_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(save_summary, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Summary: {summary_path}")
    if n_runs > 1:
        m = save_summary["mean"]
        print(f"Mean: acc={m['accuracy']}±{m['accuracy_std']}  tc={m['avg_tool_calls']}±{m['avg_tool_calls_std']}")


if __name__ == "__main__":
    main()
