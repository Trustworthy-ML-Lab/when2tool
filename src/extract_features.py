"""
Extract hidden states and no-tool labels for probing.

Phase 1 (--phase inference): uses vllm via evaluate_batched.
  - Runs hard_no_tool eval -> correctness labels (tool calls rejected, forcing direct solve)
  - If reasoning mode: also runs current + reasoning eval -> first-sentence traces
  - Processes both train and test splits if --data_path_test is provided.

Phase 2 (--phase extract): uses HF transformers for hidden state extraction.
  - no_reasoning mode: forward pass on current + no_reasoning prompt -> hidden state
  - reasoning mode: re-encode prompt + first reasoning sentence from phase 1 -> hidden state

The --require_reasoning flag controls which mode is used for BOTH label collection
and hidden state extraction, keeping them consistent.

Usage:
    # Process both train and test in one run
    python extract_features.py --model_path qwen2.5-7b --phase all \
        --data_path ./data/tasks_v1_train.json \
        --data_path_test ./data/tasks_v1_test.json \
        --output_dir ./probe_data/qwen2.5-7b

    # Reasoning mode
    python extract_features.py --model_path qwen2.5-7b --phase all --require_reasoning \
        --data_path ./data/tasks_v1_train.json \
        --data_path_test ./data/tasks_v1_test.json \
        --output_dir ./probe_data/qwen2.5-7b

    # Run phases separately
    python extract_features.py --model_path qwen2.5-7b --phase inference --tensor_parallel_size 2
    python extract_features.py --model_path qwen2.5-7b --phase extract
"""

import argparse
import gc
import json
import os
import re

import torch

from utils import (
    resolve_model_path,
    SYSTEM_PROMPT,
    build_user_message,
    build_tools_schema,
    evaluate_batched,
    item_final_eval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_reasoning_text(text):
    """Extract all reasoning from model output, before the action (tool call or boxed answer)."""
    text = (text or "").strip()
    if not text:
        return ""

    # Remove anything from tool_call / boxed markers onward
    for marker in ["<tool_call>", "\\boxed{", '{"name"']:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx].strip()

    return text


def load_tasks(data_path):
    """Load tasks from JSON file, falling back to HuggingFace if not found."""
    from utils import load_tasks_from_path
    tasks = load_tasks_from_path(data_path)
    print(f"Loaded {len(tasks)} tasks from {data_path}")
    return tasks


# ---------------------------------------------------------------------------
# Phase 1: Inference with vllm
# ---------------------------------------------------------------------------

def _run_inference_on_split(tasks, model, args, split_name):
    """Run inference for one split (train or test) and save outputs."""
    require_reasoning = args.require_reasoning
    mode_str = "reasoning" if require_reasoning else "no_reasoning"

    print(f"\n--- {split_name} split ({len(tasks)} tasks) ---")

    # Hard no-tool eval (labels): uses "current" prompt but rejects all tool calls,
    # forcing the model to solve directly. Gives accurate ground truth for tool necessity.
    print(f"  Running hard_no_tool + {mode_str} eval...")
    no_tool_outputs = evaluate_batched(
        tasks, model,
        max_rounds=args.max_rounds,
        record_mode="lite",
        prompt_mode="hard_no_tool",
        require_reasoning=require_reasoning,
    )
    out_path = os.path.join(args.output_dir, f"{split_name}_no_tool_outputs.json")
    with open(out_path, "w") as f:
        json.dump(no_tool_outputs, f, ensure_ascii=False, indent=2)

    # Current + reasoning eval (for first-sentence traces)
    if require_reasoning:
        print(f"  Running current + reasoning eval...")
        current_outputs = evaluate_batched(
            tasks, model,
            max_rounds=args.max_rounds,
            record_mode="lite",
            prompt_mode="current",
            require_reasoning=True,
        )
        out_path = os.path.join(args.output_dir, f"{split_name}_current_outputs.json")
        with open(out_path, "w") as f:
            json.dump(current_outputs, f, ensure_ascii=False, indent=2)


def run_inference(args, resolved_path, train_tasks, test_tasks):
    from model import AgentModel

    mode_str = "reasoning" if args.require_reasoning else "no_reasoning"

    print(f"\n{'='*60}")
    print(f"Phase 1: Inference (vllm) — {mode_str}")
    print(f"  Model: {resolved_path}")

    model = AgentModel(
        model_path=resolved_path,
        backend="vllm",
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        vllm_dtype="bfloat16",
        enable_thinking=False,
    )

    _run_inference_on_split(train_tasks, model, args, "train")
    if test_tasks:
        _run_inference_on_split(test_tasks, model, args, "test")

    print(f"\nPhase 1 done. Saved to {args.output_dir}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Phase 2: Hidden state extraction with HF transformers
# ---------------------------------------------------------------------------

def _extract_split(tasks, split_name, args, tokenizer, hf_model, hidden_dim):
    """Extract hidden states for one split."""
    require_reasoning = args.require_reasoning
    mode_str = "reasoning" if require_reasoning else "no_reasoning"

    print(f"\n--- {split_name} split ({len(tasks)} tasks) ---")

    # Load inference outputs
    with open(os.path.join(args.output_dir, f"{split_name}_no_tool_outputs.json")) as f:
        no_tool_outputs = json.load(f)
    no_tool_by_id = {o["id"]: o for o in no_tool_outputs}

    current_by_id = {}
    if require_reasoning:
        current_path = os.path.join(args.output_dir, f"{split_name}_current_outputs.json")
        if os.path.exists(current_path):
            with open(current_path) as f:
                current_outputs = json.load(f)
            current_by_id = {o["id"]: o for o in current_outputs}
        else:
            print(f"  WARNING: {current_path} not found, first sentences will be empty")

    def _apply_template_text(messages, tools=None):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    num_layers = hf_model.config.num_hidden_layers + 1  # +1 for embedding layer
    print(f"  Extracting all {num_layers} layers (including embedding)")

    @torch.no_grad()
    def _extract_all_layers(input_ids):
        outputs = hf_model(input_ids=input_ids, output_hidden_states=True)
        # outputs.hidden_states: tuple of (num_layers+1) tensors, each [1, seq_len, dim]
        # Index 0 = embedding output, 1..N = transformer layer outputs
        return torch.stack(
            [layer[0, -1, :].cpu().float() for layer in outputs.hidden_states]
        )  # shape: [num_layers, hidden_dim]

    n = len(tasks)
    all_hidden = torch.zeros(n, num_layers, hidden_dim)  # [n_tasks, n_layers, dim]
    labels = []
    task_meta = []

    for i, task in enumerate(tasks):
        tid = task["id"]
        difficulty = task.get("difficulty", "unknown")
        env_name = task["environments"][0]["name"] if task.get("environments") else "unknown"
        tools_schema = build_tools_schema(task)

        # Label from no-tool eval
        nt_out = no_tool_by_id.get(tid)
        if nt_out is None:
            print(f"  WARNING: task {tid} not found in no_tool outputs, skipping")
            continue
        _, _, _, nt_correct = item_final_eval(nt_out)
        tool_necessary = 0 if nt_correct else 1
        labels.append(tool_necessary)

        # Build prompt and extract hidden state
        user_content = build_user_message(
            task["instruction"], "current", require_reasoning=require_reasoning
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt_text = _apply_template_text(messages, tools=tools_schema)

        if require_reasoning:
            cur_out = current_by_id.get(tid)
            reasoning_text = ""
            if cur_out and cur_out.get("trace"):
                # Use the last round's output: earlier rounds may lack reasoning
                # (model sometimes skips reasoning and gets rejected).
                # We re-encode with just the original prompt + reasoning,
                # not the intermediate rejection rounds.
                for trace_entry in reversed(cur_out["trace"]):
                    raw_output = trace_entry.get("model_raw_output", "")
                    candidate = extract_reasoning_text(raw_output)
                    if candidate:
                        reasoning_text = candidate
                        break
            first_sentence = reasoning_text  # kept for metadata
            reencode_text = prompt_text + reasoning_text
        else:
            reencode_text = prompt_text
            first_sentence = ""

        embed_device = hf_model.get_input_embeddings().weight.device
        input_ids = tokenizer(reencode_text, return_tensors="pt")["input_ids"].to(embed_device)
        all_hidden[i] = _extract_all_layers(input_ids)

        task_meta.append({
            "id": tid,
            "difficulty": difficulty,
            "env": env_name,
            "no_tool_correct": int(nt_correct),
            "tool_necessary": tool_necessary,
            "first_sentence": first_sentence,
        })

        if (i + 1) % 10 == 0 or i == n - 1:
            nc = sum(1 for m in task_meta if m["no_tool_correct"])
            print(f"  [{i+1}/{n}] no-tool correct: {nc}/{len(task_meta)} "
                  f"({nc/len(task_meta)*100:.1f}%)")

    # Save — shape: [n_tasks, n_layers, hidden_dim]
    actual_n = len(labels)
    torch.save(
        all_hidden[:actual_n],
        os.path.join(args.output_dir, f"{split_name}_hidden_{mode_str}.pt"),
    )
    with open(os.path.join(args.output_dir, f"{split_name}_labels_{mode_str}.json"), "w") as f:
        json.dump({
            "reasoning_mode": mode_str,
            "split": split_name,
            "no_tool_correct": [m["no_tool_correct"] for m in task_meta],
            "task_meta": task_meta,
        }, f, indent=2, ensure_ascii=False)

    n_necessary = sum(labels)
    print(f"  {split_name}: {actual_n} tasks, "
          f"tool_necessary={n_necessary}/{actual_n} ({n_necessary/actual_n*100:.1f}%)")


def run_extraction(args, resolved_path, train_tasks, test_tasks):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mode_str = "reasoning" if args.require_reasoning else "no_reasoning"

    print(f"\n{'='*60}")
    print(f"Phase 2: Hidden state extraction (HF) — {mode_str}")

    print(f"Loading HF model: {resolved_path}")
    tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    hidden_dim = hf_model.config.hidden_size
    print(f"Model loaded. hidden_dim={hidden_dim}")

    _extract_split(train_tasks, "train", args, tokenizer, hf_model, hidden_dim)
    if test_tasks:
        _extract_split(test_tasks, "test", args, tokenizer, hf_model, hidden_dim)

    print(f"\nPhase 2 done. Saved to {args.output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./data/tasks_v1_train.json",
                        help="Training split")
    parser.add_argument("--data_path_test", type=str, default=None,
                        help="Test split (optional, processes both in one run)")
    parser.add_argument("--output_dir", type=str, default="./probe_data/qwen2.5-7b")
    parser.add_argument("--phase", type=str, default="all", choices=["inference", "extract", "all"])
    parser.add_argument("--require_reasoning", action="store_true",
                        help="Use reasoning mode for both labels and hidden states")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_rounds", type=int, default=12)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--max_model_len", type=int, default=32768)
    args = parser.parse_args()

    resolved_path, was_alias = resolve_model_path(args.model_path)
    if was_alias:
        print(f"Resolved: {args.model_path} -> {resolved_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_tasks = load_tasks(args.data_path)
    test_tasks = load_tasks(args.data_path_test) if args.data_path_test else None

    if args.phase in ("inference", "all"):
        run_inference(args, resolved_path, train_tasks, test_tasks)

    if args.phase in ("extract", "all"):
        run_extraction(args, resolved_path, train_tasks, test_tasks)


if __name__ == "__main__":
    main()
