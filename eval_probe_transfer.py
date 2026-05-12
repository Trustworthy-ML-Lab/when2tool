"""
Evaluate a probe trained on one domain (e.g., single-step) on another (e.g., multi-hop).

Loads a trained probe and applies it to hidden states from a different dataset.
Reports AUROC, accuracy, and per-env/difficulty breakdown.

Usage:
    python eval_probe_transfer.py \
        --source_probe ./probe_data/qwen2.5-7b/probe_no_reasoning.pt \
        --target_hidden_dir ./probe_data/qwen2.5-7b_multihop \
        --mode no_reasoning \
        --output_dir ./outputs/qwen2.5-7b_multihop
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score


def main():
    parser = argparse.ArgumentParser(description="Probe transfer evaluation")
    parser.add_argument("--source_probe", type=str, required=True,
                        help="Path to trained probe .pt file")
    parser.add_argument("--target_hidden_dir", type=str, required=True,
                        help="Directory with target hidden states and labels")
    parser.add_argument("--mode", type=str, default="no_reasoning")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load source probe
    print(f"Loading probe from {args.source_probe}")
    probe_data = torch.load(args.source_probe, map_location="cpu", weights_only=True)

    # Reconstruct sklearn objects from saved arrays
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    probe = LogisticRegression()
    probe.coef_ = probe_data["coef"].numpy().reshape(1, -1)
    probe.intercept_ = np.array([float(probe_data["intercept"])])
    probe.classes_ = np.array([0, 1])

    scaler = StandardScaler()
    scaler.mean_ = probe_data["scaler_mean"].numpy()
    scaler.scale_ = probe_data["scaler_scale"].numpy()
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    source_layer = probe_data.get("layer", -1)
    if isinstance(source_layer, str) and source_layer == "all":
        source_layer = "all"
    else:
        source_layer = int(source_layer)
    print(f"  Source probe: layer={source_layer}, "
          f"coef shape={probe.coef_.shape}")

    # Load target hidden states + labels
    hidden_path = os.path.join(args.target_hidden_dir,
                               f"test_hidden_{args.mode}.pt")
    labels_path = os.path.join(args.target_hidden_dir,
                               f"test_labels_{args.mode}.json")

    print(f"Loading target hidden states from {hidden_path}")
    hidden = torch.load(hidden_path, map_location="cpu", weights_only=True)

    with open(labels_path) as f:
        label_data = json.load(f)
    meta = label_data["task_meta"]
    labels = np.array([m["tool_necessary"] for m in meta])

    # Extract features at the same layer the source probe was trained on
    if source_layer == "all":
        # All layers concatenated
        if hidden.dim() == 3:
            X = hidden.reshape(hidden.shape[0], -1).float().numpy()
        else:
            X = hidden.float().numpy()
    elif hidden.dim() == 3:
        # [n_tasks, n_layers, hidden_dim]
        n_layers = hidden.shape[1]
        layer_idx = source_layer if source_layer >= 0 else n_layers + source_layer
        if layer_idx < n_layers:
            X = hidden[:, layer_idx, :].float().numpy()
        else:
            print(f"  WARNING: source layer {source_layer} >= target layers {n_layers}, "
                  f"using last layer")
            X = hidden[:, -1, :].float().numpy()
    elif hidden.dim() == 2:
        X = hidden.float().numpy()
    else:
        raise ValueError(f"Unexpected hidden shape: {hidden.shape}")

    # Check dimension compatibility
    expected_dim = probe.coef_.shape[1]
    actual_dim = X.shape[1]
    if actual_dim != expected_dim:
        # Source probe might be all-layers concatenated
        if hidden.dim() == 3 and actual_dim * hidden.shape[1] == expected_dim:
            print(f"  Source probe uses all-layers concat ({expected_dim}), "
                  f"reshaping target to match")
            X = hidden.reshape(hidden.shape[0], -1).numpy()
        else:
            raise ValueError(
                f"Dimension mismatch: probe expects {expected_dim}, "
                f"target has {actual_dim}. "
                f"Hidden shape: {hidden.shape}"
            )

    X = X[:len(labels)]  # align if sizes differ
    X_scaled = scaler.transform(X)

    # Predict
    y_prob = probe.predict_proba(X_scaled)[:, 1]
    y_pred = probe.predict(X_scaled)

    # Metrics
    n = len(labels)
    acc = accuracy_score(labels, y_pred)
    try:
        auroc = roc_auc_score(labels, y_prob)
    except ValueError:
        auroc = float("nan")

    print(f"\n{'='*60}")
    print(f"Transfer Results: {n} tasks")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  AUROC:    {auroc:.3f}")
    print(f"{'='*60}")

    # Per-env breakdown
    envs = sorted(set(m["env"] for m in meta))
    env_results = {}
    for env in envs:
        mask = np.array([m["env"] == env for m in meta])
        if mask.sum() == 0:
            continue
        yt, yp, ypr = labels[mask], y_pred[mask], y_prob[mask]
        e_acc = accuracy_score(yt, yp)
        try:
            e_auroc = roc_auc_score(yt, ypr)
        except ValueError:
            e_auroc = float("nan")
        env_results[env] = {"n": int(mask.sum()), "acc": e_acc, "auroc": e_auroc}
        print(f"  {env:25s} n={mask.sum():3d}  acc={e_acc:.3f}  auroc={e_auroc:.3f}")

    # Per-difficulty breakdown
    diffs = sorted(set(m["difficulty"] for m in meta))
    diff_results = {}
    for d in diffs:
        mask = np.array([m["difficulty"] == d for m in meta])
        if mask.sum() == 0:
            continue
        yt, yp, ypr = labels[mask], y_pred[mask], y_prob[mask]
        d_acc = accuracy_score(yt, yp)
        try:
            d_auroc = roc_auc_score(yt, ypr)
        except ValueError:
            d_auroc = float("nan")
        diff_results[d] = {"n": int(mask.sum()), "acc": d_acc, "auroc": d_auroc}
        print(f"  {d:25s} n={mask.sum():3d}  acc={d_acc:.3f}  auroc={d_auroc:.3f}")

    # Save
    result = {
        "source_probe": args.source_probe,
        "target_data": args.target_hidden_dir,
        "mode": args.mode,
        "source_layer": source_layer,
        "n_tasks": n,
        "accuracy": acc,
        "auroc": auroc,
        "per_env": env_results,
        "per_difficulty": diff_results,
    }
    out_path = os.path.join(args.output_dir, f"transfer_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
