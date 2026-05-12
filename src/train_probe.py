"""
Train a linear probe to predict tool necessity from hidden states.

Label: 1 = tool necessary (no-tool mode got it wrong), 0 = tool unnecessary.
Features: hidden state from extract_features.py (all layers saved).

Trains on the full training set, evaluates on a separate test set.
Sweeps all layers and reports per-layer AUROC to find where the signal lives.
Saves the best probe and a summary plot.

Usage:
    python train_probe.py --data_dir ./probe_data/qwen2.5-7b --mode no_reasoning
    python train_probe.py --data_dir ./probe_data/qwen2.5-7b --mode no_reasoning --layer -1
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def load_data(data_dir, split, mode):
    """Load hidden states and labels for a given split and mode.

    Hidden states may be:
      - [n, dim]              (legacy: last layer only)
      - [n, n_layers, dim]    (all layers)
    Returns hidden tensor, labels array, and metadata.
    """
    hidden = torch.load(
        os.path.join(data_dir, f"{split}_hidden_{mode}.pt"),
        map_location="cpu", weights_only=True,
    )

    with open(os.path.join(data_dir, f"{split}_labels_{mode}.json")) as f:
        label_data = json.load(f)

    meta = label_data["task_meta"]
    labels = np.array([m["tool_necessary"] for m in meta])
    return hidden, labels, meta


def evaluate_breakdown(y_true, y_pred, y_prob, meta, key_fn, key_order=None):
    """Compute metrics grouped by a key (difficulty, env, etc.)."""
    keys = [key_fn(m) for m in meta]
    unique_keys = key_order or sorted(set(keys))

    result = {}
    for k in unique_keys:
        mask = np.array([key == k for key in keys])
        if mask.sum() == 0:
            continue
        yt, yp, ypr = y_true[mask], y_pred[mask], y_prob[mask]
        acc = accuracy_score(yt, yp)
        auroc = (
            roc_auc_score(yt, ypr)
            if len(np.unique(yt)) > 1
            else float("nan")
        )
        cm = confusion_matrix(yt, yp, labels=[0, 1]).tolist()
        result[k] = {
            "n": int(mask.sum()),
            "n_necessary": int(yt.sum()),
            "accuracy": round(acc, 4),
            "auroc": round(auroc, 4) if not np.isnan(auroc) else None,
            "confusion_matrix": cm,
        }
    return result


def train_and_eval_layer(X_train, y_train, X_test, y_test, C):
    """Train probe on one layer, return metrics."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(
        C=C, solver="lbfgs", max_iter=2000, random_state=42,
    )
    clf.fit(X_tr, y_train)

    y_test_pred = clf.predict(X_te)
    y_test_prob = clf.predict_proba(X_te)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auroc = (
        roc_auc_score(y_test, y_test_prob)
        if len(np.unique(y_test)) > 1
        else float("nan")
    )

    y_train_pred = clf.predict(X_tr)
    y_train_prob = clf.predict_proba(X_tr)[:, 1]
    train_acc = accuracy_score(y_train, y_train_pred)
    train_auroc = roc_auc_score(y_train, y_train_prob)

    return {
        "clf": clf,
        "scaler": scaler,
        "train_acc": train_acc,
        "train_auroc": train_auroc,
        "test_acc": test_acc,
        "test_auroc": test_auroc,
        "y_test_pred": y_test_pred,
        "y_test_prob": y_test_prob,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with train_/test_ hidden states and labels")
    parser.add_argument("--mode", type=str, default="no_reasoning",
                        choices=["no_reasoning", "reasoning"])
    parser.add_argument("--reg", type=float, default=10.0,
                        help="Regularization strength (larger = more regularization). "
                             "Converted to sklearn C = 1/reg.")
    parser.add_argument("--layer", type=int, default=None,
                        help="Specific layer to probe (default: sweep all). "
                             "Use -1 for last layer, 0 for embedding layer.")
    parser.add_argument("--all_layers", action="store_true",
                        help="Concatenate all layers into one feature vector.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save probe and results (default: same as data_dir)")
    parser.add_argument("--train_envs", type=str, nargs="+", default=None,
                        help="Train only on these envs (e.g. CalculatorEnv ListManipulationEnv). "
                             "Test reports all envs with OOD breakdown.")
    parser.add_argument("--train_fraction", type=float, default=1.0,
                        help="Fraction of training data to use (0.0-1.0). For data efficiency ablation.")
    args = parser.parse_args()
    args.C = 1.0 / args.reg  # sklearn uses C = 1/lambda
    if args.output_dir is None:
        args.output_dir = args.data_dir
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Regularization: reg={args.reg} -> C={args.C}")

    # --- Load data ---
    print(f"Loading data from {args.data_dir} (mode={args.mode})")
    H_train, y_train, meta_train = load_data(args.data_dir, "train", args.mode)
    H_test, y_test, meta_test = load_data(args.data_dir, "test", args.mode)

    # --- Filter training envs if specified ---
    if args.train_envs:
        train_env_set = set(args.train_envs)
        mask = np.array([m["env"] in train_env_set for m in meta_train])
        H_train = H_train[mask]
        y_train = y_train[mask]
        meta_train = [m for m, keep in zip(meta_train, mask) if keep]
        print(f"  Filtered training to {train_env_set}: {len(meta_train)} samples")
        # Report OOD vs in-distribution on test
        ood_envs = set(m["env"] for m in meta_test) - train_env_set
        if ood_envs:
            print(f"  OOD test envs: {ood_envs}")

    # --- Subsample training data if --train_fraction < 1.0 ---
    if args.train_fraction < 1.0:
        n_total = len(y_train)
        n_keep = max(1, int(n_total * args.train_fraction))
        rng = np.random.RandomState(42)
        indices = rng.permutation(n_total)[:n_keep]
        indices.sort()
        H_train = H_train[indices]
        y_train = y_train[indices]
        meta_train = [meta_train[i] for i in indices]
        print(f"  Subsampled training data: {n_keep}/{n_total} ({args.train_fraction:.0%})")

    # Handle both legacy [n, dim] and new [n, n_layers, dim] formats
    if H_train.dim() == 2:
        # Legacy format: single layer
        H_train = H_train.unsqueeze(1)  # [n, 1, dim]
        H_test = H_test.unsqueeze(1)
        print("  Loaded legacy format (single layer)")

    n_layers = H_train.shape[1]
    hidden_dim = H_train.shape[2]
    print(f"  Train: {H_train.shape[0]} samples, {n_layers} layers, dim={hidden_dim}")
    print(f"  Test:  {H_test.shape[0]} samples")
    print(f"  Tool necessary: train={y_train.sum()}/{len(y_train)}, "
          f"test={y_test.sum()}/{len(y_test)}")

    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        print("ERROR: only one class in training data, cannot train probe.")
        return

    # --- All-layers mode: concatenate + PCA ---
    if args.all_layers:
        # Flatten: [n, n_layers, dim] -> [n, n_layers * dim]
        X_train_all = H_train.reshape(H_train.shape[0], -1).numpy()
        X_test_all = H_test.reshape(H_test.shape[0], -1).numpy()
        print(f"\nAll-layers mode: {X_train_all.shape[1]} features "
              f"({n_layers} layers × {hidden_dim} dim)")

        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train_all)
        X_test_final = scaler.transform(X_test_all)

        # Train probe
        res = train_and_eval_layer(X_train_final, y_train, X_test_final, y_test, args.C)

        print(f"\n{'='*60}")
        print(f"All-layers probe ({X_train_final.shape[1]} dims, C={args.C})")
        print(f"  Train accuracy: {res['train_acc']:.4f}  AUROC: {res['train_auroc']:.4f}")
        test_auroc_str = f"{res['test_auroc']:.4f}" if not np.isnan(res['test_auroc']) else "N/A"
        print(f"  Test  accuracy: {res['test_acc']:.4f}  AUROC: {test_auroc_str}")

        # Breakdowns
        diff_bd = evaluate_breakdown(
            y_test, res["y_test_pred"], res["y_test_prob"], meta_test,
            key_fn=lambda m: m["difficulty"],
            key_order=["easy", "medium", "hard"],
        )
        print(f"\nPer-difficulty (test):")
        for diff, r in diff_bd.items():
            auroc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "N/A"
            print(f"  {diff:8s}: acc={r['accuracy']:.4f}  auroc={auroc_str}  "
                  f"n={r['n']}  necessary={r['n_necessary']}")

        env_bd = evaluate_breakdown(
            y_test, res["y_test_pred"], res["y_test_prob"], meta_test,
            key_fn=lambda m: m["env"],
        )
        print(f"\nPer-environment (test):")
        for env, r in env_bd.items():
            ood_tag = " [OOD]" if args.train_envs and env not in set(args.train_envs) else ""
            auroc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "N/A"
            print(f"  {env:25s}: acc={r['accuracy']:.4f}  auroc={auroc_str}  "
                  f"n={r['n']}  necessary={r['n_necessary']}{ood_tag}")

        # Save probe (store PCA + scaler + clf together)
        probe_path = os.path.join(args.output_dir, f"probe_{args.mode}.pt")
        torch.save({
            "coef": torch.from_numpy(res["clf"].coef_[0]),
            "intercept": float(res["clf"].intercept_[0]),
            "scaler_mean": torch.from_numpy(scaler.mean_),
            "scaler_scale": torch.from_numpy(scaler.scale_),
            "C": args.C,
            "mode": args.mode,
            "layer": "all",
            "n_layers": n_layers,
        }, probe_path)
        print(f"\nSaved all-layers probe to {probe_path}")

        # Save results
        test_auroc_val = res["test_auroc"]
        results = {
            "mode": args.mode,
            "C": args.C,
            "all_layers": True,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "best_layer": "all",
            "best_train_acc": round(res["train_acc"], 4),
            "best_test_auroc": round(test_auroc_val, 4) if not np.isnan(test_auroc_val) else None,
            "best_test_acc": round(res["test_acc"], 4),
            "train_envs": args.train_envs,
            "per_difficulty": diff_bd,
            "per_env": env_bd,
        }
        results_path = os.path.join(args.output_dir, f"probe_results_{args.mode}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {results_path}")
        return

    # --- Determine which layers to probe ---
    if args.layer is not None:
        layer_idx = args.layer if args.layer >= 0 else n_layers + args.layer
        layers_to_probe = [layer_idx]
        print(f"\nProbing single layer: {layer_idx}")
    else:
        layers_to_probe = list(range(n_layers))
        print(f"\nSweeping all {n_layers} layers (C={args.C})...")

    # --- Train probes ---
    layer_results = {}
    best_layer = -1
    best_auroc = -1.0

    for layer in layers_to_probe:
        X_train = H_train[:, layer, :].numpy()
        X_test = H_test[:, layer, :].numpy()

        res = train_and_eval_layer(X_train, y_train, X_test, y_test, args.C)

        layer_results[layer] = {
            "train_acc": round(res["train_acc"], 4),
            "train_auroc": round(res["train_auroc"], 4),
            "test_acc": round(res["test_acc"], 4),
            "test_auroc": round(res["test_auroc"], 4) if not np.isnan(res["test_auroc"]) else None,
        }

        if len(layers_to_probe) > 1:
            test_auroc_str = f"{res['test_auroc']:.4f}" if not np.isnan(res['test_auroc']) else "N/A"
            print(f"  Layer {layer:3d}: train_auroc={res['train_auroc']:.4f}  test_auroc={test_auroc_str}")

        # Select best layer based on TRAINING accuracy (no data leakage)
        if res["train_acc"] > best_auroc:
            best_auroc = res["train_acc"]
            best_layer = layer
            best_res = res

    # --- Report best layer ---
    print(f"\n{'='*60}")
    print(f"Best layer: {best_layer} (of {n_layers}, selected by train accuracy)")
    print(f"  Train accuracy: {best_res['train_acc']:.4f}  AUROC: {best_res['train_auroc']:.4f}")
    test_auroc = best_res['test_auroc']
    test_auroc_str = f"{test_auroc:.4f}" if not np.isnan(test_auroc) else "N/A"
    print(f"  Test  accuracy: {best_res['test_acc']:.4f}  AUROC: {test_auroc_str}")

    # --- Per-difficulty breakdown on best layer ---
    diff_bd = evaluate_breakdown(
        y_test, best_res["y_test_pred"], best_res["y_test_prob"], meta_test,
        key_fn=lambda m: m["difficulty"],
        key_order=["easy", "medium", "hard"],
    )
    print(f"\nPer-difficulty (test, layer {best_layer}):")
    for diff, r in diff_bd.items():
        auroc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "N/A"
        print(f"  {diff:8s}: acc={r['accuracy']:.4f}  auroc={auroc_str}  "
              f"n={r['n']}  necessary={r['n_necessary']}")

    # --- Per-env breakdown ---
    env_bd = evaluate_breakdown(
        y_test, best_res["y_test_pred"], best_res["y_test_prob"], meta_test,
        key_fn=lambda m: m["env"],
    )
    print(f"\nPer-environment (test, layer {best_layer}):")
    for env, r in env_bd.items():
        ood_tag = " [OOD]" if args.train_envs and env not in set(args.train_envs) else ""
        auroc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "N/A"
        print(f"  {env:25s}: acc={r['accuracy']:.4f}  auroc={auroc_str}  "
              f"n={r['n']}  necessary={r['n_necessary']}{ood_tag}")

    # --- Save best probe ---
    probe_path = os.path.join(args.output_dir, f"probe_{args.mode}.pt")
    torch.save({
        "coef": torch.from_numpy(best_res["clf"].coef_[0]),
        "intercept": float(best_res["clf"].intercept_[0]),
        "scaler_mean": torch.from_numpy(best_res["scaler"].mean_),
        "scaler_scale": torch.from_numpy(best_res["scaler"].scale_),
        "C": args.C,
        "mode": args.mode,
        "layer": best_layer,
        "n_layers": n_layers,
    }, probe_path)
    print(f"\nSaved best probe (layer {best_layer}) to {probe_path}")

    # --- Save full results ---
    test_auroc_val = best_res["test_auroc"]
    results = {
        "mode": args.mode,
        "C": args.C,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "best_layer": best_layer,
        "layer_selected_by": "train_accuracy",
        "best_train_acc": round(best_res["train_acc"], 4),
        "best_test_auroc": round(test_auroc_val, 4) if not np.isnan(test_auroc_val) else None,
        "best_test_acc": round(best_res["test_acc"], 4),
        "train_envs": args.train_envs,
        "per_layer": {str(k): v for k, v in layer_results.items()},
        "per_difficulty": diff_bd,
        "per_env": env_bd,
    }
    results_path = os.path.join(args.output_dir, f"probe_results_{args.mode}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {results_path}")

    # --- Save layer sweep plot ---
    if len(layers_to_probe) > 1:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            layers = sorted(layer_results.keys())
            train_accs = [layer_results[l]["train_acc"] for l in layers]
            test_accs = [layer_results[l]["test_acc"] for l in layers]

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(layers, train_accs, "b-o", markersize=3, label="Train Accuracy (layer selection)")
            ax1.plot(layers, test_accs, "r-s", markersize=3, alpha=0.6, label="Test Accuracy")
            ax1.axvline(best_layer, color="green", linestyle="--", alpha=0.7,
                        label=f"Best layer={best_layer} (by train acc)")
            ax1.set_xlabel("Layer")
            ax1.set_ylabel("Score")
            ax1.set_title(f"Probe: Tool Necessity ({args.mode}, C={args.C})")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0.4, 1.05)

            plot_path = os.path.join(args.output_dir, f"probe_layers_{args.mode}.png")
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved layer sweep plot to {plot_path}")
        except ImportError:
            print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
