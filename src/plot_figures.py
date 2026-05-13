"""
Plot accuracy vs. total tool calls tradeoff (paper-quality figures).

Usage:
    # Single model
    python plot_figures.py --models qwen3-4b-instruct --output ./figures/tradeoff.pdf

    # Multiple models side by side
    python plot_figures.py --models qwen3-1.7b qwen3-4b-instruct qwen3-14b llama3.1-8b \
        --output ./figures/all_models.pdf
"""

import argparse
import json
import os
import glob
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import item_final_eval, item_tool_calls

# ---------------------------------------------------------------------------
# Paper style
# ---------------------------------------------------------------------------
FONT_SIZE_TITLE = 28
FONT_SIZE_LABEL = 26
FONT_SIZE_TICK = 22
FONT_SIZE_LEGEND = 20
LINE_WIDTH_BASELINE = 2.5
LINE_WIDTH_PROBE = 3.0
MARKER_SIZE_BASELINE = 10
MARKER_SIZE_PROBE = 12

COLOR_NR = "#999999"
COLOR_R = "#e07070"
COLOR_PROBE_SOFT = "#1a9641"
COLOR_PROBE_OOD = "#2166ac"

plt.rcParams.update({
    "font.size": FONT_SIZE_TICK,
    "axes.labelsize": FONT_SIZE_LABEL,
    "axes.titlesize": FONT_SIZE_TITLE,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "pdf.fonttype": 42,  # TrueType fonts in PDF
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
MODES = ["force_tool", "current", "necessary_tool", "sparse_tool", "no_tool"]

# Build display names: clean short names for paper
from utils import MODEL_NAME_MAP
MODEL_DISPLAY = {}
for alias, full_path in MODEL_NAME_MAP.items():
    name = full_path.split("/")[-1]
    MODEL_DISPLAY[alias] = name

# Override display names for paper clarity
MODEL_DISPLAY["qwen3-4b-instruct"] = "Qwen3-4B-Inst-0527"
MODEL_DISPLAY["llama3.1-8b"] = "Llama-3.1-8B-Inst"
MODEL_DISPLAY["llama3.3-70b"] = "Llama-3.3-70B-Inst"


def load_tasks(data_path, env_filter=None):
    from utils import load_tasks_from_path
    all_tasks = load_tasks_from_path(data_path)
    if env_filter:
        all_tasks = [t for t in all_tasks
                     if any(e.get("name") in env_filter for e in t.get("environments", []))]
    # Use all task IDs (works for both single-hop and multi-hop datasets)
    single_ids = set(t["id"] for t in all_tasks)
    id_to_diff = {t["id"]: t.get("difficulty", "unknown") for t in all_tasks}
    return single_ids, id_to_diff


def _metrics_from_outputs(outputs, single_ids, id_to_diff=None):
    """Compute acc, avg_tc, total_tc, and token_cost from a flat list of output dicts."""
    total, correct, tc = 0, 0, 0
    total_token_cost = 0.0
    by_diff = {}
    for o in outputs:
        if o["id"] not in single_ids:
            continue
        _, _, _, is_correct = item_final_eval(o)
        t = item_tool_calls(o)
        tok = o.get("token_cost", 0.0)
        total += 1
        correct += int(is_correct)
        tc += t
        total_token_cost += tok
        if id_to_diff:
            d = id_to_diff.get(o["id"], "unknown")
            if d not in by_diff:
                by_diff[d] = {"n": 0, "correct": 0, "tc": 0, "token_cost": 0.0}
            by_diff[d]["n"] += 1
            by_diff[d]["correct"] += int(is_correct)
            by_diff[d]["tc"] += t
            by_diff[d]["token_cost"] += tok
    if total == 0:
        return None
    result = {
        "acc": correct / total,
        "avg_tc": tc / total,
        "total_tc": tc,
        "total_token_cost": total_token_cost,
        "avg_token_cost": total_token_cost / total,
        "n": total,
    }
    if by_diff:
        result["by_diff"] = {
            k: {"acc": v["correct"] / v["n"], "avg_tc": v["tc"] / v["n"],
                "total_tc": v["tc"], "total_token_cost": v["token_cost"],
                "avg_token_cost": v["token_cost"] / v["n"], "n": v["n"]}
            for k, v in by_diff.items()
        }
    return result


def compute_metrics(eval_path, single_ids, id_to_diff):
    """Load eval JSON and compute metrics. Handles both flat and multi-run formats.

    Returns dict with acc, avg_tc, n, and optionally acc_std, avg_tc_std for multi-run.
    """
    import numpy as np
    with open(eval_path) as f:
        data = json.load(f)

    # Detect format: multi-run dict {"run_0": [...], "run_1": [...]} vs flat list
    if isinstance(data, dict) and any(k.startswith("run_") for k in data):
        run_metrics = []
        for key in sorted(k for k in data if k.startswith("run_")):
            m = _metrics_from_outputs(data[key], single_ids, id_to_diff)
            if m:
                run_metrics.append(m)
        if not run_metrics:
            return None
        accs = [m["acc"] for m in run_metrics]
        tcs = [m["avg_tc"] for m in run_metrics]
        total_tcs = [m["total_tc"] for m in run_metrics]
        total_tok = [m["total_token_cost"] for m in run_metrics]
        avg_tok = [m["avg_token_cost"] for m in run_metrics]
        result = {
            "acc": float(np.mean(accs)),
            "avg_tc": float(np.mean(tcs)),
            "total_tc": float(np.mean(total_tcs)),
            "total_token_cost": float(np.mean(total_tok)),
            "avg_token_cost": float(np.mean(avg_tok)),
            "acc_std": float(np.std(accs)),
            "avg_tc_std": float(np.std(tcs)),
            "total_tc_std": float(np.std(total_tcs)),
            "total_token_cost_std": float(np.std(total_tok)),
            "avg_token_cost_std": float(np.std(avg_tok)),
            "n": run_metrics[0]["n"],
            "n_runs": len(run_metrics),
        }
        # Aggregate per-difficulty across runs
        all_diffs = set()
        for m in run_metrics:
            all_diffs.update(m.get("by_diff", {}).keys())
        if all_diffs:
            by_diff = {}
            for d in all_diffs:
                d_accs = [m["by_diff"][d]["acc"] for m in run_metrics if d in m.get("by_diff", {})]
                d_tcs = [m["by_diff"][d]["avg_tc"] for m in run_metrics if d in m.get("by_diff", {})]
                d_ttcs = [m["by_diff"][d]["total_tc"] for m in run_metrics if d in m.get("by_diff", {})]
                d_toks = [m["by_diff"][d]["total_token_cost"] for m in run_metrics if d in m.get("by_diff", {})]
                if d_accs:
                    by_diff[d] = {
                        "acc": float(np.mean(d_accs)), "avg_tc": float(np.mean(d_tcs)),
                        "total_tc": float(np.mean(d_ttcs)), "total_token_cost": float(np.mean(d_toks)),
                        "acc_std": float(np.std(d_accs)), "avg_tc_std": float(np.std(d_tcs)),
                        "total_tc_std": float(np.std(d_ttcs)), "total_token_cost_std": float(np.std(d_toks)),
                        "n": run_metrics[0]["by_diff"].get(d, {}).get("n", 0),
                    }
            result["by_diff"] = by_diff
        return result
    else:
        # Flat list (single run / old format)
        return _metrics_from_outputs(data, single_ids, id_to_diff)


def parse_probe_filename(basename):
    m = re.match(
        r"probe_guided_t([\d.]+)(?:_temp([\d.]+))?(?:_(soft|hard))?__(\w+)\.json",
        basename,
    )
    if not m:
        return None, None, None, None
    return float(m.group(1)), float(m.group(2) or 1.0), m.group(3) or "soft", m.group(4)


def collect_data(model_dir, baseline_dir, single_ids, id_to_diff):
    """Collect baselines and probe data for one model.

    Each point is a dict with label, acc, avg_tc, total_tc, total_token_cost, and std variants.
    """
    nr_points, r_points = [], []
    for mode in MODES:
        for reasoning, points_list in [("no_reasoning", nr_points), ("reasoning", r_points)]:
            path = os.path.join(baseline_dir, f"{mode}__{reasoning}.json")
            if not os.path.exists(path):
                continue
            m = compute_metrics(path, single_ids, id_to_diff)
            if m:
                m["label"] = mode
                points_list.append(m)

    # Probe lines: keyed by (temp, prefill_mode)
    probe_lines = {}
    probe_files = sorted(glob.glob(os.path.join(model_dir, "probe_guided_t*__*.json")))
    for pf in probe_files:
        if "_metrics" in pf:
            continue
        thresh, temp, pmode, prompt = parse_probe_filename(os.path.basename(pf))
        if thresh is None:
            continue
        m = compute_metrics(pf, single_ids, id_to_diff)
        if not m:
            continue
        key = (temp, pmode)
        if key not in probe_lines:
            probe_lines[key] = []
        m["label"] = thresh
        probe_lines[key].append(m)
    for key in probe_lines:
        probe_lines[key].sort(key=lambda x: x["label"])

    return nr_points, r_points, probe_lines


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_line_with_band(ax, tc, acc, tc_std, acc_std, color, marker, markersize,
                         linewidth, label, zorder, alpha=0.7):
    """Plot a line with optional shaded error band."""
    import numpy as np
    tc = np.array(tc)
    acc = np.array(acc)
    tc_std = np.array(tc_std)
    acc_std = np.array(acc_std)

    ax.plot(tc, acc, f"-{marker}", color=color, markersize=markersize,
            linewidth=linewidth, label=label, zorder=zorder, alpha=alpha)

    if np.any(acc_std > 0) or np.any(tc_std > 0):
        ax.errorbar(tc, acc,
                     xerr=tc_std if np.any(tc_std > 0) else None,
                     yerr=acc_std if np.any(acc_std > 0) else None,
                     fmt="none", ecolor=color,
                     elinewidth=1.2, capsize=3, alpha=0.5, zorder=zorder - 1)


X_AXIS_LABELS = {
    "avg_tc": "Avg Tool Calls per Task",
    "total_tc": "Total Tool Calls",
    "total_token_cost": "Total Token Cost",
}
X_AXIS_STD_KEYS = {
    "avg_tc": "avg_tc_std",
    "total_tc": "total_tc_std",
    "total_token_cost": "total_token_cost_std",
}


def plot_one_model(ax, nr_points, r_points, probe_lines, title,
                   probe_color=COLOR_PROBE_SOFT, probe_label="Probe&Prefill (soft)",
                   x_axis="avg_tc", preferred_pmode="soft", annotate_probe=True):
    """Plot tradeoff for one model on one axes with per-point labels."""
    x_std_key = X_AXIS_STD_KEYS.get(x_axis, "avg_tc_std")

    # Prompt mode display abbreviations
    _PDISPLAY = {
        "force_tool": "F", "current": "D", "necessary_tool": "N",
        "sparse_tool": "S", "no_tool": "X",
    }
    ann_fs = 11
    ann_fw = "bold"

    # No-reasoning baseline
    if nr_points:
        tc = [p.get(x_axis, 0) for p in nr_points]
        acc = [p["acc"] for p in nr_points]
        tc_std = [p.get(x_std_key, 0.0) for p in nr_points]
        acc_std = [p.get("acc_std", 0.0) for p in nr_points]
        _plot_line_with_band(ax, tc, acc, tc_std, acc_std,
                             COLOR_NR, "o", MARKER_SIZE_BASELINE,
                             LINE_WIDTH_BASELINE, "Prompt-only", zorder=2)

    # Reasoning baseline
    if r_points:
        tc = [p.get(x_axis, 0) for p in r_points]
        acc = [p["acc"] for p in r_points]
        tc_std = [p.get(x_std_key, 0.0) for p in r_points]
        acc_std = [p.get("acc_std", 0.0) for p in r_points]
        _plot_line_with_band(ax, tc, acc, tc_std, acc_std,
                             COLOR_R, "s", MARKER_SIZE_BASELINE,
                             LINE_WIDTH_BASELINE, "Reason-then-Act", zorder=2)

    # Probe line(s) — plot preferred mode, preferring temp=2.0
    preferred_temps = sorted(
        [(t, pm) for (t, pm) in probe_lines if pm == preferred_pmode],
        key=lambda x: abs(x[0] - 2.0)  # prefer temp closest to 2.0
    )
    if preferred_temps:
        key = preferred_temps[0]
        points = probe_lines[key]
        tc = [p.get(x_axis, 0) for p in points]
        acc = [p["acc"] for p in points]
        tc_std = [p.get(x_std_key, 0.0) for p in points]
        acc_std = [p.get("acc_std", 0.0) for p in points]
        _plot_line_with_band(ax, tc, acc, tc_std, acc_std,
                             probe_color, "^", MARKER_SIZE_PROBE,
                             LINE_WIDTH_PROBE, probe_label, zorder=4, alpha=0.9)

    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))


def finalize_axes(ax, all_points, show_ylabel=True, show_xlabel=True, x_axis="avg_tc"):
    """Set axis limits, labels, etc. Points are dicts."""
    if all_points:
        vals = [p.get(x_axis, 0) for p in all_points]
        max_val = max(vals) if vals else 1.0
        if x_axis == "avg_tc":
            ax.set_xlim(max_val + 0.15, -0.05)
        else:
            ax.set_xlim(max_val * 1.05, -max_val * 0.02)
    xlabel = X_AXIS_LABELS.get(x_axis, x_axis)
    if show_xlabel:
        ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("Accuracy")


# ---------------------------------------------------------------------------
# Figure 2: Multi-model grid
# ---------------------------------------------------------------------------
def figure_multimodel(models, data_path, output, env_filter=None, ood_dirs=None, x_axis="total_tc"):
    n = len(models)
    if n == 1:
        nrows, ncols = 1, 1
        figsize = (8, 6)
    elif n <= 6:
        nrows, ncols = 1, n
        figsize = (5.0 * ncols, 4.5)
    else:
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        figsize = (5.0 * ncols, 4.5 * nrows)
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, squeeze=False)
    axes = [axes_grid[i // ncols][i % ncols] for i in range(n)]
    # Hide unused subplots
    for i in range(n, nrows * ncols):
        axes_grid[i // ncols][i % ncols].set_visible(False)

    single_ids, id_to_diff = load_tasks(data_path, env_filter)

    all_points_global = []
    for idx, model in enumerate(models):
        model_name = os.path.basename(model)
        display = MODEL_DISPLAY.get(model_name, model_name)
        model_dir = f"./outputs/{model_name}"
        baseline_dir = model_dir

        # If OOD mode, probe results are in ood/ subdir
        if ood_dirs:
            probe_dir = f"./outputs/{model_name}/ood"
        else:
            probe_dir = model_dir

        nr, r, probe = collect_data(probe_dir, baseline_dir, single_ids, id_to_diff)

        probe_color = COLOR_PROBE_OOD if ood_dirs else COLOR_PROBE_SOFT
        probe_label = "Probe&Prefill OOD" if ood_dirs else "Probe&Prefill"

        if "llama" in model_name.lower():
            plot_one_model(axes[idx], nr, r, probe, display,
                           probe_color=probe_color, probe_label=probe_label,
                           x_axis=x_axis, preferred_pmode="soft",
                           annotate_probe=False)
            COLOR_HARD = "#7b3294"
            x_std_key = X_AXIS_STD_KEYS.get(x_axis, "avg_tc_std")
            hard_temps = sorted(
                [(t, pm) for (t, pm) in probe.keys() if pm == "hard"],
                key=lambda x: abs(x[0] - 2.0)
            )
            if hard_temps:
                points = probe[hard_temps[0]]
                tc = [p.get(x_axis, 0) for p in points]
                acc = [p["acc"] for p in points]
                tc_std = [p.get(x_std_key, 0.0) for p in points]
                acc_std = [p.get("acc_std", 0.0) for p in points]
                _plot_line_with_band(axes[idx], tc, acc, tc_std, acc_std,
                                     COLOR_HARD, "D", MARKER_SIZE_PROBE,
                                     LINE_WIDTH_PROBE, "Probe&Prefill (hard)",
                                     zorder=4, alpha=0.9)
        else:
            plot_one_model(axes[idx], nr, r, probe, display,
                           probe_color=probe_color, probe_label=probe_label,
                           x_axis=x_axis, preferred_pmode="soft")

        # Collect all points for axis limits
        all_pts = list(nr) + list(r)
        for pts in probe.values():
            all_pts += list(pts)
        all_points_global.extend(all_pts)

        row, col = idx // ncols, idx % ncols
        finalize_axes(axes[idx], all_pts,
                      show_ylabel=(col == 0),
                      show_xlabel=(row == nrows - 1),
                      x_axis=x_axis)

    # Place legend at bottom center, outside subplots
    handles, labels = axes[0].get_legend_handles_labels()
    has_llama = any("llama" in os.path.basename(m).lower() for m in models)
    legend_fs = FONT_SIZE_LEGEND + 2 if has_llama else FONT_SIZE_LEGEND + 4
    fig.legend(handles, labels, loc="upper center",
               fontsize=legend_fs, framealpha=0.9,
               ncol=len(labels), bbox_to_anchor=(0.5, 0.04),
               columnspacing=2.0, handletextpad=0.5)

    # Shared y-limits
    if all_points_global:
        min_acc = min(p["acc"] for p in all_points_global)
        axes[0].set_ylim(max(0, min_acc - 0.05), 1.02)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")
    # Also save PNG if output is PDF
    if output.endswith(".pdf"):
        png_path = output.replace(".pdf", ".png")
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {png_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot accuracy vs tool-call tradeoff (paper-quality figures)."
    )
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names (matching outputs/<model_name>/ directories)")
    parser.add_argument("--data_path", type=str, default="./data/tasks_v1_test.json")
    parser.add_argument("--output", type=str, default="./figures/tradeoff.pdf",
                        help="Output file path (PDF or PNG)")
    parser.add_argument("--x_axis", type=str, default="total_tc",
                        choices=["avg_tc", "total_tc"],
                        help="X-axis metric: total_tc (default) or avg_tc (per task)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figure_multimodel(args.models, args.data_path, args.output, x_axis=args.x_axis)


if __name__ == "__main__":
    main()
