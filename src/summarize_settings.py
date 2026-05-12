#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re


SETTING_ORDER = [
    "easy",
    "medium",
    "hard",
]


def _label_from_settings_path(path):
    base = os.path.basename(path)
    if not base.endswith("_settings_table.json"):
        return None
    stem = base[: -len("_settings_table.json")]
    # Use suffix after the last underscore in the user-provided prefix match.
    return stem


def _common_prefix(strings):
    if not strings:
        return ""
    p = strings[0]
    for s in strings[1:]:
        while not s.startswith(p) and p:
            p = p[:-1]
    return p


def _mode_label_from_stem(stem, shared_prefix):
    if shared_prefix and stem.startswith(shared_prefix):
        lab = stem[len(shared_prefix):]
        if lab.startswith("_"):
            lab = lab[1:]
        if lab:
            return lab
    return stem


def _read_rows(path):
    """Read settings table JSON. Handles both flat list and multi-run dict formats.

    For multi-run, aggregates mean±std per (env, setting) cell.
    """
    import numpy as np
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flat list (single run / old format)
    if isinstance(data, list):
        return data

    # Multi-run format: {"run_0": [...], "run_1": [...], ...}
    if isinstance(data, dict) and any(k.startswith("run_") for k in data):
        run_keys = sorted(k for k in data if k.startswith("run_"))
        all_runs = [data[k] for k in run_keys]
        if not all_runs:
            return []

        # Group by (env, setting) across runs
        cell_values = {}  # (env, setting) -> {"accuracy": [...], "tool_call_rate": [...], ...}
        for run_rows in all_runs:
            for r in run_rows:
                key = (r.get("env", ""), r.get("setting", ""))
                if key not in cell_values:
                    cell_values[key] = {"accuracy": [], "tool_call_rate": [], "avg_tool_calls": [], "avg_token_cost": [], "total_token_cost": [], "n": []}
                for field in ["accuracy", "tool_call_rate", "avg_tool_calls", "avg_token_cost", "total_token_cost"]:
                    if r.get(field) is not None:
                        cell_values[key][field].append(r[field])
                if r.get("n") is not None:
                    cell_values[key]["n"].append(r["n"])

        # Build aggregated rows with mean and std
        rows = []
        for (env, setting), vals in sorted(cell_values.items()):
            row = {"env": env, "setting": setting}
            for field in ["accuracy", "tool_call_rate", "avg_tool_calls", "avg_token_cost", "total_token_cost"]:
                v = vals.get(field, [])
                if v:
                    row[field] = round(float(np.mean(v)), 6)
                    row[f"{field}_std"] = round(float(np.std(v)), 6)
                else:
                    row[field] = None
                    row[f"{field}_std"] = None
            row["n"] = int(vals["n"][0]) if vals["n"] else 0
            row["n_runs"] = len(run_keys)
            rows.append(row)
        return rows

    raise ValueError(f"{path}: unexpected format (not a list or multi-run dict)")


def _settings_comparison_rows(mode_to_rows, labels):
    by_mode = {}
    for label, rows in mode_to_rows.items():
        by_mode[label] = {(r["env"], r["setting"]): r for r in rows}

    envs = sorted({k[0] for idx in by_mode.values() for k in idx.keys()})
    out = []
    for env_name in envs:
        for setting in SETTING_ORDER:
            row = {"env": env_name, "setting": setting}
            n_values = []
            for label in labels:
                rec = by_mode.get(label, {}).get((env_name, setting))
                for field in ["accuracy", "tool_call_rate", "avg_tool_calls", "avg_token_cost", "total_token_cost"]:
                    row[f"{field}_{label}"] = (rec.get(field) if rec is not None else None)
                    row[f"{field}_std_{label}"] = (rec.get(f"{field}_std") if rec is not None else None)
                if rec is not None and "n" in rec:
                    n_values.append(rec["n"])
            row["n"] = (n_values[0] if n_values else 0)
            out.append(row)
    return out


def _fmt(v, std=None):
    if v is None:
        return "NA"
    if std is not None and std > 0:
        return f"{v:.4f}±{std:.4f}"
    return f"{v:.4f}"


def _settings_comparison_markdown(rows, labels, title):
    acc_cols = " | ".join([f"Acc({x})" for x in labels])
    tcr_cols = " | ".join([f"TCR({x})" for x in labels])
    atc_cols = " | ".join([f"ATC({x})" for x in labels])
    tc_cols = " | ".join([f"TokCost({x})" for x in labels])
    header = f"| Env | Setting | N | {acc_cols} | {tcr_cols} | {atc_cols} | {tc_cols} |"
    sep = "|---|---|---:|" + "|".join(["---:"] * (4 * len(labels))) + "|"
    lines = [f"## {title}", "", header, sep]
    for r in rows:
        acc_vals = " | ".join([_fmt(r.get(f"accuracy_{x}"), r.get(f"accuracy_std_{x}")) for x in labels])
        tcr_vals = " | ".join([_fmt(r.get(f"tool_call_rate_{x}"), r.get(f"tool_call_rate_std_{x}")) for x in labels])
        atc_vals = " | ".join([_fmt(r.get(f"avg_tool_calls_{x}"), r.get(f"avg_tool_calls_std_{x}")) for x in labels])
        tc_vals = " | ".join([_fmt(r.get(f"avg_token_cost_{x}"), r.get(f"avg_token_cost_std_{x}")) for x in labels])
        lines.append(f"| {r['env']} | {r['setting']} | {r['n']} | {acc_vals} | {tcr_vals} | {atc_vals} | {tc_vals} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default="",
        help="Glob for per-setting files, e.g. ./outputs/qwen/full_eval_all_settings_*_settings_table.json",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Explicit list of *_settings_table.json files. If set, overrides --glob.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="",
        help="Prefix for outputs. Defaults to shared prefix inferred from inputs, else ./settings_summary",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Settings Accuracy Comparison",
        help="Markdown table title.",
    )
    args = parser.parse_args()

    input_paths = list(args.inputs)
    if not input_paths:
        if not args.glob_pattern:
            raise ValueError("Provide --inputs or --glob.")
        input_paths = sorted(glob.glob(args.glob_pattern))

    input_paths = [p for p in input_paths if p.endswith("_settings_table.json")]
    if not input_paths:
        raise ValueError("No *_settings_table.json files found.")

    stems = [os.path.basename(p)[: -len("_settings_table.json")] for p in input_paths]
    shared_prefix = _common_prefix(stems)
    if shared_prefix.endswith("_"):
        shared_prefix = shared_prefix[:-1]

    mode_to_rows = {}
    labels = []
    for path, stem in zip(input_paths, stems):
        label = _mode_label_from_stem(stem, shared_prefix)
        # Keep labels stable and readable.
        label = re.sub(r"\s+", "_", label.strip()) or stem
        rows = _read_rows(path)
        mode_to_rows[label] = rows
        labels.append(label)

    # Preserve input order while deduplicating.
    labels = list(dict.fromkeys(labels))
    cmp_rows = _settings_comparison_rows(mode_to_rows, labels=labels)

    if args.output_prefix:
        out_prefix = args.output_prefix
    elif shared_prefix:
        # Keep outputs near the current run files.
        first_dir = os.path.dirname(input_paths[0]) or "."
        out_prefix = os.path.join(first_dir, shared_prefix)
    else:
        out_prefix = "./settings_summary"

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    json_path = out_prefix + "_settings_comparison.json"
    md_path = out_prefix + "_settings_comparison.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cmp_rows, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_settings_comparison_markdown(cmp_rows, labels=labels, title=args.title))

    print(f"Read {len(input_paths)} setting tables")
    print(f"Labels: {', '.join(labels)}")
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
