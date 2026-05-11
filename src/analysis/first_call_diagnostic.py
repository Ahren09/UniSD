#!/usr/bin/env python3
"""First-call-only diagnostic: rescore tool-use results using only the first predicted call.

Usage:
    python scripts/first_call_diagnostic.py \
        outputs/eval_sanity/0.5B_base \
        outputs/eval_step4_compare/baseline_contextK1_910 \
        outputs/eval_step4_compare/multicontext_3ctx_910
"""
import json
import sys
import statistics

from src.utils.eval_utils import load_results_dir
from src.utils.tooluse_utils import score_first_call


def analyze_directory(directory):
    records, summary, fpath = load_results_dir(directory)
    label = directory.rstrip("/").split("/")[-1]
    n = len(records)

    # Call-count distribution (all examples)
    call_counts = []
    first_call_valid_json = 0
    for r in records:
        pred = r.get("predicted_calls", [])
        call_counts.append(len(pred))
        if pred:
            try:
                json.loads(pred[0]["Action_Input"])
                first_call_valid_json += 1
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

    # Original metrics from summary
    orig = {
        "action_acc": summary.get("action_accuracy", 0) if summary else 0,
        "arg_acc": summary.get("argument_accuracy", 0) if summary else 0,
        "full_acc": summary.get("full_accuracy", 0) if summary else 0,
    }

    # First-call-only rescoring
    fc_action, fc_arg, fc_full = 0.0, 0.0, 0.0
    fcs_action, fcs_arg, fcs_full = 0.0, 0.0, 0.0
    for r in records:
        pred = r.get("predicted_calls", [])
        gold = r.get("golden_calls", [])
        a, ar, f = score_first_call(pred, gold, truncate_newline=False)
        fc_action += a; fc_arg += ar; fc_full += f
        a2, ar2, f2 = score_first_call(pred, gold, truncate_newline=True)
        fcs_action += a2; fcs_arg += ar2; fcs_full += f2

    fc = {"action_acc": fc_action/n, "arg_acc": fc_arg/n, "full_acc": fc_full/n}
    fcs = {"action_acc": fcs_action/n, "arg_acc": fcs_arg/n, "full_acc": fcs_full/n}

    # Call-count stats
    zero_calls = sum(1 for c in call_counts if c == 0)
    one_call = sum(1 for c in call_counts if c == 1)
    multi_call = sum(1 for c in call_counts if c > 1)
    mean_calls = statistics.mean(call_counts) if call_counts else 0
    median_calls = statistics.median(call_counts) if call_counts else 0
    valid_json_pct = 100 * first_call_valid_json / n if n else 0

    return {
        "label": label,
        "fpath": fpath,
        "n": n,
        "orig": orig,
        "first_call": fc,
        "first_call_strict": fcs,
        "call_stats": {
            "mean": mean_calls,
            "median": median_calls,
            "zero_pct": 100 * zero_calls / n,
            "one_pct": 100 * one_call / n,
            "multi_pct": 100 * multi_call / n,
            "valid_json_pct": valid_json_pct,
        }
    }


def print_comparison(results):
    # Header
    print(f"\n{'='*100}")
    print(f"  First-Call Diagnostic Comparison")
    print(f"{'='*100}")

    # Metric table
    header = f"{'Model':>35s} | {'Variant':>18s} | {'action_acc':>10s} | {'arg_acc':>10s} | {'full_acc':>10s}"
    print(f"\n{header}")
    print(f"{'-'*len(header)}")

    for r in results:
        label = r["label"][:35]
        for variant_name, variant_key in [("original", "orig"),
                                           ("first_call_only", "first_call"),
                                           ("first_call_strict", "first_call_strict")]:
            v = r[variant_key]
            print(f"{label:>35s} | {variant_name:>18s} | "
                  f"{v['action_acc']:>10.4f} | {v['arg_acc']:>10.4f} | {v['full_acc']:>10.4f}")
        print(f"{'-'*len(header)}")

    # Call-count table
    print(f"\n{'Call-Count Distribution':^100}")
    ch = (f"{'Model':>35s} | {'Mean':>6s} | {'Median':>6s} | "
          f"{'0-Call%':>7s} | {'1-Call%':>7s} | {'>1-Call%':>8s} | {'ValidJSON%':>10s}")
    print(ch)
    print(f"{'-'*len(ch)}")

    for r in results:
        cs = r["call_stats"]
        print(f"{r['label'][:35]:>35s} | {cs['mean']:>6.2f} | {cs['median']:>6.1f} | "
              f"{cs['zero_pct']:>6.1f}% | {cs['one_pct']:>6.1f}% | {cs['multi_pct']:>7.1f}% | "
              f"{cs['valid_json_pct']:>9.1f}%")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/first_call_diagnostic.py <dir1> [dir2] [dir3] ...")
        sys.exit(1)

    results = []
    for directory in sys.argv[1:]:
        results.append(analyze_directory(directory))

    print_comparison(results)


if __name__ == "__main__":
    main()
