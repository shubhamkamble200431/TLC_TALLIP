#!/usr/bin/env python
"""
Compare the V1 and V2 Sanskrit->Hindi corpora, structure + source-length
buckets + overlap + train/eval leakage.

Prints a table and writes:
    results/json/dataset_comparison.json   full per-split stats (both corpora)
    results/json/leakage_report.json       V1<->V2 train/test overlap counts

The pipeline does NOT de-leak the data (decision: "use the data exactly as
pasted"); this report just documents the overlap so it can be cited.

Usage:
    python scripts/compare_datasets.py
"""
import _paths  # noqa: F401

import argparse
import os

from src.data.buckets import DEFAULT_BOUNDARIES, assign_bucket, n_tokens
from src.utils.io import ensure_dir, read_lines, write_json


def _pair_set(data_dir, split, s="sa", t="hi"):
    src = read_lines(os.path.join(data_dir, f"{split}.{s}"))
    tgt = read_lines(os.path.join(data_dir, f"{split}.{t}"))
    return set(zip(src, tgt)), src


def _split_stats(data_dir, split, s="sa", t="hi", boundaries=DEFAULT_BOUNDARIES):
    p = os.path.join(data_dir, f"{split}.{s}")
    if not os.path.isfile(p):
        return None
    lines = read_lines(p)
    counts = {b: 0 for b in (1, 2, 3, 4, 5)}
    toks = 0
    for ln in lines:
        k = n_tokens(ln)
        toks += k
        counts[assign_bucket(k, boundaries)] += 1
    return {
        "n_lines": len(lines),
        "avg_src_tokens": round(toks / max(len(lines), 1), 3),
        "source_bucket_counts": counts,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1-dir", default="data/v1")
    ap.add_argument("--v2-dir", default="data/v2")
    ap.add_argument("--v1-test", default="test_5414_5")
    ap.add_argument("--v2-test", default="test_7264_5")
    ap.add_argument("--out-dir", default="results/json")
    args = ap.parse_args()

    corpora = {
        "V1": {"dir": args.v1_dir, "splits": ["train", "val", "test", args.v1_test]},
        "V2": {"dir": args.v2_dir, "splits": ["train", "val", "test", args.v2_test]},
    }

    comparison = {}
    print(f"\n{'corpus/split':<22} {'n':>8} {'B1':>7} {'B2':>7} {'B3':>7} {'B4':>7} {'B5':>7} {'avgtok':>7}")
    print("-" * 78)
    for name, meta in corpora.items():
        comparison[name] = {"dir": meta["dir"], "splits": {}}
        for sp in meta["splits"]:
            st = _split_stats(meta["dir"], sp)
            if st is None:
                continue
            comparison[name]["splits"][sp] = st
            b = st["source_bucket_counts"]
            print(f"{name + ' ' + sp:<22} {st['n_lines']:>8} {b[1]:>7} {b[2]:>7} {b[3]:>7} "
                  f"{b[4]:>7} {b[5]:>7} {st['avg_src_tokens']:>7}")

    # ---- overlap + leakage --------------------------------------
    v1_tr, _ = _pair_set(args.v1_dir, "train")
    v2_tr, _ = _pair_set(args.v2_dir, "train")
    v1_val, _ = _pair_set(args.v1_dir, "val")
    v2_val, _ = _pair_set(args.v2_dir, "val")
    v1_te, _ = _pair_set(args.v1_dir, "test")
    v2_te, _ = _pair_set(args.v2_dir, "test")
    v2_te_bal, _ = _pair_set(args.v2_dir, args.v2_test)

    leak = {
        "note": "Corpora are used exactly as pasted; NO de-leaking is applied. "
                "This report documents the overlap for the paper.",
        "unique_train_pairs": {"V1": len(v1_tr), "V2": len(v2_tr)},
        "train_train_overlap": {
            "V1_inter_V2": len(v1_tr & v2_tr),
            "only_V1": len(v1_tr - v2_tr),
            "only_V2": len(v2_tr - v1_tr),
            "union": len(v1_tr | v2_tr),
            "V1_subset_of_V2": v1_tr.issubset(v2_tr),
            "V2_subset_of_V1": v2_tr.issubset(v1_tr),
        },
        "cross_train_test_leakage": {
            "V2_train_inter_V1_test": len(v2_tr & v1_te),
            "V1_train_inter_V2_test": len(v1_tr & v2_te),
            "V1_train_inter_V2_test_7264_5": len(v1_tr & v2_te_bal),
            "V2_train_inter_V2_test_7264_5 (own split)": len(v2_tr & v2_te_bal),
            "V1uV2_train_inter_V2_test_7264_5": len((v1_tr | v2_tr) & v2_te_bal),
        },
        "val_overlap": {
            "V1_val_inter_V2_val": len(v1_val & v2_val),
            "V1uV2_train_inter_V2_val": len((v1_tr | v2_tr) & v2_val),
        },
        "main_experiment_eval_set": {
            "val": f"{args.v2_dir}/val.*  ({len(v2_val)} pairs)",
            "test": f"{args.v2_dir}/{args.v2_test}.*",
        },
    }

    ensure_dir(args.out_dir)
    write_json(os.path.join(args.out_dir, "dataset_comparison.json"), comparison)
    write_json(os.path.join(args.out_dir, "leakage_report.json"), leak)

    # bucket histograms (V1 vs V2)
    try:
        import sys
        from subprocess import run as _run

        _run([sys.executable,
              os.path.join(os.path.dirname(__file__), "plot_dataset_histograms.py"),
              "--v1-dir", args.v1_dir, "--v2-dir", args.v2_dir], check=False)
    except Exception as exc:
        print(f"[compare_datasets] histogram step skipped: {exc}")

    print("\n---- train/train overlap ----")
    for k, v in leak["train_train_overlap"].items():
        print(f"  {k:<20} {v}")
    print("\n---- train <-> eval leakage (pairs in BOTH) ----")
    for k, v in leak["cross_train_test_leakage"].items():
        print(f"  {k:<44} {v}")
    print(f"\n  wrote {args.out_dir}/dataset_comparison.json")
    print(f"  wrote {args.out_dir}/leakage_report.json")


if __name__ == "__main__":
    main()
