#!/usr/bin/env python
"""
news_test out-of-domain eval for the mamba_small/base/large baselines.
Separate from evaluate_news_test.py because these are decoder-only
prefix-LM checkpoints (see scripts/train_mamba_baseline.py) -- a different
generation path (no encoder/decoder split, no forced_bos) from every other
model in this project, so they can't go through _load_ckpt/translate().

Output, same layout convention as evaluate_news_test.py:
    results/news_test/<mamba_size>/sa-hi/X-0000/seed_<seed>/
        source / references / predictions / metrics.json
    results/news_test/mamba_summary.csv
"""
import _paths  # noqa: F401

import argparse
import csv
import os
import statistics as stats
from collections import defaultdict

import torch
from transformers import AutoTokenizer, MambaForCausalLM

from scripts.train_mamba_baseline import SIZES, generate_predictions
from src.data.buckets import BUCKET_IDS, BUCKET_LABELS_LONG
from src.evaluation.metrics import compute_all
from src.utils.io import ensure_dir, read_lines, write_json, write_lines

SEEDS = [42, 43, 44]
ALL_METRICS = ["bleu", "chrf2", "ter", "meteor", "bertscore", "comet"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news-dir", default="data/news_test")
    ap.add_argument("--src-file", default="test.sa")
    ap.add_argument("--tgt-file", default="test.hi")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--dataset-version", default="v2")
    ap.add_argument("--out-root", default="results/news_test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--sizes", nargs="*", choices=list(SIZES), default=list(SIZES))
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def build_plan(args):
    plan = []
    for size in args.sizes:
        for seed in args.seeds:
            ckpt = os.path.join(args.experiments_root, args.dataset_version, "baselines",
                                f"mamba_{size}", "X-0000", f"seed_{seed}", "checkpoint-best")
            out_dir = os.path.join(args.out_root, f"mamba_{size}", "sa-hi", "X-0000", f"seed_{seed}")
            plan.append({"name": f"mamba_{size}", "seed": seed, "checkpoint": ckpt, "out_dir": out_dir})
    return plan


def run_cell(args, cell, srcs, refs):
    ckpt = cell["checkpoint"]
    print(f"[{cell['name']}/seed_{cell['seed']}] loading {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = MambaForCausalLM.from_pretrained(ckpt).to(args.device).eval()
    sep_id = tokenizer.eos_token_id

    preds = generate_predictions(model, tokenizer, srcs, sep_id, args.device, batch_size=args.batch_size)
    scores = compute_all(preds, refs, srcs, metrics=ALL_METRICS, run_comet=True,
                         max_gen_len=128, bertscore_lang="hi")
    del model
    torch.cuda.empty_cache()

    ensure_dir(cell["out_dir"])
    write_lines(os.path.join(cell["out_dir"], "source"), srcs)
    write_lines(os.path.join(cell["out_dir"], "references"), refs)
    write_lines(os.path.join(cell["out_dir"], "predictions"), preds)
    write_json(os.path.join(cell["out_dir"], "metrics.json"), {
        "model": cell["name"], "lang_pair": "sa-hi", "config": "X-0000", "seed": cell["seed"],
        "checkpoint": ckpt, "test_corpus": "news_test", "n_examples": len(srcs), "metrics": scores,
    })
    bleu, chrf2, comet = scores.get("bleu"), scores.get("chrf2"), scores.get("comet")
    print(f"    -> BLEU={bleu:.2f} chrF2={chrf2:.2f}"
          f"{f' COMET={comet:.2f}' if comet is not None else ''}")


def summarize(args, plan):
    import json

    overall_keys = ALL_METRICS
    bucket_keys = [f"{BUCKET_LABELS_LONG[b]}_{m}" for b in BUCKET_IDS for m in ALL_METRICS]
    all_keys = overall_keys + bucket_keys
    rows = defaultdict(lambda: defaultdict(list))
    n_seeds = defaultdict(int)
    for cell in plan:
        mpath = os.path.join(cell["out_dir"], "metrics.json")
        if not os.path.isfile(mpath):
            continue
        m = json.load(open(mpath))["metrics"]
        n_seeds[cell["name"]] += 1
        for mk in all_keys:
            if mk in m:
                rows[cell["name"]][mk].append(m[mk])

    def mean_var(vals):
        if not vals:
            return "", ""
        mean = round(stats.mean(vals), 3)
        var = round(stats.variance(vals), 4) if len(vals) > 1 else 0.0
        return mean, var

    out_csv = os.path.join(args.out_root, "mamba_summary.csv")
    header = ["model", "n_seeds"] + [f"{k}_{s}" for k in all_keys for s in ("mean", "var")]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for name in sorted(rows):
            row = [name, n_seeds[name]]
            for mk in all_keys:
                row += list(mean_var(rows[name][mk]))
            w.writerow(row)
    print(f"\nwrote {out_csv}")
    print("\n-- OVERALL --")
    hdr = f"  {'model':12s} {'n':>2s} | " + " | ".join(f"{m:>12s}" for m in overall_keys)
    print(hdr)
    for name in sorted(rows):
        cells = []
        for mk in overall_keys:
            mean, var = mean_var(rows[name][mk])
            cells.append(f"{'--':>12s}" if mean == "" else f"{mean:6.2f}(v{var:4.2f})")
        print(f"  {name:12s} {n_seeds[name]:>2d} | " + " | ".join(cells))


def main():
    args = parse_args()
    srcs = read_lines(os.path.join(args.news_dir, args.src_file))
    refs = read_lines(os.path.join(args.news_dir, args.tgt_file))
    plan = build_plan(args)
    n_have = sum(1 for c in plan if os.path.isdir(c["checkpoint"]))
    print(f"mamba news_test eval plan: {len(plan)} cells ({n_have} with checkpoint-best/)")

    for cell in plan:
        mpath = os.path.join(cell["out_dir"], "metrics.json")
        if os.path.isfile(mpath) and not args.overwrite:
            print(f"  [skip] {cell['name']}/seed_{cell['seed']} already scored")
            continue
        if not os.path.isdir(cell["checkpoint"]):
            print(f"  [skip] {cell['name']}/seed_{cell['seed']} -- no checkpoint-best/ ({cell['checkpoint']})")
            continue
        if args.dry_run:
            print(f"  [would run] {cell['name']}/seed_{cell['seed']}")
            continue
        run_cell(args, cell, srcs, refs)

    if not args.dry_run:
        summarize(args, plan)


if __name__ == "__main__":
    main()
