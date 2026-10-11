#!/usr/bin/env python
"""
news_test out-of-domain eval for the TLC ablations, X-0100/seed_42/v2.
Generalized over --run-group-prefix so it covers both ablation batches:

  --run-group-prefix tlc_ablation     (default; jobs/slurm/run_tlc_ablation.sh)
      12 variants: baseline, huber, squared, l1_minus1,
      lambda_{0.01,0.05,0.2,0.5,1.0}, dim_{512,2048,4096}
  --run-group-prefix tlc_ablation_q3  (jobs/slurm/run_tlc_ablation_q3.sh)
      2 variants: predicted_target_length, oracle_ratio_class
      (pass --variants predicted_target_length oracle_ratio_class)

Reuses the standard checkpoint-loading path (_load_ckpt/_forced_bos/
translate/compute_all) -- _load_ckpt already transparently re-applies the
dedicated bucket embedding for the dim_* variants via the
bucket_embed_dim.json sidecar (see src/evaluation/evaluate.py), so no
special-casing is needed here.

Output, one folder per (model, variant), under a prefix-specific root so the
two batches never collide:
    results/news_test/<run-group-prefix>/<model>/<variant>/X-0100/seed_42/
        source / references / predictions / metrics.json
    results/news_test/<run-group-prefix>/summary.csv
"""
import _paths  # noqa: F401

import argparse
import csv
import os
import statistics as stats
from collections import defaultdict

import torch

from src.config import RunConfig
from src.data.buckets import BUCKET_IDS, BUCKET_LABELS_LONG
from src.evaluation.evaluate import _forced_bos, _load_ckpt
from src.evaluation.generate import translate
from src.evaluation.metrics import compute_all
from src.models.lang_codes import infer_model_family
from src.utils.io import ensure_dir, read_lines, write_json, write_lines

MODELS = [
    {"name": "nllb", "model_key": "nllb", "experiment": "factorial"},
    {"name": "mbart50", "model_key": "mbart50_many_to_many", "experiment": "mbart50"},
]
VARIANTS = (
    ["baseline", "huber", "squared", "l1_minus1"]
    + [f"lambda_{v}" for v in ("0.01", "0.05", "0.2", "0.5", "1.0")]
    + [f"dim_{v}" for v in (512, 2048, 4096)]
)
CONFIG = "X-0100"
SEED = 42
LANG_PAIR = "sa-hi"
ALL_METRICS = ["bleu", "chrf2", "ter", "meteor", "bertscore", "comet"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--news-dir", default="data/news_test")
    ap.add_argument("--src-file", default="test.sa")
    ap.add_argument("--tgt-file", default="test.hi")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--dataset-version", default="v2")
    ap.add_argument("--run-group-prefix", default="tlc_ablation",
                    help="experiments/v2/<prefix>/<model>/<variant>/... -- tlc_ablation (default, "
                         "12 formula/lambda/dim variants) or tlc_ablation_q3 (2 bucket-mode variants)")
    ap.add_argument("--out-root", default=None,
                    help="default: results/news_test/<run-group-prefix>")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--metrics", nargs="*", default=ALL_METRICS, choices=ALL_METRICS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", nargs="*", choices=[m["name"] for m in MODELS], default=None)
    ap.add_argument("--variants", nargs="*", default=VARIANTS,
                    help="default: the 12 tlc_ablation variants; pass explicitly for other prefixes "
                         "(e.g. --run-group-prefix tlc_ablation_q3 --variants predicted_target_length "
                         "oracle_ratio_class)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-comet", action="store_true")
    args = ap.parse_args()
    if args.out_root is None:
        args.out_root = os.path.join("results", "news_test", args.run_group_prefix)
    return args


def _oom_safe_translate(model, tokenizer, srcs, batch_size, **kw):
    bs = max(1, batch_size)
    while True:
        try:
            return translate(model, tokenizer, srcs, batch_size=bs, **kw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise RuntimeError("tlc_ablation news_test translate: OOM even at batch_size=1")
            bs = max(1, bs // 2)
            print(f"    [oom] retrying generation at batch_size={bs}")


def build_plan(args):
    models = [m for m in MODELS if args.models is None or m["name"] in args.models]
    plan = []
    for m in models:
        for variant in args.variants:
            run_group = f"{args.run_group_prefix}/{m['name']}/{variant}"
            cfg = RunConfig.resolve(CONFIG, SEED, args.dataset_version, m["model_key"], m["experiment"],
                                    run_group=run_group, experiments_root=args.experiments_root)
            ckpt = os.path.join(cfg.output_dir, "checkpoint-best")
            out_dir = os.path.join(args.out_root, m["name"], variant, CONFIG, f"seed_{SEED}")
            plan.append({"model": m["name"], "variant": variant, "cfg": cfg,
                        "checkpoint": ckpt, "out_dir": out_dir})
    return plan


def run_cell(args, cell, srcs, refs):
    cfg, ckpt, out_dir = cell["cfg"], cell["checkpoint"], cell["out_dir"]
    family = infer_model_family(cfg.base_model)
    print(f"[{cell['model']}/{cell['variant']}] loading {ckpt}")
    model, tokenizer = _load_ckpt(ckpt, family, args.device)
    tok_src = cfg.mbart_src_lang if family == "mbart" else cfg.src_lang
    try:
        preds = _oom_safe_translate(
            model, tokenizer, srcs, args.batch_size,
            family=family, tok_src_lang=tok_src,
            forced_bos_token_id=_forced_bos(cfg, tokenizer, family),
            max_input_length=cfg.max_input_length, max_new_tokens=cfg.max_target_length,
            num_beams=args.num_beams, device=args.device,
        )
        scores = compute_all(
            preds, refs, srcs, boundaries=cfg.bucket_boundaries, metrics=args.metrics,
            run_comet=("comet" in args.metrics) and not args.no_comet,
            max_gen_len=cfg.max_target_length, bertscore_lang="hi",
        )
    finally:
        del model
        torch.cuda.empty_cache()

    ensure_dir(out_dir)
    write_lines(os.path.join(out_dir, "source"), srcs)
    write_lines(os.path.join(out_dir, "references"), refs)
    write_lines(os.path.join(out_dir, "predictions"), preds)
    write_json(os.path.join(out_dir, "metrics.json"), {
        "model": cell["model"], "variant": cell["variant"], "config": CONFIG, "seed": SEED,
        "checkpoint": ckpt, "test_corpus": "news_test", "num_beams": args.num_beams,
        "n_examples": len(srcs), "metrics": scores,
    })
    bleu, chrf2, comet = scores.get("bleu"), scores.get("chrf2"), scores.get("comet")
    print(f"    -> BLEU={bleu:.2f} chrF2={chrf2:.2f}"
          f"{f' COMET={comet:.2f}' if comet is not None else ''}")


def summarize(args, plan):
    import json

    overall_keys = list(args.metrics)
    bucket_keys = [f"{BUCKET_LABELS_LONG[b]}_{m}" for b in BUCKET_IDS for m in args.metrics]
    all_keys = overall_keys + bucket_keys
    rows = {}
    for cell in plan:
        mpath = os.path.join(cell["out_dir"], "metrics.json")
        if not os.path.isfile(mpath):
            continue
        m = json.load(open(mpath))["metrics"]
        rows[(cell["model"], cell["variant"])] = {k: m[k] for k in all_keys if k in m}

    out_csv = os.path.join(args.out_root, "summary.csv")
    ensure_dir(args.out_root)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant"] + all_keys)
        for key in sorted(rows):
            w.writerow([key[0], key[1]] + [rows[key].get(k, "") for k in all_keys])
    print(f"\nwrote {out_csv}")

    print("\n-- OVERALL --")
    hdr = f"  {'model':8s} {'variant':12s} | " + " | ".join(f"{m:>10s}" for m in overall_keys)
    print(hdr)
    for key in sorted(rows):
        cells = [f"{rows[key].get(mk, '--'):>10}" if isinstance(rows[key].get(mk), str)
                else f"{rows[key].get(mk, float('nan')):10.2f}" for mk in overall_keys]
        print(f"  {key[0]:8s} {key[1]:12s} | " + " | ".join(cells))


def main():
    args = parse_args()
    srcs = read_lines(os.path.join(args.news_dir, args.src_file))
    refs = read_lines(os.path.join(args.news_dir, args.tgt_file))
    plan = build_plan(args)
    n_have = sum(1 for c in plan if os.path.isdir(c["checkpoint"]))
    print(f"TLC ablation news_test eval plan: {len(plan)} cells ({n_have} with checkpoint-best/)")

    if not args.summarize_only:
        for cell in plan:
            mpath = os.path.join(cell["out_dir"], "metrics.json")
            if os.path.isfile(mpath) and not args.overwrite:
                print(f"  [skip] {cell['model']}/{cell['variant']} already scored")
                continue
            if not os.path.isdir(cell["checkpoint"]):
                print(f"  [skip] {cell['model']}/{cell['variant']} -- no checkpoint-best/ ({cell['checkpoint']})")
                continue
            if args.dry_run:
                print(f"  [would run] {cell['model']}/{cell['variant']}")
                continue
            run_cell(args, cell, srcs, refs)

    if not args.dry_run:
        summarize(args, plan)


if __name__ == "__main__":
    main()
