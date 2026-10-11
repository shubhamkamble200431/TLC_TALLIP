#!/usr/bin/env python
"""
Out-of-domain check for the REAL (actually fine-tuned, not zero-shot)
Sanskrit->Gujarati and Sanskrit->Marathi checkpoints that already exist under
experiments/san-guj/{factorial,mbart50}/X-*/seed_*/ and
experiments/san-mar/{factorial,mbart50}/X-*/seed_*/ -- 6 X-ABCD configs
(X-0000, X-1000, X-0100, X-0010, X-0001, X-1100) x 2 models (nllb factorial,
mbart50), trained directly on data/indic_data/san-guj and san-mar.

These runs predate this session (trained 2026-09-19/20) and were never
resolved through RunConfig.resolve()/configs/datasets/<version>.yaml here --
there is no configs/datasets/san-guj.yaml or san-mar.yaml -- so each run's
own saved config.yaml (RunConfig.from_dict) is the sole source of truth:
base_model, tgt_lang/mbart_tgt_lang, bucket_boundaries etc. are read back
exactly as trained, not re-derived.

Source: data/news_test/test.sa (same news_test corpus used throughout).
References: data/news_test/test.gu / test.mr.

Output, one folder per (model, lang_pair, config, seed):
    <out-root>/<model_name>/<lang_pair>/<config>/seed_<seed>/
        source / references / predictions / metrics.json
Then a mean/variance summary (only 1 seed exists today, so var=0; written so
re-running after any additional seeds are trained aggregates correctly):
    <out-root>/summary.csv
"""
import _paths  # noqa: F401

import argparse
import csv
import glob
import os
import statistics as stats
from collections import defaultdict

import torch
import yaml

from src.config import RunConfig
from src.data.buckets import BUCKET_IDS, BUCKET_LABELS_LONG
from src.evaluation.evaluate import _forced_bos, _load_ckpt
from src.evaluation.generate import translate
from src.evaluation.metrics import compute_all
from src.models.lang_codes import infer_model_family
from src.utils.io import ensure_dir, read_lines, write_json, write_lines

LANG_GROUPS = [
    {"dataset_version": "san-guj", "ref_file": "test.gu", "lang_pair": "sa-gu", "bertscore_lang": "gu"},
    {"dataset_version": "san-mar", "ref_file": "test.mr", "lang_pair": "sa-mr", "bertscore_lang": "mr"},
]
RUN_GROUPS = [
    {"name": "nllb", "run_group": "factorial"},
    {"name": "mbart50", "run_group": "mbart50"},
]
ALL_METRICS = ["bleu", "chrf2", "ter", "meteor", "bertscore", "comet"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--news-dir", default="data/news_test")
    ap.add_argument("--src-file", default="test.sa")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--out-root", default="results/news_test_indic")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--metrics", nargs="*", default=ALL_METRICS, choices=ALL_METRICS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", nargs="*", choices=[m["name"] for m in RUN_GROUPS], default=None)
    ap.add_argument("--lang-pairs", nargs="*", choices=[g["lang_pair"] for g in LANG_GROUPS], default=None)
    ap.add_argument("--configs", nargs="*", default=None, help="subset of X-ABCD ids; default: whatever's on disk")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-comet", action="store_true")
    return ap.parse_args()


def _oom_safe_translate(model, tokenizer, srcs, batch_size, **kw):
    bs = max(1, batch_size)
    while True:
        try:
            return translate(model, tokenizer, srcs, batch_size=bs, **kw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise RuntimeError("news_test_indic translate: OOM even at batch_size=1")
            bs = max(1, bs // 2)
            print(f"    [oom] retrying generation at batch_size={bs}")


def build_plan(args):
    models = [m for m in RUN_GROUPS if args.models is None or m["name"] in args.models]
    groups = [g for g in LANG_GROUPS if args.lang_pairs is None or g["lang_pair"] in args.lang_pairs]
    plan = []
    for m in models:
        for g in groups:
            base = os.path.join(args.experiments_root, g["dataset_version"], m["run_group"])
            cfg_paths = sorted(glob.glob(os.path.join(base, "X-*", "seed_*", "config.yaml")))
            for cfg_path in cfg_paths:
                run_dir = os.path.dirname(cfg_path)
                seed_name = os.path.basename(run_dir)          # "seed_42"
                cid = os.path.basename(os.path.dirname(run_dir))  # "X-0000"
                if args.configs and cid not in [c.upper() for c in args.configs]:
                    continue
                seed = int(seed_name.replace("seed_", ""))
                with open(cfg_path) as f:
                    cfg = RunConfig.from_dict(yaml.safe_load(f))
                ckpt = os.path.join(run_dir, "checkpoint-best")
                out_dir = os.path.join(args.out_root, m["name"], g["lang_pair"], cid, seed_name)
                plan.append({"model": m["name"], "lang_pair": g["lang_pair"], "config": cid, "seed": seed,
                            "cfg": cfg, "checkpoint": ckpt, "out_dir": out_dir,
                            "ref_file": g["ref_file"], "bertscore_lang": g["bertscore_lang"]})
    return plan


def run_cell(args, cell, srcs, refs):
    cfg, ckpt, out_dir = cell["cfg"], cell["checkpoint"], cell["out_dir"]
    family = infer_model_family(cfg.base_model)
    print(f"[{cell['model']}/{cell['lang_pair']}/{cell['config']}/seed_{cell['seed']}] loading {ckpt}")
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
            max_gen_len=cfg.max_target_length, bertscore_lang=cell["bertscore_lang"],
        )
    finally:
        del model
        torch.cuda.empty_cache()

    ensure_dir(out_dir)
    write_lines(os.path.join(out_dir, "source"), srcs)
    write_lines(os.path.join(out_dir, "references"), refs)
    write_lines(os.path.join(out_dir, "predictions"), preds)
    write_json(os.path.join(out_dir, "metrics.json"), {
        "model": cell["model"], "lang_pair": cell["lang_pair"], "config": cell["config"],
        "seed": cell["seed"], "checkpoint": ckpt, "test_corpus": "news_test",
        "num_beams": args.num_beams, "n_examples": len(srcs), "metrics": scores,
    })
    bleu, chrf2, comet = scores.get("bleu"), scores.get("chrf2"), scores.get("comet")
    print(f"    -> BLEU={bleu:.2f} chrF2={chrf2:.2f}"
          f"{f' COMET={comet:.2f}' if comet is not None else ''}")


def _metric_keys(args):
    overall = list(args.metrics)
    bucketed = [f"{BUCKET_LABELS_LONG[b]}_{m}" for b in BUCKET_IDS for m in args.metrics]
    return overall, bucketed


def summarize(args, plan):
    import json

    overall_keys, bucket_keys = _metric_keys(args)
    all_keys = overall_keys + bucket_keys
    rows = defaultdict(lambda: defaultdict(list))
    n_seeds = defaultdict(int)
    for cell in plan:
        mpath = os.path.join(cell["out_dir"], "metrics.json")
        if not os.path.isfile(mpath):
            continue
        m = json.load(open(mpath))["metrics"]
        key = (cell["model"], cell["lang_pair"], cell["config"])
        n_seeds[key] += 1
        for mk in all_keys:
            if mk in m:
                rows[key][mk].append(m[mk])

    def mean_var(vals):
        if not vals:
            return "", ""
        mean = round(stats.mean(vals), 3)
        var = round(stats.variance(vals), 4) if len(vals) > 1 else 0.0
        return mean, var

    out_csv = os.path.join(args.out_root, "summary.csv")
    ensure_dir(args.out_root)
    header = ["model", "lang_pair", "config", "n_seeds"] + \
        [f"{k}_{stat}" for k in all_keys for stat in ("mean", "var")]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for key in sorted(rows, key=lambda k: (k[0], k[1], k[2])):
            row = [key[0], key[1], key[2], n_seeds[key]]
            for mk in all_keys:
                row += list(mean_var(rows[key][mk]))
            w.writerow(row)
    print(f"\n{'=' * 100}\n  news_test INDIC (san-guj / san-mar) summary (n_seeds as available)\n"
          f"  overall + per-bucket for {args.metrics}\n  full table -> {out_csv}\n{'=' * 100}")

    print("\n-- OVERALL --")
    hdr = f"  {'model':8s} {'lang_pair':10s} {'config':7s} {'n':>2s} | " + \
        " | ".join(f"{m:>16s}" for m in overall_keys)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key in sorted(rows, key=lambda k: (k[0], k[1], k[2])):
        cells = []
        for mk in overall_keys:
            mean, var = mean_var(rows[key][mk])
            cells.append(f"{'--':>16s}" if mean == "" else f"{mean:7.2f} (v{var:5.3f})")
        print(f"  {key[0]:8s} {key[1]:10s} {key[2]:7s} {n_seeds[key]:>2d} | " + " | ".join(cells))
    print()


def main():
    args = parse_args()
    news_src = read_lines(os.path.join(args.news_dir, args.src_file))
    refs_cache = {}

    plan = build_plan(args)
    n_have_ckpt = sum(1 for c in plan if os.path.isdir(c["checkpoint"]))
    print(f"news_test INDIC eval plan: {len(plan)} (model,lang_pair,config,seed) cells "
          f"({n_have_ckpt} with a checkpoint-best/ so far), {len(news_src)} source sentences")

    if not args.summarize_only:
        for cell in plan:
            mpath = os.path.join(cell["out_dir"], "metrics.json")
            if os.path.isfile(mpath) and not args.overwrite:
                print(f"  [skip] {cell['model']}/{cell['lang_pair']}/{cell['config']}/seed_{cell['seed']} "
                      f"already scored")
                continue
            if not os.path.isdir(cell["checkpoint"]):
                print(f"  [skip] {cell['model']}/{cell['lang_pair']}/{cell['config']}/seed_{cell['seed']} "
                      f"-- no checkpoint-best/ ({cell['checkpoint']})")
                continue
            if args.dry_run:
                print(f"  [would run] {cell['model']}/{cell['lang_pair']}/{cell['config']}/seed_{cell['seed']} "
                      f"ckpt={cell['checkpoint']}")
                continue
            rf = cell["ref_file"]
            if rf not in refs_cache:
                refs = read_lines(os.path.join(args.news_dir, rf))
                if len(refs) != len(news_src):
                    raise SystemExit(f"{args.src_file} has {len(news_src)} lines but {rf} has {len(refs)}")
                refs_cache[rf] = refs
            run_cell(args, cell, news_src, refs_cache[rf])

    if not args.dry_run:
        summarize(args, plan)


if __name__ == "__main__":
    main()
