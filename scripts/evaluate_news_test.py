#!/usr/bin/env python
"""
Out-of-domain check: score every trained checkpoint (NLLB factorial + mBART-50
factorial -- all 16 X-ABCD configs x seeds 42/43/44) against the NEW news_test
corpus (data/news_test/test.sa -> test.hi), with every metric this repo
supports.  Separate from the main v2/v1 pipeline entirely: its own output
tree, never touches experiments/v2/{factorial,mbart50}/.../evaluation/.

Sanskrit -> Hindi only for now (test.sa / test.hi); test_manifest.tsv also
carries Gujarati/Marathi references (test.gu / test.mr) if a gu/mr pass is
wanted later -- not used here.

Output, one folder per (model, lang_pair, config, seed):
    <out-root>/<model_name>/<lang_pair>/<config>/seed_<seed>/
        source        the news_test source sentences (same for every cell)
        references    the news_test reference translations
        predictions   this checkpoint's beam-search output
        metrics.json  every requested metric, OVERALL and per source-length
                      bucket (B1 1-5 tok .. B5 21+ tok) -- compute_all()
                      produces both from one call, keyed "bleu" / "B1_1-5_bleu" / ...

Then a mean/variance-over-3-seeds summary table per (model, config):
    <out-root>/summary.csv  (+ printed to stdout)

Incremental: a (model, config, seed) whose metrics.json already exists is
skipped unless --overwrite; one whose checkpoint-best/ doesn't exist yet
(training not finished) is skipped with a note -- safe to re-run anytime as
more factorial/mbart50 runs complete.

Inference only (no gradients/optimizer state), fp16, so batch size can be
large -- default 128 comfortably fits a 600M-1B model in 40 GB; halves
automatically on a CUDA OOM.

Examples
--------
  # see the full plan without loading anything
  python scripts/evaluate_news_test.py --dry-run

  # just NLLB, just 4 configs, quick look
  python scripts/evaluate_news_test.py --models nllb --configs X-0000 X-1000 X-0100 X-1100

  # everything found so far
  python scripts/evaluate_news_test.py

  # rebuild only the summary table from whatever's already scored
  python scripts/evaluate_news_test.py --summarize-only
"""
import _paths  # noqa: F401

import argparse
import csv
import os
import statistics as stats
from collections import defaultdict

import torch

from src.config import ALL_CASE_IDS, RunConfig
from src.data.buckets import BUCKET_IDS, BUCKET_LABELS_LONG
from src.evaluation.evaluate import _forced_bos, _load_ckpt
from src.evaluation.generate import translate
from src.evaluation.metrics import compute_all
from src.models.lang_codes import infer_model_family
from src.utils.io import ensure_dir, read_lines, write_json, write_lines

MODELS = [
    {"name": "nllb", "model_key": "nllb", "experiment": "factorial", "run_group": "factorial",
     "configs": None},  # None -> ALL_CASE_IDS (or --configs override)
    {"name": "mbart50", "model_key": "mbart50_many_to_many", "experiment": "mbart50", "run_group": "mbart50",
     "configs": None},
    # the extra "baselines/" architectures -- X-0000 (all switches off) only, 7-epoch generic recipe
    {"name": "mt5", "model_key": "mt5_base", "experiment": "baseline_mt5", "run_group": "baselines/mt5",
     "configs": ["X-0000"]},
    {"name": "tf6", "model_key": "tf6", "experiment": "baseline_tf6", "run_group": "baselines/tf6",
     "configs": ["X-0000"]},
    {"name": "tf12", "model_key": "tf12", "experiment": "baseline_tf12", "run_group": "baselines/tf12",
     "configs": ["X-0000"]},
    {"name": "tf18", "model_key": "tf18", "experiment": "baseline_tf18", "run_group": "baselines/tf18",
     "configs": ["X-0000"]},
    {"name": "indictrans2", "model_key": "indictrans2_indic_indic_1b", "experiment": "baseline_indictrans2",
     "run_group": "baselines/indictrans2", "configs": ["X-0000"]},
]
SEEDS = [42, 43, 44]
LANG_PAIR = "sa-hi"
ALL_METRICS = ["bleu", "chrf2", "ter", "meteor", "bertscore", "comet"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--news-dir", default="data/news_test")
    ap.add_argument("--src-file", default="test.sa")
    ap.add_argument("--tgt-file", default="test.hi")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--dataset-version", default="v2", help="where the trained checkpoints live")
    ap.add_argument("--out-root", default="results/news_test")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="inference-only (no grads/optimizer state) -> 40GB has plenty of headroom; "
                         "halves automatically on OOM")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--metrics", nargs="*", default=ALL_METRICS, choices=ALL_METRICS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", nargs="*", choices=[m["name"] for m in MODELS], default=None)
    ap.add_argument("--configs", nargs="*", default=None, help="subset of X-ABCD ids; default all 16")
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, load nothing, run nothing")
    ap.add_argument("--summarize-only", action="store_true", help="skip evaluation, just rebuild the table")
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
                raise RuntimeError("news_test translate: OOM even at batch_size=1")
            bs = max(1, bs // 2)
            print(f"    [oom] retrying generation at batch_size={bs}")


def build_plan(args):
    models = [m for m in MODELS if args.models is None or m["name"] in args.models]
    plan = []
    for m in models:
        # per-model default (baselines/* only ever have X-0000); an explicit
        # --configs always wins, applied the same way to every selected model
        configs = args.configs or m["configs"] or ALL_CASE_IDS
        for cid in configs:
            for seed in args.seeds:
                cfg = RunConfig.resolve(
                    cid.upper(), seed, args.dataset_version, m["model_key"], m["experiment"],
                    run_group=m["run_group"], experiments_root=args.experiments_root,
                )
                ckpt = os.path.join(cfg.output_dir, "checkpoint-best")
                # v2 keeps its original (pre-existing) path with no version
                # segment, for back-compat with results already written there;
                # any OTHER training dataset (v1, ...) gets its own segment so
                # it can never collide with/overwrite the v2 results.
                dver_segment = "" if args.dataset_version == "v2" else args.dataset_version
                out_dir = os.path.join(args.out_root, m["name"], dver_segment, LANG_PAIR,
                                        cid.upper(), f"seed_{seed}")
                plan.append({"model": m["name"], "config": cid.upper(), "seed": seed,
                            "cfg": cfg, "checkpoint": ckpt, "out_dir": out_dir})
    return plan


def run_cell(args, cell, srcs, refs):
    cfg, ckpt, out_dir = cell["cfg"], cell["checkpoint"], cell["out_dir"]
    family = infer_model_family(cfg.base_model)
    print(f"[{cell['model']}/{cell['config']}/seed_{cell['seed']}] loading {ckpt}")
    model, tokenizer = _load_ckpt(ckpt, family, args.device,
                                  tokenizer_source=cfg.base_model if family == "indictrans2" else None)
    tok_src = cfg.mbart_src_lang if family == "mbart" else cfg.src_lang
    tok_tgt = cfg.mbart_tgt_lang if family == "mbart" else cfg.tgt_lang
    try:
        preds = _oom_safe_translate(
            model, tokenizer, srcs, args.batch_size,
            family=family, tok_src_lang=tok_src, tok_tgt_lang=tok_tgt,
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
        "model": cell["model"], "lang_pair": LANG_PAIR, "config": cell["config"], "seed": cell["seed"],
        "checkpoint": ckpt, "test_corpus": "news_test", "num_beams": args.num_beams,
        "n_examples": len(srcs), "metrics": scores,
    })
    bleu, chrf2, comet = scores.get("bleu"), scores.get("chrf2"), scores.get("comet")
    print(f"    -> BLEU={bleu:.2f} chrF2={chrf2:.2f}"
          f"{f' COMET={comet:.2f}' if comet is not None else ''}")


def _metric_keys(args):
    """'overall' keys (bleu, chrf2, ...) + one set per source-length bucket
    (B1_1-5_bleu, ..., B5_21+_comet, ...) -- compute_all() already produces
    both from a single call, we just need to know the full key list here."""
    overall = list(args.metrics)
    bucketed = [f"{BUCKET_LABELS_LONG[b]}_{m}" for b in BUCKET_IDS for m in args.metrics]
    return overall, bucketed


def summarize(args, plan):
    import glob
    import json

    overall_keys, bucket_keys = _metric_keys(args)
    all_keys = overall_keys + bucket_keys
    rows = defaultdict(lambda: defaultdict(list))  # rows[(model,dver,config)][metric_key] = [values]
    n_seeds = defaultdict(int)
    # Scan the WHOLE out_root tree, not just this invocation's `plan` -- a
    # --models/--configs-restricted run must not blow away every other
    # architecture's rows that are already sitting on disk (this clobbered
    # the combined summary.csv more than once before this fix: scoping a run
    # to one architecture silently dropped every other architecture's row).
    # Path depth varies by dataset_version: v2 is <model>/<lang_pair>/<config>/
    # seed_*/metrics.json (no version segment, back-compat); any other
    # dataset_version inserts its own segment: <model>/<dver>/<lang_pair>/
    # <config>/seed_*/metrics.json (see build_plan's dver_segment). Parse
    # whichever depth is actually there instead of assuming v2's.
    for mpath in sorted(glob.glob(os.path.join(args.out_root, "*", "**", "metrics.json"), recursive=True)):
        rel_parts = os.path.relpath(mpath, args.out_root).split(os.sep)
        model_name = rel_parts[0]
        cid, seed_name = rel_parts[-3], rel_parts[-2]
        middle = rel_parts[1:-3]  # [lang_pair] for v2, [dver, lang_pair] otherwise
        if len(middle) == 1:
            dver = "v2"
        elif len(middle) == 2:
            dver = middle[0]
        else:
            continue  # unrecognized layout -- skip rather than misattribute
        m = json.load(open(mpath))["metrics"]
        key = (model_name, dver, cid)
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
    header = ["model", "dataset_version", "config", "n_seeds"] + \
        [f"{k}_{stat}" for k in all_keys for stat in ("mean", "var")]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for key in sorted(rows, key=lambda k: (k[0], k[1], k[2])):
            row = [key[0], key[1], key[2], n_seeds[key]]
            for mk in all_keys:
                row += list(mean_var(rows[key][mk]))
            w.writerow(row)
    print(f"\n{'=' * 100}\n  news_test summary (mean / var over up to {len(SEEDS)} seeds)\n"
          f"  overall + per-bucket (B1 1-5 tok .. B5 21+ tok) for {args.metrics}\n"
          f"  full table -> {out_csv}\n{'=' * 100}")

    # -- console: OVERALL table (every requested metric) --------------------
    print("\n-- OVERALL --")
    hdr = f"  {'model':8s} {'dver':5s} {'config':7s} {'n':>2s} | " + " | ".join(f"{m:>16s}" for m in overall_keys)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key in sorted(rows, key=lambda k: (k[0], k[1], k[2])):
        cells = []
        for mk in overall_keys:
            mean, var = mean_var(rows[key][mk])
            cells.append(f"{'--':>16s}" if mean == "" else f"{mean:7.2f} (v{var:5.3f})")
        print(f"  {key[0]:8s} {key[1]:5s} {key[2]:7s} {n_seeds[key]:>2d} | " + " | ".join(cells))

    # -- console: PER-BUCKET table, one metric at a time (BLEU first) -------
    bucket_labels = [BUCKET_LABELS_LONG[b] for b in BUCKET_IDS]
    for metric in overall_keys:
        print(f"\n-- PER-BUCKET {metric.upper()} (mean, var) --")
        hdr = f"  {'model':8s} {'dver':5s} {'config':7s} | " + " | ".join(f"{b:>16s}" for b in bucket_labels)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for key in sorted(rows, key=lambda k: (k[0], k[1], k[2])):
            cells = []
            for b in BUCKET_IDS:
                mean, var = mean_var(rows[key][f"{BUCKET_LABELS_LONG[b]}_{metric}"])
                cells.append(f"{'--':>16s}" if mean == "" else f"{mean:7.2f} (v{var:5.3f})")
            print(f"  {key[0]:8s} {key[1]:5s} {key[2]:7s} | " + " | ".join(cells))
    print()


def main():
    args = parse_args()
    src_path = os.path.join(args.news_dir, args.src_file)
    tgt_path = os.path.join(args.news_dir, args.tgt_file)
    srcs = read_lines(src_path)
    refs = read_lines(tgt_path)
    if len(srcs) != len(refs):
        raise SystemExit(f"{src_path} has {len(srcs)} lines but {tgt_path} has {len(refs)}")

    plan = build_plan(args)
    n_have_ckpt = sum(1 for c in plan if os.path.isdir(c["checkpoint"]))
    print(f"news_test eval plan: {len(plan)} (model,config,seed) cells "
          f"({n_have_ckpt} with a checkpoint-best/ so far), "
          f"{len(srcs)} sentences from {src_path}, lang_pair={LANG_PAIR}, metrics={args.metrics}")

    if not args.summarize_only:
        for cell in plan:
            mpath = os.path.join(cell["out_dir"], "metrics.json")
            if os.path.isfile(mpath) and not args.overwrite:
                print(f"  [skip] {cell['model']}/{cell['config']}/seed_{cell['seed']} already scored")
                continue
            if not os.path.isdir(cell["checkpoint"]):
                print(f"  [skip] {cell['model']}/{cell['config']}/seed_{cell['seed']} "
                      f"-- no checkpoint-best/ yet ({cell['checkpoint']})")
                continue
            if args.dry_run:
                print(f"  [would run] {cell['model']}/{cell['config']}/seed_{cell['seed']} "
                      f"ckpt={cell['checkpoint']}")
                continue
            run_cell(args, cell, srcs, refs)

    if not args.dry_run:
        summarize(args, plan)


if __name__ == "__main__":
    main()
