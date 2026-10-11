#!/usr/bin/env python
"""
Zero-shot cross-lingual check: take the 6 v1-trained checkpoints (NLLB
factorial + mBART-50, each fine-tuned ONLY on Sanskrit -> Hindi) and decode
the SAME news_test Sanskrit source sentences into Gujarati and Marathi
instead, by forcing a different target-language token at generation time --
no retraining, no new checkpoints. Scored against the Gujarati/Marathi
references that already ship with data/news_test (test.gu / test.mr).

Both NLLB-200 and mBART-50-many-to-many are natively multilingual and carry
dedicated Gujarati/Marathi target tokens (guj_Gujr / mar_Deva for NLLB,
gu_IN / mr_IN for mBART-50) even though these checkpoints were only ever
fine-tuned toward Hindi -- this measures how much of that multilingual
decoding ability survives sa-hi fine-tuning with each X-ABCD factor
combination on.

Output, one folder per (model, lang_pair, config, seed):
    <out-root>/<model_name>/<lang_pair>/<config>/seed_<seed>/
        source        news_test Sanskrit source (same for every cell)
        references    news_test Gujarati or Marathi references
        predictions   this checkpoint's beam-search output, decoded into
                      the OTHER language via a forced BOS override
        metrics.json  every requested metric, overall + per source-length
                      bucket

Then a mean/variance-over-3-seeds summary table:
    <out-root>/summary.csv  (+ printed to stdout)

Incremental, same convention as evaluate_news_test.py: skip if metrics.json
exists (unless --overwrite); skip if checkpoint-best/ is missing.
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
from src.evaluation.evaluate import _load_ckpt
from src.evaluation.generate import translate
from src.evaluation.metrics import compute_all
from src.models.lang_codes import infer_model_family
from src.utils.io import ensure_dir, read_lines, write_json, write_lines

MODELS = [
    {"name": "nllb", "model_key": "nllb", "experiment": "factorial", "run_group": "factorial"},
    {"name": "mbart50", "model_key": "mbart50_many_to_many", "experiment": "mbart50", "run_group": "mbart50"},
]
CONFIGS = ["X-0000", "X-1000", "X-0100", "X-0010", "X-0001", "X-1100"]
SEEDS = [42, 43, 44]
SRC_LANG_PAIR_PREFIX = "sa"

# target language: NLLB token, mBART-50 token, news_test reference file, bertscore lang code
TARGET_LANGS = {
    "gu": {"nllb": "guj_Gujr", "mbart": "gu_IN", "ref_file": "test.gu", "bertscore_lang": "gu"},
    "mr": {"nllb": "mar_Deva", "mbart": "mr_IN", "ref_file": "test.mr", "bertscore_lang": "mr"},
}
ALL_METRICS = ["bleu", "chrf2", "ter", "meteor", "bertscore", "comet"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--news-dir", default="data/news_test")
    ap.add_argument("--src-file", default="test.sa")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--dataset-version", default="v1", help="where the trained checkpoints live")
    ap.add_argument("--out-root", default="results/news_test_crosslingual")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--metrics", nargs="*", default=ALL_METRICS, choices=ALL_METRICS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", nargs="*", choices=[m["name"] for m in MODELS], default=None)
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--target-langs", nargs="*", choices=list(TARGET_LANGS), default=list(TARGET_LANGS))
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
                raise RuntimeError("news_test_crosslingual translate: OOM even at batch_size=1")
            bs = max(1, bs // 2)
            print(f"    [oom] retrying generation at batch_size={bs}")


def _forced_bos_for_code(tokenizer, lang_code):
    fb = None
    try:
        fb = tokenizer.convert_tokens_to_ids(lang_code)
    except Exception:
        pass
    if (fb is None or fb == tokenizer.unk_token_id) and hasattr(tokenizer, "lang_code_to_id"):
        fb = tokenizer.lang_code_to_id.get(lang_code)
    return fb


def build_plan(args):
    models = [m for m in MODELS if args.models is None or m["name"] in args.models]
    plan = []
    for m in models:
        for cid in args.configs:
            for seed in args.seeds:
                cfg = RunConfig.resolve(
                    cid.upper(), seed, args.dataset_version, m["model_key"], m["experiment"],
                    run_group=m["run_group"], experiments_root=args.experiments_root,
                )
                ckpt = os.path.join(cfg.output_dir, "checkpoint-best")
                for tgt in args.target_langs:
                    lang_pair = f"{SRC_LANG_PAIR_PREFIX}-{tgt}"
                    out_dir = os.path.join(args.out_root, m["name"], lang_pair, cid.upper(), f"seed_{seed}")
                    plan.append({"model": m["name"], "config": cid.upper(), "seed": seed,
                                "target_lang": tgt, "lang_pair": lang_pair,
                                "cfg": cfg, "checkpoint": ckpt, "out_dir": out_dir})
    return plan


def run_cell(args, cell, srcs, refs):
    cfg, ckpt, out_dir = cell["cfg"], cell["checkpoint"], cell["out_dir"]
    tgt = cell["target_lang"]
    family = infer_model_family(cfg.base_model)
    tgt_code = TARGET_LANGS[tgt]["mbart" if family == "mbart" else "nllb"]
    print(f"[{cell['model']}/{cell['lang_pair']}/{cell['config']}/seed_{cell['seed']}] "
          f"loading {ckpt}  (forcing target={tgt_code})")
    model, tokenizer = _load_ckpt(ckpt, family, args.device)
    tok_src = cfg.mbart_src_lang if family == "mbart" else cfg.src_lang
    forced_bos = _forced_bos_for_code(tokenizer, tgt_code)
    if forced_bos is None:
        raise RuntimeError(f"no token id for target language {tgt_code!r} ({family} tokenizer)")
    try:
        preds = _oom_safe_translate(
            model, tokenizer, srcs, args.batch_size,
            family=family, tok_src_lang=tok_src,
            forced_bos_token_id=forced_bos,
            max_input_length=cfg.max_input_length, max_new_tokens=cfg.max_target_length,
            num_beams=args.num_beams, device=args.device,
        )
        scores = compute_all(
            preds, refs, srcs, boundaries=cfg.bucket_boundaries, metrics=args.metrics,
            run_comet=("comet" in args.metrics) and not args.no_comet,
            max_gen_len=cfg.max_target_length, bertscore_lang=TARGET_LANGS[tgt]["bertscore_lang"],
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
        "forced_target_token": tgt_code, "num_beams": args.num_beams,
        "n_examples": len(srcs), "metrics": scores,
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
    rows = defaultdict(lambda: defaultdict(list))  # rows[(model,lang_pair,config)][metric_key] = [values]
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
    print(f"\n{'=' * 100}\n  news_test CROSS-LINGUAL summary (mean / var over up to {len(SEEDS)} seeds)\n"
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
    src_path = os.path.join(args.news_dir, args.src_file)
    srcs = read_lines(src_path)
    refs_by_lang = {}
    for tgt in args.target_langs:
        rp = os.path.join(args.news_dir, TARGET_LANGS[tgt]["ref_file"])
        refs = read_lines(rp)
        if len(refs) != len(srcs):
            raise SystemExit(f"{src_path} has {len(srcs)} lines but {rp} has {len(refs)}")
        refs_by_lang[tgt] = refs

    plan = build_plan(args)
    n_have_ckpt = sum(1 for c in plan if os.path.isdir(c["checkpoint"]))
    print(f"news_test cross-lingual eval plan: {len(plan)} (model,lang_pair,config,seed) cells "
          f"({n_have_ckpt} with a checkpoint-best/ so far), {len(srcs)} source sentences, "
          f"targets={args.target_langs}, metrics={args.metrics}")

    if not args.summarize_only:
        for cell in plan:
            mpath = os.path.join(cell["out_dir"], "metrics.json")
            if os.path.isfile(mpath) and not args.overwrite:
                print(f"  [skip] {cell['model']}/{cell['lang_pair']}/{cell['config']}/seed_{cell['seed']} "
                      f"already scored")
                continue
            if not os.path.isdir(cell["checkpoint"]):
                print(f"  [skip] {cell['model']}/{cell['lang_pair']}/{cell['config']}/seed_{cell['seed']} "
                      f"-- no checkpoint-best/ yet ({cell['checkpoint']})")
                continue
            if args.dry_run:
                print(f"  [would run] {cell['model']}/{cell['lang_pair']}/{cell['config']}/seed_{cell['seed']} "
                      f"ckpt={cell['checkpoint']}")
                continue
            run_cell(args, cell, srcs, refs_by_lang[cell["target_lang"]])

    if not args.dry_run:
        summarize(args, plan)


if __name__ == "__main__":
    main()
