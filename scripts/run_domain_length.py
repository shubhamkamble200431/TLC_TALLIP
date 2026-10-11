#!/usr/bin/env python
"""
Domain x length-bucket evaluation (brief section 16).

Runs a frozen checkpoint over each configured domain corpus and reports
BLEU / chrF2 / COMET / length error per source-length bucket.

Outputs:
    results/csv/domain_length.csv              (domain, bucket, metric, ...)
    results/figures/domain_length_<metric>.{png,pdf}

Example:
  python scripts/run_domain_length.py \\
      --checkpoint experiments/v2/factorial/X-1101/seed_42/checkpoint-best \\
      --config X-1101 --seed 42
"""
import _paths  # noqa: F401

import argparse
import csv
import os
import sys

import numpy as np

from src.config import RunConfig, load_experiment_yaml
from src.data.buckets import assign_bucket, n_tokens, BUCKET_IDS
from src.evaluation.generate import translate
from src.evaluation.metrics import compute_all
from src.models.lang_codes import infer_model_family
from src.utils.io import ensure_dir, read_lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="X-1101")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="v2")
    ap.add_argument("--model", default="nllb")
    ap.add_argument("--experiment", default="domain_length")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--out-csv", default="results/csv/domain_length.csv")
    ap.add_argument("--fig-dir", default="results/figures")
    args = ap.parse_args()

    exp = load_experiment_yaml(args.experiment)
    cfg = RunConfig.resolve(args.config, args.seed, args.dataset, args.model, "factorial")
    family = infer_model_family(cfg.base_model)
    metrics = exp.get("metrics", ["bleu", "chrf2", "comet", "absolute_length_error"])

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    if family == "mbart":
        from transformers import MBartForConditionalGeneration as M
    else:
        from transformers import AutoModelForSeq2SeqLM as M
    import torch

    model = M.from_pretrained(args.checkpoint, torch_dtype=torch.float16).to(args.device).eval()
    tok_src = cfg.mbart_src_lang if family == "mbart" else cfg.src_lang
    tok_tgt = cfg.mbart_tgt_lang if family == "mbart" else cfg.tgt_lang
    forced_bos = None
    if cfg.force_bos_target_lang:
        try:
            forced_bos = tok.convert_tokens_to_ids(tok_tgt)
        except Exception:
            forced_bos = None

    rows = []
    for dom in exp.get("domains", []):
        sp = os.path.join(dom["data_dir"], dom["src"])
        tp = os.path.join(dom["data_dir"], dom["tgt"])
        if not (os.path.isfile(sp) and os.path.isfile(tp)):
            print(f"[domain_length] SKIP {dom['name']} — no files at {dom['data_dir']}")
            continue
        srcs, refs = read_lines(sp), read_lines(tp)
        n = min(len(srcs), len(refs))
        srcs, refs = srcs[:n], refs[:n]
        preds = translate(model, tok, srcs, family=family, tok_src_lang=tok_src,
                          forced_bos_token_id=forced_bos, num_beams=args.num_beams,
                          max_input_length=cfg.max_input_length, max_new_tokens=cfg.max_target_length,
                          device=args.device)
        buckets = [assign_bucket(n_tokens(s), cfg.bucket_boundaries) for s in srcs]
        for b in BUCKET_IDS:
            idx = [i for i, bk in enumerate(buckets) if bk == b]
            if not idx:
                continue
            sc = compute_all([preds[i] for i in idx], [refs[i] for i in idx],
                             [srcs[i] for i in idx], boundaries=cfg.bucket_boundaries,
                             metrics=["bleu", "chrf2", "comet"], run_comet=not args.no_comet,
                             max_gen_len=cfg.max_target_length)
            rows.append({
                "domain": dom["name"], "bucket": f"B{b}", "n_examples": len(idx),
                "bleu": sc.get("bleu"), "chrf2": sc.get("chrf2"), "comet": sc.get("comet"),
                "absolute_length_error": sc.get("absolute_length_error"),
            })
            print(f"  {dom['name']:<16} B{b}: n={len(idx):<5} BLEU={sc.get('bleu'):.2f}")

    if not rows:
        sys.exit("[domain_length] no domain corpora found — nothing written.")
    ensure_dir(os.path.dirname(os.path.abspath(args.out_csv)))
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[domain_length] -> {args.out_csv}")

    # heatmaps
    try:
        import pandas as pd

        from src.visualization.heatmaps import annotated_heatmap

        df = pd.DataFrame(rows)
        for metric in ("bleu", "comet", "absolute_length_error"):
            piv = df.pivot_table(index="domain", columns="bucket", values=metric)
            piv = piv.reindex(columns=[f"B{b}" for b in BUCKET_IDS])
            annotated_heatmap(piv.values, list(piv.index), list(piv.columns),
                              os.path.join(args.fig_dir, f"domain_length_{metric}"),
                              title=f"Domain x length bucket — {metric}",
                              xlabel="source-length bucket", ylabel="domain",
                              cbar_label=metric, cmap=exp.get("heatmap", {}).get("colormap", "viridis"))
    except Exception as exc:
        print(f"[domain_length] heatmap skipped: {exc}")


if __name__ == "__main__":
    main()
