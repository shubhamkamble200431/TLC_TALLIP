#!/usr/bin/env python
"""
Bucket-cap x repetition INFERENCE ablation (reviewer request).

Builds a source-length-bucket-EQUAL test set from the UNSAMPLED test pool
(data/<ds>/test.sa) and sweeps:

    columns : k sentences per bucket, k = round(f * M) for f in --fractions
              M = min over buckets of the pool's bucket count
                  (v2 -> B5 = 178 ; v1 -> B5 = 56).  Nested prefixes.
    rows    : --repetitions r  (each bucket's k sentences repeated r times,
              applied to ALL five buckets identically).

Cell = BLEU / chrF2 / COMET (+ length error, + per-bucket) over the
5*k*r-sentence set.  Every unique sentence is translated ONCE (max 5*M) and
re-used across all cells.

INFERENCE ONLY — over one frozen checkpoint (the best model after training).

Outputs:
    results/csv/bucket_repeat_ablation_<ds>.csv
    (via scripts/generate_bucket_repeat_heatmaps.py)
    results/figures/bucket_repeat_heatmap_<ds>_{bleu,chrf2,comet}.{png,pdf}

Example:
  python scripts/run_bucket_repeat_ablation.py \\
      --checkpoint experiments/v2/factorial/X-1101/seed_42/checkpoint-best \\
      --config X-1101 --dataset v2
"""
import _paths  # noqa: F401

import argparse
import csv
import os
import random
import sys

import numpy as np

from src.config import RunConfig, load_experiment_yaml
from src.data.buckets import BUCKET_IDS, DEFAULT_BOUNDARIES, assign_bucket, n_tokens
from src.evaluation.evaluate import _forced_bos, _load_ckpt   # reuse the exact eval path
from src.evaluation.generate import translate
from src.evaluation.metrics import corpus_bleu, corpus_chrf2, corpus_comet, length_metrics
from src.models.lang_codes import infer_model_family
from src.utils.io import ensure_dir, read_lines


def _bucket_pools(src, tgt, boundaries):
    pools = {b: [] for b in BUCKET_IDS}
    for s, t in zip(src, tgt):
        pools[assign_bucket(n_tokens(s), boundaries)].append((s, t))
    return pools


def deterministic_order(n, seed):
    """A fixed shuffled ordering of range(n) — same seed always yields the same order."""
    idx = list(range(n))
    random.Random(int(seed)).shuffle(idx)
    return idx


def pick_indices(repeat_mode, order_b, pool_len, k, j, seed, b):
    """
    Pool indices for bucket ``b``, copy ``j`` (0-based), asking for ``k`` sentences.

      duplicate : the SAME k-prefix of ``order_b`` for every copy j
                  -> corpus metrics are exactly repetition-invariant (fairness proof).
      resample  : a fresh deterministic shuffle per copy j, wrapping if k > pool_len
                  -> exposes sampling noise across draws.
    """
    if repeat_mode == "duplicate":
        return order_b[:k]
    if pool_len == 0:
        return []
    oj = deterministic_order(pool_len, seed + 1000 * (j + 1) + b)
    return [oj[t % pool_len] for t in range(k)]


def _score(preds, refs, srcs, run_comet, max_gen_len):
    if not preds:
        return {"bleu": float("nan"), "chrf2": float("nan"), "comet": float("nan"),
                "absolute_length_error": float("nan")}
    out = {
        "bleu": corpus_bleu(preds, refs),
        "chrf2": corpus_chrf2(preds, refs),
        "comet": corpus_comet(preds, refs, srcs) if run_comet else float("nan"),
    }
    out.update(length_metrics(preds, refs, max_gen_len=max_gen_len))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="frozen model dir (checkpoint-best)")
    ap.add_argument("--config", default="X-1101", help="X-ABCD of that checkpoint (for lang/beam settings)")
    ap.add_argument("--dataset", default="v2", choices=["v2", "v1"])
    ap.add_argument("--model", default="nllb")
    ap.add_argument("--experiment", default="bucket_repeat")
    ap.add_argument("--fractions", nargs="*", type=float, default=None)
    ap.add_argument("--repetitions", nargs="*", type=int, default=None)
    ap.add_argument("--M", type=int, default=None, help="override the per-bucket cap (default: min bucket count)")
    ap.add_argument("--repeat-mode", default=None, choices=["duplicate", "resample"])
    ap.add_argument("--sample-seed", type=int, default=None)
    ap.add_argument("--num-beams", type=int, default=None)
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    exp = load_experiment_yaml(args.experiment)
    fractions = args.fractions or exp.get("fractions", [0.2, 0.4, 0.6, 0.8, 1.0])
    repetitions = args.repetitions or exp.get("repetitions", list(range(1, 11)))
    repeat_mode = args.repeat_mode or exp.get("repeat_mode", "duplicate")
    seed = args.sample_seed if args.sample_seed is not None else exp.get("sample_seed", 42)
    num_beams = args.num_beams or exp.get("num_beams", 4)
    run_comet = not args.no_comet and "comet" in exp.get("metrics", ["bleu", "chrf2", "comet"])
    out_csv = args.out_csv or f"results/csv/bucket_repeat_ablation_{args.dataset}.csv"

    cfg = RunConfig.resolve(args.config, seed, args.dataset, args.model, "factorial")
    family = infer_model_family(cfg.base_model)
    boundaries = cfg.bucket_boundaries or DEFAULT_BOUNDARIES

    pool_dir = f"data/{args.dataset}"
    src = read_lines(os.path.join(pool_dir, "test.sa"))
    tgt = read_lines(os.path.join(pool_dir, "test.hi"))
    pools = _bucket_pools(src, tgt, boundaries)
    counts = {b: len(pools[b]) for b in BUCKET_IDS}
    M = args.M or min(counts.values())
    if M < 1:
        raise SystemExit(f"M={M} — every bucket must have >=1 sentence (counts {counts}).")

    # fraction -> k/bucket (>=1); drop fractions that collapse to a k already seen
    frac_k, seen_k = [], set()
    for f in sorted(set(fractions)):
        k = max(1, round(f * M))
        if k in seen_k:
            print(f"[bucket_repeat] note: f={f} -> k={k} duplicates an earlier column; skipped.")
            continue
        seen_k.add(k)
        frac_k.append((f, k))
    max_k = max(k for _, k in frac_k)
    print(f"[bucket_repeat] {args.dataset}  bucket counts {counts}  ->  M = {M}")
    print(f"[bucket_repeat] columns f->k: {frac_k}  (k up to {max_k}/bucket)  x  "
          f"repetitions {repetitions}  mode={repeat_mode}")

    # ---- deterministic per-bucket orderings ------------------------
    order = {b: deterministic_order(len(pools[b]), seed + b) for b in BUCKET_IDS}
    max_r = max(repetitions)

    def _picks(b, k, j):
        return pick_indices(repeat_mode, order[b], len(pools[b]), k, j, seed, b)

    # every pool index any cell could reference
    need = {b: set() for b in BUCKET_IDS}
    for _, k in frac_k:
        for j in range(max_r):
            for b in BUCKET_IDS:
                need[b].update(_picks(b, k, j))

    # ---- translate every needed unique sentence ONCE --------------------
    # exact same load / forced-BOS / decode path as src.evaluation.evaluate
    import torch

    model, tok = _load_ckpt(args.checkpoint, family, args.device)
    tok_src = cfg.mbart_src_lang if family == "mbart" else cfg.src_lang
    forced_bos = _forced_bos(cfg, tok, family)

    pred_cache = {}   # (bucket, pool_idx) -> prediction
    for b in BUCKET_IDS:
        ids = sorted(need[b])
        if not ids:
            continue
        preds = translate(model, tok, [pools[b][i][0] for i in ids], family=family, tok_src_lang=tok_src,
                          forced_bos_token_id=forced_bos, num_beams=num_beams,
                          max_input_length=cfg.max_input_length, max_new_tokens=cfg.max_target_length,
                          device=args.device)
        for i, p in zip(ids, preds):
            pred_cache[(b, i)] = p
        print(f"[bucket_repeat] translated B{b}: {len(ids)} unique sentences")
    del model
    torch.cuda.empty_cache()

    # ---- assemble every (fraction, repetition) cell --------------------
    # In `duplicate` mode a cell at repetition r is just r identical copies of
    # the r=1 cell, so every corpus/segment metric equals the r=1 value -- we
    # score the r=1 assembly ONCE per column and replay it for r>1 (only
    # n_examples grows).  That is also the cleanest demonstration of the
    # "cannot be gamed by duplication" point.  In `resample` mode each r is a
    # genuinely different draw, so every cell is scored.
    def _assemble(k, r):
        pb_p, pb_r, pb_s = {}, {}, {}
        ap, ar, as_ = [], [], []
        for b in BUCKET_IDS:
            bp, br, bs = [], [], []
            for j in range(r):
                picks = _picks(b, k, j)
                bp += [pred_cache[(b, i)] for i in picks]
                br += [pools[b][i][1] for i in picks]
                bs += [pools[b][i][0] for i in picks]
            pb_p[b], pb_r[b], pb_s[b] = bp, br, bs
            ap += bp; ar += br; as_ += bs
        return ap, ar, as_, pb_p, pb_r, pb_s

    def _score_cell(k, r):
        ap, ar, as_, pb_p, pb_r, pb_s = _assemble(k, r)
        agg = _score(ap, ar, as_, run_comet, cfg.max_target_length)
        per_bucket = {b: _score(pb_p[b], pb_r[b], pb_s[b], run_comet, cfg.max_target_length)
                      for b in BUCKET_IDS}
        return agg, per_bucket

    rows = []
    for f, k in frac_k:
        base = None
        for r in repetitions:
            if repeat_mode == "resample" or base is None:
                agg, per_bucket = _score_cell(k, r)
                if repeat_mode == "duplicate":
                    base = (agg, per_bucket)   # replay for the remaining r
                scored = True
            else:
                agg, per_bucket = base
                scored = False
            n_ex = len(BUCKET_IDS) * k * r

            row = {
                "dataset": args.dataset, "M": M, "fraction": f, "k_per_bucket": k,
                "repetitions": r, "n_examples": n_ex, "scored": bool(scored),
                "bleu": agg["bleu"], "chrf2": agg["chrf2"], "comet": agg.get("comet"),
                "length_error": agg["absolute_length_error"],
                # macro = unweighted mean over the 5 equal buckets
                "bleu_macro": float(np.nanmean([per_bucket[b]["bleu"] for b in BUCKET_IDS])),
                "chrf2_macro": float(np.nanmean([per_bucket[b]["chrf2"] for b in BUCKET_IDS])),
                "comet_macro": float(np.nanmean([per_bucket[b].get("comet", np.nan) for b in BUCKET_IDS]))
                if run_comet else None,
            }
            for b in BUCKET_IDS:
                row[f"b{b}_bleu"] = per_bucket[b]["bleu"]
                row[f"b{b}_chrf2"] = per_bucket[b]["chrf2"]
                row[f"b{b}_comet"] = per_bucket[b].get("comet")
            rows.append(row)
            tag = "" if scored else "  (replayed)"
            print(f"  f={f:.1f} (k={k})  r={r:>2}  n={n_ex:>5}  "
                  f"BLEU={agg['bleu']:.2f}  chrF2={agg['chrf2']:.2f}  "
                  f"COMET={agg.get('comet') if not np.isnan(agg.get('comet', float('nan'))) else '--'}{tag}")

    ensure_dir(os.path.dirname(os.path.abspath(out_csv)))
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[bucket_repeat] -> {out_csv}")
    if repeat_mode == "duplicate":
        print("[bucket_repeat] note: corpus BLEU/chrF2/COMET are mathematically invariant to uniform "
              "repetition -> rows are identical by design (this demonstrates the score cannot be "
              "gamed by duplication; the informative axis is k/bucket).")

    from subprocess import run as _run

    _run([sys.executable, os.path.join(os.path.dirname(__file__), "generate_bucket_repeat_heatmaps.py"),
          "--csv", out_csv, "--dataset", args.dataset], check=False)


if __name__ == "__main__":
    main()
