"""
Paired bootstrap resampling significance test (Koehn, 2004) between the
best-2 news_test configs of each architecture (NLLB, mBART-50).

For each architecture:
  1. Rank all evaluated configs by mean overall BLEU (over available seeds,
     from results/news_test/<arch>/sa-hi/<CFG>/seed_*/metrics.json) and take
     the top 2.
  2. For each of those 2 configs, use a single representative seed run (the
     seed whose overall BLEU is closest to that config's 3-seed mean), since
     paired bootstrap compares two sentence-aligned systems on a fixed test
     set, not seed variance.
  3. Build a fixed evaluation pool: --n-per-bucket (default 200) sentences
     drawn (without replacement, seeded) from each of the 5 news_test length
     buckets (B1..B5), using data/news_test/test_manifest.tsv -- i.e. a fixed
     N = 5 * n_per_bucket subset of the 2500 news_test sentences.
  4. Draw max(--bootstrap-sizes) resamples (with replacement, size N each)
     from that pool ONCE; for each requested bootstrap count B in
     --bootstrap-sizes, use the first B of those resamples (nested prefixes,
     so larger B is a strict refinement of smaller B rather than an
     independent draw -- shows convergence honestly).
  5. Per resample, recompute corpus BLEU and chrF2 (sacrebleu, chrF2 =
     chrF++ / word_order=2, matching this repo's eval convention) for both
     systems on the resampled multiset; collect the paired differences.
  6. From the difference distribution report: mean diff, 95% CI (2.5/97.5
     percentiles -- two-sided), one-sided p-value (fraction of resamples
     where the nominally-worse system is >= the nominally-better one) and
     two-sided p-value (2x the smaller tail, capped at 1).

Not GPU-bound -- pure CPU (string metrics only), reuses on-disk predictions
already produced by evaluate_news_test.py. Safe to re-run: each arch's top-2
configs are only known once ranking data exists, so this must run AFTER the
news_test eval job(s) for that architecture have completed.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import sacrebleu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO, "results", "news_test")
MANIFEST = os.path.join(REPO, "data", "news_test", "test_manifest.tsv")
LANG_PAIR = "sa-hi"


def load_manifest_buckets():
    buckets = {}
    with open(MANIFEST) as f:
        header = f.readline().strip().split("\t")
        idx_i, bucket_i = header.index("index"), header.index("bucket")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            idx0 = int(parts[idx_i]) - 1  # manifest is 1-indexed -> 0-indexed line number
            bucket = parts[bucket_i]
            buckets.setdefault(bucket, []).append(idx0)
    return buckets


def rank_configs(arch):
    pattern = os.path.join(RESULTS_ROOT, arch, LANG_PAIR, "X-*")
    scored = []
    for cfg_dir in sorted(glob.glob(pattern)):
        cfg = os.path.basename(cfg_dir)
        seed_bleus = {}
        for seed_dir in sorted(glob.glob(os.path.join(cfg_dir, "seed_*"))):
            mpath = os.path.join(seed_dir, "metrics.json")
            if not os.path.isfile(mpath):
                continue
            with open(mpath) as f:
                m = json.load(f)
            seed = os.path.basename(seed_dir).replace("seed_", "")
            seed_bleus[seed] = m["metrics"]["bleu"]
        if not seed_bleus:
            continue
        mean_bleu = sum(seed_bleus.values()) / len(seed_bleus)
        rep_seed = min(seed_bleus, key=lambda s: abs(seed_bleus[s] - mean_bleu))
        scored.append({"config": cfg, "mean_bleu": mean_bleu, "seed_bleus": seed_bleus,
                        "rep_seed": rep_seed})
    scored.sort(key=lambda r: r["mean_bleu"], reverse=True)
    return scored


def load_run(arch, cfg, seed):
    d = os.path.join(RESULTS_ROOT, arch, LANG_PAIR, cfg, f"seed_{seed}")
    with open(os.path.join(d, "predictions")) as f:
        preds = f.read().splitlines()
    with open(os.path.join(d, "references")) as f:
        refs = f.read().splitlines()
    return preds, refs


def corpus_scores(hyp, ref):
    bleu = sacrebleu.corpus_bleu(hyp, [ref]).score
    chrf2 = sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score
    return bleu, chrf2


def bootstrap_compare(predsA, refsA, predsB, refsB, pool_indices, max_b, rng):
    n = len(pool_indices)
    resample_idx = rng.choice(len(pool_indices), size=(max_b, n), replace=True)
    diffs_bleu = np.empty(max_b, dtype=np.float64)
    diffs_chrf2 = np.empty(max_b, dtype=np.float64)
    for b in range(max_b):
        sel = pool_indices[resample_idx[b]]
        hypA = [predsA[i] for i in sel]
        hypB = [predsB[i] for i in sel]
        ref = [refsA[i] for i in sel]  # refsA == refsB at these indices (same test set)
        bleuA, chrf2A = corpus_scores(hypA, ref)
        bleuB, chrf2B = corpus_scores(hypB, ref)
        diffs_bleu[b] = bleuA - bleuB
        diffs_chrf2[b] = chrf2A - chrf2B
        if (b + 1) % 500 == 0:
            print(f"    resample {b + 1}/{max_b}")
    return diffs_bleu, diffs_chrf2


def summarize(diffs, b):
    d = diffs[:b]
    mean_diff = float(d.mean())
    ci_lo, ci_hi = np.percentile(d, [2.5, 97.5])
    p_le0 = float((d <= 0).mean())
    p_ge0 = float((d >= 0).mean())
    p_one_sided = p_le0 if mean_diff > 0 else p_ge0
    p_two_sided = min(1.0, 2 * min(p_le0, p_ge0))
    return mean_diff, float(ci_lo), float(ci_hi), p_one_sided, p_two_sided


def run_arch(arch, n_per_bucket, bootstrap_sizes, seed, out_root):
    print(f"\n=== {arch} ===")
    ranked = rank_configs(arch)
    if len(ranked) < 2:
        print(f"  only {len(ranked)} scored config(s) found for {arch} -- skipping "
              f"(news_test eval for this architecture may not have finished yet)")
        return
    top2 = ranked[:2]
    print("  config ranking by mean news_test BLEU:")
    for r in ranked:
        marker = " <-- selected" if r in top2 else ""
        print(f"    {r['config']:>8s}  mean_bleu={r['mean_bleu']:.3f}  "
              f"rep_seed={r['rep_seed']}{marker}")

    cfgA, cfgB = top2[0]["config"], top2[1]["config"]
    seedA, seedB = top2[0]["rep_seed"], top2[1]["rep_seed"]
    predsA, refsA = load_run(arch, cfgA, seedA)
    predsB, refsB = load_run(arch, cfgB, seedB)
    assert refsA == refsB, f"{arch}: references differ between {cfgA} and {cfgB} -- not the same test set"

    buckets = load_manifest_buckets()
    rng = np.random.default_rng(seed)
    pool = []
    for bname in sorted(buckets):
        avail = buckets[bname]
        take = min(n_per_bucket, len(avail))
        pool.extend(rng.choice(avail, size=take, replace=False).tolist())
    pool_indices = np.array(sorted(pool))
    print(f"  evaluation pool: {len(pool_indices)} sentences "
          f"({n_per_bucket}/bucket x {len(buckets)} buckets)")
    print(f"  comparing {cfgA} (seed {seedA}) vs {cfgB} (seed {seedB})")

    max_b = max(bootstrap_sizes)
    diffs_bleu, diffs_chrf2 = bootstrap_compare(
        predsA, refsA, predsB, refsB, pool_indices, max_b, rng)

    out_dir = os.path.join(out_root, arch)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{cfgA}_vs_{cfgB}.csv")
    rows = []
    for metric_name, diffs in (("bleu", diffs_bleu), ("chrf2", diffs_chrf2)):
        for b in bootstrap_sizes:
            mean_diff, ci_lo, ci_hi, p1, p2 = summarize(diffs, b)
            rows.append({
                "arch": arch, "config_a": cfgA, "config_b": cfgB,
                "seed_a": seedA, "seed_b": seedB, "metric": metric_name,
                "n_bootstrap": b, "pool_size": len(pool_indices),
                "mean_diff_a_minus_b": round(mean_diff, 4),
                "ci95_lower": round(ci_lo, 4), "ci95_upper": round(ci_hi, 4),
                "p_one_sided": round(p1, 5), "p_two_sided": round(p2, 5),
            })
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out_csv}")

    print(f"\n  {cfgA} (seed {seedA})  vs  {cfgB} (seed {seedB})")
    print(f"  {'metric':<6} {'B':>6} {'mean_diff':>10} {'95% CI':>22} "
          f"{'p(1-sided)':>11} {'p(2-sided)':>11}")
    for row in rows:
        ci = f"[{row['ci95_lower']:.3f}, {row['ci95_upper']:.3f}]"
        print(f"  {row['metric']:<6} {row['n_bootstrap']:>6} {row['mean_diff_a_minus_b']:>10.3f} "
              f"{ci:>22} {row['p_one_sided']:>11.5f} {row['p_two_sided']:>11.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["nllb", "mbart50"])
    ap.add_argument("--n-per-bucket", type=int, default=200)
    ap.add_argument("--bootstrap-sizes", type=int, nargs="+",
                     default=[100, 200, 300, 500, 1000, 1500, 2000, 3000, 5000])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-root", default=os.path.join(RESULTS_ROOT, "significance"))
    args = ap.parse_args()

    for arch in args.archs:
        run_arch(arch, args.n_per_bucket, sorted(args.bootstrap_sizes), args.seed, args.out_root)


if __name__ == "__main__":
    main()
