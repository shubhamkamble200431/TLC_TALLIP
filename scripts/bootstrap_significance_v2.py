"""
Publication-grade paired bootstrap significance test, superseding
scripts/bootstrap_significance.py (which used a 1000-sentence subsample and
only BLEU/chrF2). This version:

  * Uses the FULL 2500-sentence news_test corpus, stratified into its five
    existing length buckets (500 sentences/bucket) -- every bootstrap
    resample independently redraws 500-with-replacement WITHIN each bucket,
    preserving the original bucket composition exactly.
  * Tests all six metrics: BLEU, chrF2, TER, METEOR, BERTScore, COMET.
      - BLEU/chrF2/TER are genuine corpus-level metrics (not a mean of
        per-sentence scores), so they're recomputed per resample via
        sacrebleu's sufficient-statistics API (_extract_corpus_statistics
        once per system, then _compute_score_from_stats on the resampled
        SUM of per-sentence stats -- no re-tokenization per resample).
      - METEOR/BERTScore/COMET, AS IMPLEMENTED IN THIS REPO
        (src/evaluation/metrics.py), are literally the mean of a
        per-sentence score, so each is computed once per sentence, then
        bootstrapped by resampling+averaging that precomputed array.
  * Bootstrap sizes up to 50000 (one draw, nested prefixes for the smaller
    reported sizes: 1000/5000/10000/15000/20000/30000/40000/50000 -- six
    values strictly above 10000, satisfying "at least 5 numbers above
    10000").
  * Direction: diff = score(A) - score(B) for every metric; TER is
    lower-is-better, flagged explicitly everywhere it's interpreted.
  * Repeats every comparison per seed (42/43/44) using that seed's own
    predictions for BOTH systems (never mixes seeds within one resample),
    then reports a seed-level aggregate as a DESCRIPTIVE summary only
    (observed diff per seed, mean/std across the 3 seeds) -- explicitly NOT
    a statistical test, since n=3 seeds has no meaningful power and must
    not be confused with the n=50000 bootstrap-over-sentences distribution.
  * Holm-Bonferroni correction applied two ways: within each (arch, seed)
    family of 6 metrics, and globally across all 2*3*6=36 tests.
  * Comparisons (fixed, NOT re-selected here): nllb X-1110 vs X-0110;
    mbart50 X-0100 vs X-0110 -- i.e. the same top-2-by-mean-BLEU configs
    the original (unauthorized) bootstrap script auto-selected. This is a
    DELIBERATE replication of that selection, not a re-derivation, and the
    selection-bias caveat below applies.

SELECTION BIAS: these 4 configs (2 per architecture) were chosen because
they ranked #1/#2 by mean news_test BLEU among the SAME 16 candidates the
p-values below are computed on. Comparing only the best two configs out of
16 is itself a selection step ("winner's curse" / post-selection inference):
it inflates the apparent gap and the p-values here condition on having
already observed that gap. They should be read as "conditional on these
being the top-2, how much does sentence-resampling noise alone explain the
gap we see" -- NOT as if these two configs were specified before seeing any
news_test score. No post-selection correction (e.g. sample splitting,
Berk et al. 2013 PoSI) is applied; this is a documented limitation, not an
oversight.
"""
import _paths  # noqa: F401

import argparse
import json
import os
import statistics as stats
from collections import defaultdict

import numpy as np
from sacrebleu.metrics import BLEU, CHRF, TER

from src.utils.io import ensure_dir, read_lines

NEWS_DIR = "data/news_test"
MANIFEST = os.path.join(NEWS_DIR, "test_manifest.tsv")
RESULTS_ROOT = "results/news_test"
OUT_ROOT = "results/news_test/significance_v2"
SEEDS = [42, 43, 44]
BOOTSTRAP_SIZES = [1000, 5000, 10000, 15000, 20000, 30000, 40000, 50000]
MAX_B = max(BOOTSTRAP_SIZES)
RNG_SEED = 20261001  # fixed bootstrap RNG seed, documented, distinct from training seeds 42/43/44
N_PER_BUCKET_FULL = 500  # news_test has exactly 500/bucket; this uses ALL of them, every bucket

COMPARISONS = [
    {"arch": "nllb", "lang_pair": "sa-hi", "config_a": "X-1110", "config_b": "X-0110"},
    {"arch": "mbart50", "lang_pair": "sa-hi", "config_a": "X-0100", "config_b": "X-0110"},
]
LOWER_IS_BETTER = {"ter"}
METRICS = ["bleu", "chrf2", "ter", "meteor", "bertscore", "comet"]


# ---------------------------------------------------------------------------
# bucket pool
# ---------------------------------------------------------------------------
def load_bucket_pool():
    buckets = defaultdict(list)
    with open(MANIFEST) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx_i, bucket_i = header.index("index"), header.index("bucket")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            idx0 = int(parts[idx_i]) - 1
            buckets[parts[bucket_i]].append(idx0)
    for b, idxs in buckets.items():
        assert len(idxs) == N_PER_BUCKET_FULL, f"bucket {b} has {len(idxs)} sentences, expected {N_PER_BUCKET_FULL}"
    return buckets


def stratified_resample_indices(buckets, n_resamples, rng):
    """(n_resamples, 2500) int array; each row independently redraws
    500-with-replacement from WITHIN every bucket, concatenated."""
    per_bucket = []
    for b in sorted(buckets):
        idxs = np.array(buckets[b])
        draw = rng.choice(idxs, size=(n_resamples, len(idxs)), replace=True)
        per_bucket.append(draw)
    return np.concatenate(per_bucket, axis=1)


# ---------------------------------------------------------------------------
# per-sentence sufficient statistics / scores
# ---------------------------------------------------------------------------
def load_run(arch, lang_pair, config, seed):
    d = os.path.join(RESULTS_ROOT, arch, lang_pair, config, f"seed_{seed}")
    preds = read_lines(os.path.join(d, "predictions"))
    refs = read_lines(os.path.join(d, "references"))
    srcs = read_lines(os.path.join(d, "source"))
    return preds, refs, srcs


def per_sentence_meteor(preds, refs):
    import nltk
    from nltk.translate.meteor_score import meteor_score
    for pkg in ("wordnet", "omw-1.4", "punkt"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return np.array([meteor_score([r.split()], p.split()) for p, r in zip(preds, refs)], dtype="float64") * 100.0


def per_sentence_bertscore(preds, refs, lang="hi"):
    from bert_score import score as bertscore
    _, _, f1 = bertscore(preds, refs, lang=lang, verbose=False)
    return f1.numpy().astype("float64") * 100.0


def per_sentence_comet(preds, refs, srcs):
    from comet import download_model, load_from_checkpoint
    import torch
    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(srcs, preds, refs)]
    out = model.predict(data, batch_size=64, gpus=1 if torch.cuda.is_available() else 0)
    return np.array(out.scores, dtype="float64") * 100.0


class SystemStats:
    """All per-sentence sufficient statistics / scores for one (arch, config,
    seed) run, precomputed ONCE, reused across every bootstrap resample."""

    def __init__(self, arch, lang_pair, config, seed):
        preds, refs, srcs = load_run(arch, lang_pair, config, seed)
        self.preds, self.refs, self.srcs = preds, refs, srcs
        self.n = len(preds)

        self._bleu_metric = BLEU(tokenize="flores200")
        self._chrf_metric = CHRF(word_order=2)
        self._ter_metric = TER()
        self.bleu_stats = np.array(self._bleu_metric._extract_corpus_statistics(preds, [refs]), dtype="float64")
        self.chrf2_stats = np.array(self._chrf_metric._extract_corpus_statistics(preds, [refs]), dtype="float64")
        self.ter_stats = np.array(self._ter_metric._extract_corpus_statistics(preds, [refs]), dtype="float64")

        print(f"    [{arch}/{config}/seed_{seed}] computing per-sentence METEOR/BERTScore/COMET "
              f"({self.n} sentences, one-time cost)...")
        self.meteor_scores = per_sentence_meteor(preds, refs)
        self.bertscore_scores = per_sentence_bertscore(preds, refs, lang="hi")
        self.comet_scores = per_sentence_comet(preds, refs, srcs)

    def observed_score(self, metric):
        if metric == "bleu":
            return self._bleu_metric._compute_score_from_stats(self.bleu_stats.sum(0)).score
        if metric == "chrf2":
            return self._chrf_metric._compute_score_from_stats(self.chrf2_stats.sum(0)).score
        if metric == "ter":
            return self._ter_metric._compute_score_from_stats(self.ter_stats.sum(0)).score
        if metric == "meteor":
            return float(self.meteor_scores.mean())
        if metric == "bertscore":
            return float(self.bertscore_scores.mean())
        if metric == "comet":
            return float(self.comet_scores.mean())
        raise ValueError(metric)

    def resampled_scores(self, metric, idxs):
        """idxs: (n_resamples, 2500) int array -> (n_resamples,) score array."""
        if metric == "bleu":
            summed = self.bleu_stats[idxs].sum(axis=1)
            return np.array([self._bleu_metric._compute_score_from_stats(row).score for row in summed])
        if metric == "chrf2":
            summed = self.chrf2_stats[idxs].sum(axis=1)
            return np.array([self._chrf_metric._compute_score_from_stats(row).score for row in summed])
        if metric == "ter":
            summed = self.ter_stats[idxs].sum(axis=1)
            return np.array([self._ter_metric._compute_score_from_stats(row).score for row in summed])
        if metric == "meteor":
            return self.meteor_scores[idxs].mean(axis=1)
        if metric == "bertscore":
            return self.bertscore_scores[idxs].mean(axis=1)
        if metric == "comet":
            return self.comet_scores[idxs].mean(axis=1)
        raise ValueError(metric)


# ---------------------------------------------------------------------------
# bootstrap + summary stats
# ---------------------------------------------------------------------------
def summarize_diffs(diffs, b):
    d = diffs[:b]
    mean_diff = float(d.mean())
    ci_lo, ci_hi = np.percentile(d, [2.5, 97.5])
    p_le0 = float((d <= 0).mean())
    p_ge0 = float((d >= 0).mean())
    p_one_sided = p_le0 if mean_diff > 0 else p_ge0
    p_two_sided = min(1.0, 2 * min(p_le0, p_ge0))
    return mean_diff, float(ci_lo), float(ci_hi), p_one_sided, p_two_sided


def holm_bonferroni(pvals):
    """Returns adjusted p-values (same order as input), Holm's step-down method."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = (m - rank) * pvals[idx]
        running_max = max(running_max, corrected)
        adj[idx] = min(1.0, running_max)
    return adj


def run_comparison(comp, buckets, rng):
    arch, lang_pair = comp["arch"], comp["lang_pair"]
    cfg_a, cfg_b = comp["config_a"], comp["config_b"]
    print(f"\n=== {arch}: {cfg_a} vs {cfg_b} ===")

    per_seed_rows = []
    for seed in SEEDS:
        print(f"  -- seed {seed} --")
        sys_a = SystemStats(arch, lang_pair, cfg_a, seed)
        sys_b = SystemStats(arch, lang_pair, cfg_b, seed)
        assert sys_a.refs == sys_b.refs, f"{arch}/seed_{seed}: references differ between {cfg_a} and {cfg_b}"

        idxs = stratified_resample_indices(buckets, MAX_B, rng)

        for metric in METRICS:
            scores_a = sys_a.resampled_scores(metric, idxs)
            scores_b = sys_b.resampled_scores(metric, idxs)
            diffs = scores_a - scores_b
            observed_diff = sys_a.observed_score(metric) - sys_b.observed_score(metric)

            for b in BOOTSTRAP_SIZES:
                mean_diff, ci_lo, ci_hi, p1, p2 = summarize_diffs(diffs, b)
                per_seed_rows.append({
                    "arch": arch, "config_a": cfg_a, "config_b": cfg_b, "seed": seed,
                    "metric": metric, "lower_is_better": metric in LOWER_IS_BETTER,
                    "n_bootstrap": b, "observed_diff_a_minus_b": round(observed_diff, 4),
                    "bootstrap_mean_diff": round(mean_diff, 4),
                    "ci95_lower": round(ci_lo, 4), "ci95_upper": round(ci_hi, 4),
                    "p_one_sided": round(p1, 6), "p_two_sided": round(p2, 6),
                })
            print(f"    {metric:<10s} observed_diff(A-B)={observed_diff:+.4f}  "
                  f"(at B={MAX_B}: p_two_sided={per_seed_rows[-1]['p_two_sided']:.5f})")
    return per_seed_rows


def aggregate_across_seeds(all_rows, comp):
    """Descriptive-only aggregate at the max bootstrap size: mean/std of the
    OBSERVED (not bootstrap-mean) per-seed diff across the 3 training seeds.
    This is NOT a statistical test -- n=3 has no real power; see module
    docstring."""
    arch = comp["arch"]
    rows_at_maxb = [r for r in all_rows if r["arch"] == arch and r["n_bootstrap"] == MAX_B]
    out = []
    for metric in METRICS:
        vals = [r["observed_diff_a_minus_b"] for r in rows_at_maxb if r["metric"] == metric]
        out.append({
            "arch": arch, "config_a": comp["config_a"], "config_b": comp["config_b"], "metric": metric,
            "lower_is_better": metric in LOWER_IS_BETTER, "n_seeds": len(vals),
            "seed_diffs": vals,
            "mean_observed_diff": round(stats.mean(vals), 4),
            "std_observed_diff": round(stats.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "all_seeds_agree_in_sign": len({v > 0 for v in vals}) == 1 if vals else None,
        })
    return out


def apply_corrections(all_rows):
    for r in all_rows:
        r["p_two_sided_holm_within_arch_seed"] = None
        r["p_two_sided_holm_global"] = None
    maxb_rows = [r for r in all_rows if r["n_bootstrap"] == MAX_B]

    # (a) within each (arch, seed): Holm across the 6 metrics
    by_group = defaultdict(list)
    for r in maxb_rows:
        by_group[(r["arch"], r["seed"])].append(r)
    for group_rows in by_group.values():
        pvals = np.array([r["p_two_sided"] for r in group_rows])
        adj = holm_bonferroni(pvals)
        for r, a in zip(group_rows, adj):
            r["p_two_sided_holm_within_arch_seed"] = round(float(a), 6)

    # (b) globally across all (arch, seed, metric) = 36 tests
    pvals_global = np.array([r["p_two_sided"] for r in maxb_rows])
    adj_global = holm_bonferroni(pvals_global)
    for r, a in zip(maxb_rows, adj_global):
        r["p_two_sided_holm_global"] = round(float(a), 6)


def main():
    global BOOTSTRAP_SIZES, MAX_B

    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-sizes", type=int, nargs="+", default=BOOTSTRAP_SIZES)
    args = ap.parse_args()
    BOOTSTRAP_SIZES = sorted(args.bootstrap_sizes)
    MAX_B = max(BOOTSTRAP_SIZES)

    ensure_dir(OUT_ROOT)
    buckets = load_bucket_pool()
    rng = np.random.default_rng(RNG_SEED)

    all_rows = []
    aggregates = []
    for comp in COMPARISONS:
        rows = run_comparison(comp, buckets, rng)
        all_rows.extend(rows)
        aggregates.extend(aggregate_across_seeds(rows, comp))

    apply_corrections(all_rows)

    import csv
    full_csv = os.path.join(OUT_ROOT, "bootstrap_full.csv")
    with open(full_csv, "w", newline="") as f:
        fieldnames = list(all_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {full_csv} ({len(all_rows)} rows)")

    agg_csv = os.path.join(OUT_ROOT, "seed_aggregate.csv")
    with open(agg_csv, "w", newline="") as f:
        fieldnames = [k for k in aggregates[0].keys() if k != "seed_diffs"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for a in aggregates:
            w.writerow({k: v for k, v in a.items() if k != "seed_diffs"})
    print(f"wrote {agg_csv}")

    with open(os.path.join(OUT_ROOT, "all_rows.json"), "w") as f:
        json.dump({"rows": all_rows, "aggregates": aggregates,
                    "bootstrap_sizes": BOOTSTRAP_SIZES, "rng_seed": RNG_SEED,
                    "n_per_bucket": N_PER_BUCKET_FULL}, f, indent=2)

    # -- publication table at B=max, Holm-corrected --------------------------
    print(f"\n{'=' * 130}\n  PUBLICATION TABLE  (B={MAX_B} bootstrap resamples, stratified by length bucket, "
          f"news_test full 2500 sentences)\n{'=' * 130}")
    hdr = (f"  {'arch':8s} {'seed':5s} {'metric':9s} {'A-B diff':>9s} {'95% CI':>20s} "
           f"{'p (raw)':>9s} {'p (Holm/arch-seed)':>19s} {'p (Holm/global)':>16s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sorted((x for x in all_rows if x["n_bootstrap"] == MAX_B),
                    key=lambda r: (r["arch"], r["seed"], r["metric"])):
        ci = f"[{r['ci95_lower']:.3f}, {r['ci95_upper']:.3f}]"
        lib = " (lower better)" if r["lower_is_better"] else ""
        print(f"  {r['arch']:8s} {r['seed']:<5d} {r['metric']+lib:24s} {r['observed_diff_a_minus_b']:>9.3f} "
              f"{ci:>20s} {r['p_two_sided']:>9.5f} {r['p_two_sided_holm_within_arch_seed']:>19.5f} "
              f"{r['p_two_sided_holm_global']:>16.5f}")

    print(f"\n{'=' * 100}\n  SEED-LEVEL AGGREGATE (descriptive only, n=3 seeds, NOT a statistical test)\n{'=' * 100}")
    for a in aggregates:
        lib = " (lower better)" if a["lower_is_better"] else ""
        print(f"  {a['arch']:8s} {a['metric']+lib:24s} seed_diffs={[round(v, 3) for v in a['seed_diffs']]}  "
              f"mean={a['mean_observed_diff']:+.3f}  std={a['std_observed_diff']:.3f}  "
              f"all_seeds_agree_in_sign={a['all_seeds_agree_in_sign']}")
    print()


if __name__ == "__main__":
    main()
