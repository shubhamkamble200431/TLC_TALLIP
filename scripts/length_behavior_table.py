#!/usr/bin/env python
"""
Reviewer Q10: length-behavior statistics (length ratio, truncation rate,
premature-EOS rate, overgeneration rate, relative length error) by bucket,
across all 16 X-ABCD configs x 3 seeds x {nllb, mbart50} -- speaks more
directly to "learning when to stop" than BLEU alone.

All of this is already computed and saved in every run's own metrics.json
(src/evaluation/metrics.py::length_metrics, called from compute_all for
both the overall test set and each source-length bucket B1..B5) -- this
script only assembles it into one table; no new computation.

Output:
    results/length_behavior/length_behavior_overall.csv   (mean/var over 3 seeds)
    results/length_behavior/length_behavior_by_bucket.csv (mean/var over 3 seeds, per B1..B5)
    results/length_behavior/length_behavior.txt            (plain-text tables, same data)
"""
import _paths  # noqa: F401

import json
import os
import statistics as stats
from collections import defaultdict

from src.config import ALL_CASE_IDS
from src.data.buckets import BUCKET_IDS, BUCKET_LABELS_LONG
from src.utils.io import ensure_dir

ARCHS = [{"name": "nllb", "run_group": "factorial"}, {"name": "mbart50", "run_group": "mbart50"}]
SEEDS = [42, 43, 44]
LEN_METRICS = ["length_ratio", "absolute_length_error", "relative_length_error",
               "premature_eos_rate", "overgeneration_rate", "truncation_rate",
               "zero_length_prediction_rate"]
OUT_ROOT = "results/length_behavior"


def mean_var(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "", ""
    mean = round(stats.mean(vals), 4)
    var = round(stats.variance(vals), 5) if len(vals) > 1 else 0.0
    return mean, var


def load():
    overall = defaultdict(lambda: defaultdict(list))   # [(arch,config)][metric] = [v,v,v]
    by_bucket = defaultdict(lambda: defaultdict(list))  # [(arch,config,bucket)][metric] = [v,v,v]
    for arch in ARCHS:
        for cid in ALL_CASE_IDS:
            for seed in SEEDS:
                p = f"experiments/v2/{arch['run_group']}/{cid}/seed_{seed}/evaluation/v2/unsampled/metrics.json"
                if not os.path.isfile(p):
                    continue
                m = json.load(open(p))["metrics"]
                key = (arch["name"], cid)
                for metric in LEN_METRICS:
                    if metric in m:
                        overall[key][metric].append(m[metric])
                for b in BUCKET_IDS:
                    bkey = (arch["name"], cid, BUCKET_LABELS_LONG[b])
                    for metric in LEN_METRICS:
                        fk = f"{BUCKET_LABELS_LONG[b]}_{metric}"
                        if fk in m:
                            by_bucket[bkey][metric].append(m[fk])
    return overall, by_bucket


def write_csv(path, rows, key_fields):
    import csv
    header = key_fields + [f"{m}_{s}" for m in LEN_METRICS for s in ("mean", "var")]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for key in sorted(rows):
            row = list(key)
            for m in LEN_METRICS:
                row += list(mean_var(rows[key][m]))
            w.writerow(row)


def main():
    ensure_dir(OUT_ROOT)
    overall, by_bucket = load()

    write_csv(os.path.join(OUT_ROOT, "length_behavior_overall.csv"), overall, ["arch", "config"])
    write_csv(os.path.join(OUT_ROOT, "length_behavior_by_bucket.csv"), by_bucket, ["arch", "config", "bucket"])

    txt_path = os.path.join(OUT_ROOT, "length_behavior.txt")
    with open(txt_path, "w") as f:
        def p(s=""):
            print(s)
            f.write(s + "\n")

        p("LENGTH-BEHAVIOR STATISTICS BY BUCKET (reviewer Q10)")
        p("All values: mean over 3 seeds (variance in parens). Source: existing per-run metrics.json,")
        p("src/evaluation/metrics.py::length_metrics -- no new computation, assembled here.")
        p("")
        p("relative_length_error / truncation_rate / premature_eos_rate / overgeneration_rate: lower is better.")
        p("length_ratio: 1.0 = perfect length match; <1.0 = too short, >1.0 = too long.")
        p("")

        for arch in ARCHS:
            p(f"\n{'=' * 130}\n  {arch['name'].upper()} -- OVERALL (all buckets pooled)\n{'=' * 130}")
            hdr = f"  {'config':8s} | " + " | ".join(f"{m:>22s}" for m in LEN_METRICS)
            p(hdr)
            p("  " + "-" * (len(hdr) - 2))
            for cid in ALL_CASE_IDS:
                key = (arch["name"], cid)
                if key not in overall:
                    continue
                cells = []
                for m in LEN_METRICS:
                    mean, var = mean_var(overall[key][m])
                    cells.append(f"{'--':>22s}" if mean == "" else f"{mean:12.4f} (v{var:7.5f})")
                p(f"  {cid:8s} | " + " | ".join(cells))

            for b in BUCKET_IDS:
                blabel = BUCKET_LABELS_LONG[b]
                p(f"\n  -- {arch['name'].upper()} -- bucket {blabel} --")
                p(hdr)
                p("  " + "-" * (len(hdr) - 2))
                for cid in ALL_CASE_IDS:
                    key = (arch["name"], cid, blabel)
                    if key not in by_bucket:
                        continue
                    cells = []
                    for m in LEN_METRICS:
                        mean, var = mean_var(by_bucket[key][m])
                        cells.append(f"{'--':>22s}" if mean == "" else f"{mean:12.4f} (v{var:7.5f})")
                    p(f"  {cid:8s} | " + " | ".join(cells))

    print(f"\nwrote {OUT_ROOT}/length_behavior_overall.csv, "
          f"{OUT_ROOT}/length_behavior_by_bucket.csv, {txt_path}")


if __name__ == "__main__":
    main()
