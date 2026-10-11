"""
Three-seed aggregation (brief section 14) and length-bucket collection
(section 15).

Walks the experiment tree, collects per-run test metrics, and emits:

  results/csv/results_3seed_summary.csv   -- one row per configuration
  results/csv/bucket_results.csv          -- one row per (config, seed, bucket)
  results/csv/run_index.csv               -- one row per completed run

All numbers are kept at full precision; rounding is a presentation concern.
"""
from __future__ import annotations

import csv
import glob
import math
import os
from typing import Dict, List, Optional

import numpy as np

from src.utils.io import read_json

try:
    from scipy import stats as _sps
except Exception:  # pragma: no cover
    _sps = None

QUALITY_METRICS = ["bleu", "chrf2", "ter", "meteor", "bertscore", "comet"]
LENGTH_METRICS = [
    "length_ratio",
    "absolute_length_error",
    "relative_length_error",
    "premature_eos_rate",
    "overgeneration_rate",
    "truncation_rate",
]


# ---------------------------------------------------------------------------
def _eval_subdir(test_corpus: str, test_variant: str) -> str:
    return os.path.join("evaluation", test_corpus, test_variant)


def collect_runs(experiments_root: str, dataset_version: str, run_group: str = "factorial",
                 test_corpus: str = "v2", test_variant: str = "unsampled") -> List[Dict]:
    """Run records for every run with an evaluation/<test_corpus>/<test_variant>/metrics.json."""
    sub = _eval_subdir(test_corpus, test_variant)
    pattern = os.path.join(experiments_root, dataset_version, run_group, "X-*", "seed_*")
    records = []
    for run_dir in sorted(glob.glob(pattern)):
        mpath = os.path.join(run_dir, sub, "metrics.json")
        spath = os.path.join(run_dir, "status.json")
        if not os.path.isfile(mpath):
            continue
        status = read_json(spath).get("status") if os.path.isfile(spath) else None
        payload = read_json(mpath)
        case_id = payload.get("case_id") or os.path.basename(os.path.dirname(run_dir))
        seed = payload.get("seed")
        if seed is None:
            base = os.path.basename(run_dir)
            seed = int(base.split("_")[-1]) if base.startswith("seed_") else None
        brpath = os.path.join(run_dir, sub, "bucket_rows.json")
        rec = {
            "run_dir": run_dir,
            "case_id": case_id,
            "seed": seed,
            "status": status,
            "test_corpus": test_corpus,
            "test_variant": test_variant,
            "metrics": payload.get("metrics", {}),
            "bucket_rows": read_json(brpath) if os.path.isfile(brpath) else [],
        }
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
def _ci95(vals: List[float]):
    v = [x for x in vals if x is not None and math.isfinite(x)]
    n = len(v)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)
    mean = float(np.mean(v))
    if n == 1:
        return (mean, 0.0, mean, mean, 1)
    sd = float(np.std(v, ddof=1))
    se = sd / math.sqrt(n)
    if _sps is not None:
        tcrit = float(_sps.t.ppf(0.975, df=n - 1))
    else:
        tcrit = 1.96
    return (mean, sd, mean - tcrit * se, mean + tcrit * se, n)


def aggregate_by_config(records: List[Dict], seeds: Optional[List[int]] = None) -> List[Dict]:
    by_cfg: Dict[str, List[Dict]] = {}
    for r in records:
        if seeds and r["seed"] not in seeds:
            continue
        if r["status"] not in (None, "completed"):
            continue
        by_cfg.setdefault(r["case_id"], []).append(r)

    rows = []
    for case_id, runs in sorted(by_cfg.items()):
        row = {"config": case_id, "seed_count": len(runs),
               "seeds": ",".join(str(x["seed"]) for x in sorted(runs, key=lambda z: z["seed"] or 0))}
        for m in QUALITY_METRICS + LENGTH_METRICS:
            vals = [run["metrics"].get(m) for run in runs]
            mean, sd, lo, hi, n = _ci95(vals)
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = sd
            row[f"{m}_ci_low"] = lo
            row[f"{m}_ci_high"] = hi
        rows.append(row)
    return rows


def bucket_rows_long(records: List[Dict]) -> List[Dict]:
    out = []
    for r in records:
        for br in r["bucket_rows"]:
            row = {"config": r["case_id"], "seed": r["seed"], "bucket": br.get("bucket"),
                   "n_examples": br.get("n_examples", 0)}
            for m in QUALITY_METRICS + LENGTH_METRICS:
                if m in br:
                    row[m] = br[m]
            out.append(row)
    return out


# ---------------------------------------------------------------------------
def _write_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        open(path, "w").close()
        return
    keys = list({k for row in rows for k in row})
    # stable, readable column order
    preferred = ["config", "seed", "bucket", "n_examples", "seed_count", "seeds"]
    keys = [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_all(
    experiments_root: str,
    dataset_version: str,
    run_group: str,
    out_dir: str = "results/csv",
    test_corpus: str = "v2",
    test_variant: str = "unsampled",
    seeds: Optional[List[int]] = None,
) -> Dict[str, str]:
    records = collect_runs(experiments_root, dataset_version, run_group, test_corpus, test_variant)
    summary = aggregate_by_config(records, seeds=seeds)
    buckets = bucket_rows_long(records)
    index = [
        {"config": r["case_id"], "seed": r["seed"], "status": r["status"],
         "run_dir": r["run_dir"], **{m: r["metrics"].get(m) for m in QUALITY_METRICS}}
        for r in records
    ]
    paths = {
        "summary": os.path.join(out_dir, "results_3seed_summary.csv"),
        "buckets": os.path.join(out_dir, "bucket_results.csv"),
        "index": os.path.join(out_dir, "run_index.csv"),
    }
    _write_csv(paths["summary"], summary)
    _write_csv(paths["buckets"], buckets)
    _write_csv(paths["index"], index)
    return paths
