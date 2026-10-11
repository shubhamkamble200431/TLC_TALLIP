#!/usr/bin/env python
"""
Evaluate completed runs + build the aggregated result CSVs (brief sections 14, 15).

Every run is scored on {test_corpora} x {test_variants}:
    evaluation/<corpus>/<variant>/          corpus in {v2, v1}
                                            variant in {sampled, unsampled}
  sampled   = bucket-balanced test set   (v2: test_7264_5 , v1: test_5414_5)
  unsampled = raw test set               (test.sa / test.hi)

For each (corpus, variant) it writes results_3seed_summary / bucket_results /
run_index CSVs.  The HEADLINE combination (v2 / sampled) goes to results/csv/ ;
every other combination goes to results/csv/<corpus>_<variant>/ .

Examples
--------
  python scripts/evaluate_all.py --dataset v2
  python scripts/evaluate_all.py --dataset v2 --groups factorial mbart50 --no-eval
  python scripts/evaluate_all.py --dataset v2 --test-corpora v2 --test-variants sampled
"""
import _paths  # noqa: F401

import argparse
import glob
import logging
import os
import sys

import yaml

from src.config import RunConfig
from src.evaluation.evaluate import evaluate_test_variants
from src.experiments.status import read_status
from src.statistics.aggregate import write_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("evaluate_all")


def _iter_runs(experiments_root, dataset, group):
    yield from sorted(glob.glob(os.path.join(experiments_root, dataset, group, "X-*", "seed_*")))


def _needs_eval(run_dir, corpora, variants):
    for c in corpora:
        for v in variants:
            if not os.path.isfile(os.path.join(run_dir, "evaluation", c, v, "metrics.json")):
                return True
    return False


def _reeval(run_dir, corpora, variants, device, metrics):
    cfg_path = os.path.join(run_dir, "config.yaml")
    best = os.path.join(run_dir, "checkpoint-best")
    if not (os.path.isfile(cfg_path) and os.path.isdir(best)):
        return False
    cfg = RunConfig.from_dict(yaml.safe_load(open(cfg_path)))
    evaluate_test_variants(cfg, checkpoint_path=best, corpora=corpora, variants=variants,
                           num_beams=cfg.final_eval_num_beams, metrics=metrics, device=device)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="v2")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--groups", nargs="*", default=["factorial"])
    ap.add_argument("--out-dir", default="results/csv")
    ap.add_argument("--test-corpora", nargs="*", default=["v2", "v1"])
    ap.add_argument("--test-variants", nargs="*", default=["unsampled"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true", help="only aggregate; do not (re)run generation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metrics", nargs="*", default=None)
    args = ap.parse_args()

    if not args.no_eval:
        for group in args.groups:
            for run_dir in _iter_runs(args.experiments_root, args.dataset, group):
                st = read_status(run_dir)
                if not st or st.get("status") != "completed":
                    continue
                if not _needs_eval(run_dir, args.test_corpora, args.test_variants):
                    continue
                log.info("[eval] %s — (re)evaluating missing corpus/variant combos", run_dir)
                try:
                    _reeval(run_dir, args.test_corpora, args.test_variants, args.device, args.metrics)
                except Exception as exc:
                    log.exception("re-eval failed for %s: %s", run_dir, exc)

    for group in args.groups:
        for corpus in args.test_corpora:
            for variant in args.test_variants:
                is_v2_unsampled = (corpus == "v2" and variant == "unsampled")
                headline = (group == "factorial" and is_v2_unsampled)
                if headline:
                    out_dir = args.out_dir                                   # results/csv/
                elif group != "factorial" and is_v2_unsampled:
                    out_dir = os.path.join(args.out_dir, group)              # results/csv/mbart50/
                else:
                    tag = f"{corpus}_{variant}" if group == "factorial" else f"{group}_{corpus}_{variant}"
                    out_dir = os.path.join(args.out_dir, tag)               # results/csv/<corpus>_<variant>/ or <group>_<corpus>_<variant>/
                paths = write_all(args.experiments_root, args.dataset, group, out_dir=out_dir,
                                  test_corpus=corpus, test_variant=variant, seeds=args.seeds)
                mark = "  [HEADLINE]" if headline else ""
                print(f"[aggregate] {group}  {corpus}/{variant}{mark}")
                for k, v in paths.items():
                    print(f"    {k:<9} -> {v}")


if __name__ == "__main__":
    main()
